"""Parse scheme names, pair Direct<->Regular plans, and compute return drag.

The whole analysis rests on one idea: a fund's Direct and Regular plans hold an
*identical* portfolio, so the gap in their realised returns is a clean measure of
the recurring distributor commission ("fee drag"). To exploit that we must:

  1. parse each messy scheme name into (fund_house-ish base, plan, option),
  2. match the Direct-Growth plan to its Regular-Growth twin,
  3. compute each plan's annualised return over their *common* date window,
  4. take the difference.

Every non-obvious decision here is documented in notebook 02_cleaning.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# --- token vocabularies -----------------------------------------------------

# Anything signalling an income/dividend share class -> NOT a Growth plan.
# (Dividend payouts distort NAV-based returns, so we keep Growth plans only.)
_DIVIDEND_TOKENS = (
    "idcw", "dividend", "income distribution", "payout", "reinvest",
    "bonus", "daily", "weekly", "fortnightly", "monthly", "quarterly",
    "half yearly", "annual dividend",
)

# Tokens stripped when building the match key. These are exactly the parts that
# DIFFER between a fund's two plans (plan + option), plus pure noise words.
_NOISE_PATTERNS = [
    r"\bdirect\b", r"\bregular\b", r"\bplan\b", r"\bgrowth\b", r"\boption\b",
    r"\bidcw\b", r"\bdividend\b", r"\bpayout\b", r"\breinvest\w*\b",
    r"\bincome distribution( cum capital withdrawal)?\b",
    r"\boption\b", r"\bcum\b", r"\bappreciation\b",
]


def classify_plan(name: str) -> str:
    """Return 'Direct', 'Regular', or 'Unknown' for a scheme name."""
    low = name.lower()
    if "direct" in low:
        return "Direct"
    if "regular" in low:
        return "Regular"
    return "Unknown"


def classify_option(name: str) -> str:
    """Return 'Growth', 'IDCW', or 'Unknown' for a scheme name."""
    low = name.lower()
    if any(tok in low for tok in _DIVIDEND_TOKENS):
        return "IDCW"
    if "growth" in low:
        return "Growth"
    return "Unknown"


def normalize_base(name: str) -> str:
    """Collapse a scheme name to a plan/option-independent match key.

    Drops parenthetical asides and 'formerly known as ...' clauses (which appear
    inconsistently across a fund's two plans and would otherwise break matching),
    then strips plan/option tokens and non-alphanumerics.
    """
    low = name.lower()
    low = re.sub(r"\(.*?\)", " ", low)                 # drop "(erstwhile ...)"
    low = re.sub(r"formerly known as.*$", " ", low)     # drop rename clauses
    low = re.sub(r"formerly.*$", " ", low)
    for pat in _NOISE_PATTERNS:
        low = re.sub(pat, " ", low)
    low = re.sub(r"[^a-z0-9]+", " ", low)               # punctuation -> space
    low = re.sub(r"\s+", " ", low).strip()
    return low


def clean_fund_label(name: str) -> str:
    """Human-readable fund name: drop everything from the plan token onward.

    Robust to the many separators AMFI uses ('- Direct Plan -', '- Direct',
    'DIRECT PLAN', '-Direct-', ...) by cutting at the first plan keyword.
    """
    m = re.search(r"\b(direct|regular)\b", name, flags=re.IGNORECASE)
    label = name[: m.start()] if m else name
    return re.sub(r"[-\s]+$", "", label).strip(" -")


def build_pairs(raw_scheme_list: list[dict]) -> pd.DataFrame:
    """From the raw AMFI scheme list, return matched Direct<->Regular Growth pairs.

    Conservative matching rule (documented in nb 02): a base key qualifies only if
    it maps to *exactly one* Direct-Growth scheme and *exactly one* Regular-Growth
    scheme. Ambiguous bases (multiple candidates, legacy plans missing an explicit
    plan token) are dropped in favour of correctness over coverage.
    """
    rows = []
    for s in raw_scheme_list:
        name = s["schemeName"]
        rows.append({
            "scheme_code": s["schemeCode"],
            "scheme_name": name,
            "plan": classify_plan(name),
            "option": classify_option(name),
            "base_key": normalize_base(name),
        })
    df = pd.DataFrame(rows)

    growth = df[(df["option"] == "Growth") & (df["plan"].isin(["Direct", "Regular"]))]

    direct = growth[growth["plan"] == "Direct"]
    regular = growth[growth["plan"] == "Regular"]

    # Keep only base keys with exactly one candidate on each side.
    d_counts = direct["base_key"].value_counts()
    r_counts = regular["base_key"].value_counts()
    unique_d = set(d_counts[d_counts == 1].index)
    unique_r = set(r_counts[r_counts == 1].index)
    shared = (unique_d & unique_r) - {""}

    d_lookup = direct[direct["base_key"].isin(shared)].set_index("base_key")
    r_lookup = regular[regular["base_key"].isin(shared)].set_index("base_key")

    pairs = pd.DataFrame({
        "base_key": sorted(shared),
    })
    pairs["direct_code"] = pairs["base_key"].map(d_lookup["scheme_code"])
    pairs["direct_name"] = pairs["base_key"].map(d_lookup["scheme_name"])
    pairs["regular_code"] = pairs["base_key"].map(r_lookup["scheme_code"])
    pairs["regular_name"] = pairs["base_key"].map(r_lookup["scheme_name"])
    pairs["fund_name"] = pairs["direct_name"].map(clean_fund_label)
    return pairs.sort_values("fund_name").reset_index(drop=True)


# --- NAV series & returns ---------------------------------------------------

def nav_payload_to_series(payload: dict) -> pd.Series:
    """Convert a /mf/{code} payload into a clean, ascending NAV Series.

    Drops zero/non-positive NAVs (data errors) and duplicate dates.
    """
    data = payload["data"]
    dates = pd.to_datetime([d["date"] for d in data], format="%d-%m-%Y")
    navs = pd.to_numeric([d["nav"] for d in data], errors="coerce")
    s = pd.Series(navs, index=dates, name="nav")
    s = s[(s > 0) & s.notna()]
    s = s[~s.index.duplicated(keep="first")]
    return s.sort_index()


def annualized_return(series: pd.Series, start: pd.Timestamp,
                      end: pd.Timestamp) -> float | None:
    """CAGR of a NAV series between the first point >= start and last point <= end."""
    head = series[series.index >= start]
    tail = series[series.index <= end]
    if head.empty or tail.empty:
        return None
    start_date, start_val = head.index[0], head.iloc[0]
    end_date, end_val = tail.index[-1], tail.iloc[-1]
    years = (end_date - start_date).days / 365.25
    if years <= 0 or start_val <= 0:
        return None
    return (end_val / start_val) ** (1.0 / years) - 1.0


def compute_pair_metrics(direct_payload: dict, regular_payload: dict,
                         min_years: float = 2.0) -> dict | None:
    """Return drag metrics for one fund, or None if the window is too short.

    drag_pp = direct_cagr - regular_cagr, in percentage points/yr. Positive means
    Direct earned more, i.e. that gap is the annual commission the Regular plan paid.
    """
    d = nav_payload_to_series(direct_payload)
    r = nav_payload_to_series(regular_payload)
    if d.empty or r.empty:
        return None

    start = max(d.index[0], r.index[0])
    end = min(d.index[-1], r.index[-1])
    years = (end - start).days / 365.25
    if years < min_years:
        return None

    d_cagr = annualized_return(d, start, end)
    r_cagr = annualized_return(r, start, end)
    if d_cagr is None or r_cagr is None:
        return None

    meta = direct_payload.get("meta", {})
    return {
        "fund_house": meta.get("fund_house"),
        "scheme_type": meta.get("scheme_type"),
        "scheme_category": meta.get("scheme_category"),
        "window_start": start.date().isoformat(),
        "window_end": end.date().isoformat(),
        "window_years": round(years, 2),
        "direct_cagr": d_cagr,
        "regular_cagr": r_cagr,
        "drag_pp": (d_cagr - r_cagr) * 100.0,
    }


# --- investor-facing cost translation ---------------------------------------

def sip_future_value(monthly: float, annual_return: float, years: int) -> float:
    """Future value of a monthly SIP compounded monthly."""
    r = annual_return / 12.0
    n = years * 12
    if r == 0:
        return monthly * n
    return monthly * (((1 + r) ** n - 1) / r) * (1 + r)


def sip_drag_cost(monthly: float, gross_annual_return: float, drag_pp: float,
                  years: int) -> dict:
    """Rupee gap between Direct and Regular outcomes for a monthly SIP.

    gross_annual_return is the Direct (no-commission) return; the Regular plan
    earns drag_pp percentage points less each year.
    """
    direct_fv = sip_future_value(monthly, gross_annual_return, years)
    regular_fv = sip_future_value(monthly, gross_annual_return - drag_pp / 100.0, years)
    invested = monthly * years * 12
    return {
        "invested": invested,
        "direct_fv": direct_fv,
        "regular_fv": regular_fv,
        "lost_to_drag": direct_fv - regular_fv,
        "lost_pct_of_corpus": (direct_fv - regular_fv) / direct_fv * 100.0,
    }


def categorize_broad(scheme_category: str | float) -> str:
    """Map AMFI's granular scheme_category to a broad bucket for grouping."""
    if not isinstance(scheme_category, str):
        return "Unknown"
    c = scheme_category.lower()
    if "equity" in c or "elss" in c:
        return "Equity"
    if "debt" in c or "duration" in c or "bond" in c or "gilt" in c or "money market" in c:
        return "Debt"
    if "hybrid" in c or "balanced" in c or "solution" in c:
        return "Hybrid"
    if "other" in c or "index" in c or "etf" in c or "fof" in c or "fund of funds" in c:
        return "Other/Index"
    return "Other/Index"
