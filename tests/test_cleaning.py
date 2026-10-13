"""Unit tests for the scheme-name parsing, pairing and return-drag maths.

Everything here is pure and offline: no AMFI/mfapi calls, no files read.
"""
import math

import pandas as pd
import pytest

from src.cleaning import (
    annualized_return,
    build_pairs,
    categorize_broad,
    classify_option,
    classify_plan,
    clean_fund_label,
    compute_pair_metrics,
    nav_payload_to_series,
    normalize_base,
    sip_drag_cost,
    sip_future_value,
)


# --- name parsing -------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("Axis Bluechip Fund - Direct Plan - Growth", "Direct"),
    ("Axis Bluechip Fund - Regular Plan - Growth", "Regular"),
    ("HDFC TOP 100 FUND - DIRECT PLAN - GROWTH", "Direct"),
    ("Some Legacy Fund - Growth", "Unknown"),
])
def test_classify_plan(name, expected):
    assert classify_plan(name) == expected


@pytest.mark.parametrize("name,expected", [
    ("Axis Bluechip Fund - Direct Plan - Growth", "Growth"),
    ("Axis Bluechip Fund - Direct Plan - IDCW", "IDCW"),
    ("Fund - Regular - Dividend Payout", "IDCW"),
    ("Fund - Monthly Dividend Reinvestment", "IDCW"),
    ("Fund - Direct - Growth - Daily IDCW", "IDCW"),
    ("Fund - Direct Plan", "Unknown"),
])
def test_classify_option(name, expected):
    assert classify_option(name) == expected


def test_normalize_base_strips_plan_and_option_tokens():
    direct = "Axis Bluechip Fund - Direct Plan - Growth Option"
    regular = "Axis Bluechip Fund - Regular Plan - Growth Option"
    assert normalize_base(direct) == normalize_base(regular) == "axis bluechip fund"


def test_normalize_base_drops_parentheticals_and_rename_clauses():
    a = "ABC Flexi Cap Fund (erstwhile ABC Multi Cap) - Direct - Growth"
    b = "ABC Flexi Cap Fund - Regular - Growth (formerly known as ABC Multi Cap Fund)"
    assert normalize_base(a) == normalize_base(b) == "abc flexi cap fund"


@pytest.mark.parametrize("name,expected", [
    ("Axis Bluechip Fund - Direct Plan - Growth", "Axis Bluechip Fund"),
    ("HDFC TOP 100 FUND-Direct-Growth", "HDFC TOP 100 FUND"),
    ("Kotak Gilt Fund DIRECT PLAN GROWTH", "Kotak Gilt Fund"),
    ("No Plan Token Fund", "No Plan Token Fund"),
])
def test_clean_fund_label(name, expected):
    assert clean_fund_label(name) == expected


# --- pairing --------------------------------------------------------------------

def _scheme(code, name):
    return {"schemeCode": code, "schemeName": name}


def test_build_pairs_matches_one_direct_to_one_regular_growth():
    schemes = [
        _scheme(1, "Axis Bluechip Fund - Direct Plan - Growth"),
        _scheme(2, "Axis Bluechip Fund - Regular Plan - Growth"),
        _scheme(3, "Axis Bluechip Fund - Direct Plan - IDCW"),      # not Growth
        _scheme(4, "Axis Bluechip Fund - Regular Plan - IDCW"),
    ]
    pairs = build_pairs(schemes)
    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert row["direct_code"] == 1
    assert row["regular_code"] == 2
    assert row["fund_name"] == "Axis Bluechip Fund"


def test_build_pairs_drops_ambiguous_and_unmatched_bases():
    schemes = [
        # Two Direct-Growth candidates for the same base -> ambiguous, dropped.
        _scheme(10, "Dup Fund - Direct Plan - Growth"),
        _scheme(11, "Dup Fund - Direct - Growth"),
        _scheme(12, "Dup Fund - Regular Plan - Growth"),
        # Direct only, no Regular twin -> dropped.
        _scheme(20, "Lonely Fund - Direct Plan - Growth"),
        # A clean pair that must survive.
        _scheme(30, "Good Fund - Direct Plan - Growth"),
        _scheme(31, "Good Fund - Regular Plan - Growth"),
    ]
    pairs = build_pairs(schemes)
    assert pairs["fund_name"].tolist() == ["Good Fund"]



# --- NAV series and returns -----------------------------------------------------

def _payload(points, **meta):
    """points: list of (dd-mm-yyyy, nav) in whatever order."""
    return {"meta": meta, "data": [{"date": d, "nav": str(n)} for d, n in points]}


def test_nav_payload_to_series_sorts_and_drops_bad_rows():
    payload = _payload([
        ("03-01-2020", 12.0),
        ("01-01-2020", 10.0),
        ("02-01-2020", 0),         # non-positive -> dropped
        ("01-01-2020", 99.0),      # duplicate date -> first kept
        ("04-01-2020", "abc"),     # unparseable -> dropped
    ])
    s = nav_payload_to_series(payload)
    assert list(s.index) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-03")]
    assert s.tolist() == [10.0, 12.0]


def test_annualized_return_is_cagr_over_window():
    s = pd.Series(
        [100.0, 121.0],
        index=[pd.Timestamp("2020-01-01"), pd.Timestamp("2022-01-01")],
    )
    cagr = annualized_return(s, s.index[0], s.index[-1])
    # 21% over ~2 years -> ~10%/yr (365.25-day years, so not exactly 0.10)
    assert cagr == pytest.approx(0.10, abs=1e-3)


def test_annualized_return_returns_none_outside_window():
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-01", "2021-01-01"]))
    assert annualized_return(s, pd.Timestamp("2022-01-01"), pd.Timestamp("2023-01-01")) is None


def test_compute_pair_metrics_measures_direct_minus_regular_drag():
    direct = _payload(
        [("01-01-2015", 10.0), ("01-01-2025", 10.0 * 1.12 ** 10)],
        fund_house="X AMC", scheme_type="Open Ended", scheme_category="Equity Scheme - Large Cap",
    )
    regular = _payload([("01-01-2015", 10.0), ("01-01-2025", 10.0 * 1.10 ** 10)])
    m = compute_pair_metrics(direct, regular)
    assert m is not None
    assert m["fund_house"] == "X AMC"
    assert m["window_start"] == "2015-01-01"
    assert m["window_end"] == "2025-01-01"
    assert m["direct_cagr"] == pytest.approx(0.12, abs=1e-3)
    assert m["regular_cagr"] == pytest.approx(0.10, abs=1e-3)
    assert m["drag_pp"] == pytest.approx(2.0, abs=0.1)


def test_compute_pair_metrics_uses_common_window():
    direct = _payload([("01-01-2010", 5.0), ("01-01-2020", 10.0), ("01-01-2025", 20.0)])
    regular = _payload([("01-01-2020", 10.0), ("01-01-2025", 20.0)])
    m = compute_pair_metrics(direct, regular)
    assert m["window_start"] == "2020-01-01"
    assert m["drag_pp"] == pytest.approx(0.0, abs=1e-9)


def test_compute_pair_metrics_rejects_short_window():
    direct = _payload([("01-01-2024", 10.0), ("01-06-2024", 11.0)])
    regular = _payload([("01-01-2024", 10.0), ("01-06-2024", 11.0)])
    assert compute_pair_metrics(direct, regular, min_years=2.0) is None


# --- SIP cost translation -------------------------------------------------------

def test_sip_future_value_zero_return_is_just_contributions():
    assert sip_future_value(1000, 0.0, 10) == 1000 * 120


def test_sip_future_value_grows_with_return():
    fv = sip_future_value(10_000, 0.12, 20)
    assert fv > 10_000 * 240
    # Closed-form annuity-due check for a single year at 12%/yr.
    one_year = sip_future_value(100, 0.12, 1)
    r = 0.01
    expected = 100 * (((1 + r) ** 12 - 1) / r) * (1 + r)
    assert one_year == pytest.approx(expected)


def test_sip_drag_cost_is_positive_and_consistent():
    out = sip_drag_cost(10_000, 0.12, 1.4, 20)
    assert out["invested"] == 10_000 * 240
    assert out["direct_fv"] > out["regular_fv"]
    assert out["lost_to_drag"] == pytest.approx(out["direct_fv"] - out["regular_fv"])
    assert 0 < out["lost_pct_of_corpus"] < 100
    assert not math.isnan(out["lost_pct_of_corpus"])


def test_sip_drag_cost_zero_drag_loses_nothing():
    out = sip_drag_cost(5_000, 0.10, 0.0, 15)
    assert out["lost_to_drag"] == pytest.approx(0.0)


# --- categories -----------------------------------------------------------------

@pytest.mark.parametrize("cat,expected", [
    ("Equity Scheme - Large Cap Fund", "Equity"),
    ("Equity Scheme - ELSS", "Equity"),
    ("Debt Scheme - Low Duration Fund", "Debt"),
    ("Debt Scheme - Gilt Fund", "Debt"),
    ("Hybrid Scheme - Balanced Advantage", "Hybrid"),
    ("Solution Oriented Scheme - Retirement Fund", "Hybrid"),
    ("Other Scheme - Index Funds", "Other/Index"),
    ("Other Scheme - FoF Overseas", "Other/Index"),
    (float("nan"), "Unknown"),
    (None, "Unknown"),
])
def test_categorize_broad(cat, expected):
    assert categorize_broad(cat) == expected
