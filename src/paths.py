"""Canonical project paths, resolved relative to the repo root.

Importing these instead of hard-coding strings keeps notebooks runnable no
matter which working directory Jupyter is launched from.
"""
from __future__ import annotations

from pathlib import Path

# src/paths.py -> src/ -> repo root
ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
RAW = DATA / "raw"
RAW_NAV = RAW / "nav"            # one JSON per scheme code (git-ignored, large)
PROCESSED = DATA / "processed"

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

# Key cached artifacts
SCHEME_LIST_JSON = RAW / "scheme_list.json"
PAIRS_CSV = PROCESSED / "pairs.csv"                 # matched Direct<->Regular funds
RETURNS_CSV = PROCESSED / "fund_returns.csv"        # per-fund returns + drag (analysis-ready)


def ensure_dirs() -> None:
    """Create every directory the pipeline writes to (idempotent)."""
    for d in (RAW, RAW_NAV, PROCESSED, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
