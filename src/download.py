"""Fetch and cache Indian mutual-fund NAV history from the mfapi.in API.

mfapi.in is a free, public wrapper over AMFI's official daily NAV files.

Endpoints
---------
GET /mf            -> list of every scheme: {schemeCode, schemeName, isin...}
GET /mf/{code}     -> {meta: {fund_house, scheme_category, ...},
                       data: [{date: 'dd-mm-yyyy', nav: '...'}, ...]}

Everything is cached on disk so the analysis reproduces without re-downloading,
and a re-run only fetches schemes it doesn't already have.

CLI
---
    python -m src.download              # full pipeline (list -> pair -> NAVs)
    python -m src.download --limit 200  # quick smoke test on 200 pairs
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from . import paths

BASE_URL = "https://api.mfapi.in"
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "mf-fee-drag-analysis/1.0 (portfolio project)"})


def fetch_scheme_list(force: bool = False) -> list[dict]:
    """Return the full scheme list, caching it to data/raw/scheme_list.json."""
    paths.ensure_dirs()
    if paths.SCHEME_LIST_JSON.exists() and not force:
        return json.loads(paths.SCHEME_LIST_JSON.read_text())

    resp = _SESSION.get(f"{BASE_URL}/mf", timeout=60)
    resp.raise_for_status()
    data = resp.json()
    paths.SCHEME_LIST_JSON.write_text(json.dumps(data))
    return data


def _nav_cache_path(code: int | str) -> Path:
    return paths.RAW_NAV / f"{code}.json"


def fetch_scheme_nav(code: int | str, force: bool = False,
                     retries: int = 3) -> dict | None:
    """Fetch one scheme's NAV history + metadata, caching to data/raw/nav/{code}.json.

    Returns the parsed payload, or None if the scheme has no usable data after
    retries. Cached responses are returned immediately (the common case on re-run).
    """
    paths.ensure_dirs()
    cache = _nav_cache_path(code)
    if cache.exists() and not force:
        try:
            return json.loads(cache.read_text())
        except json.JSONDecodeError:
            pass  # corrupt cache -> re-fetch

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = _SESSION.get(f"{BASE_URL}/mf/{code}", timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            # mfapi returns {"status":"FAIL"} or empty data for dead codes
            if not payload.get("data"):
                return None
            cache.write_text(json.dumps(payload))
            return payload
        except (requests.RequestException, json.JSONDecodeError) as err:
            last_err = err
            time.sleep(0.5 * (attempt + 1))  # simple backoff
    print(f"  ! giving up on scheme {code}: {last_err}")
    return None


def download_navs(codes: list[int | str], max_workers: int = 12,
                  force: bool = False) -> dict[int | str, dict]:
    """Download NAV history for many schemes concurrently, returning {code: payload}.

    Politely capped concurrency; already-cached codes resolve instantly.
    """
    results: dict[int | str, dict] = {}
    todo = [c for c in codes if force or not _nav_cache_path(c).exists()]
    cached = len(codes) - len(todo)
    print(f"download_navs: {len(codes)} requested, {cached} cached, {len(todo)} to fetch")

    # Serve cached ones without a thread pool.
    for c in codes:
        if c not in todo:
            payload = fetch_scheme_nav(c)
            if payload is not None:
                results[c] = payload

    if not todo:
        return results

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_scheme_nav, c, force): c for c in todo}
        for fut in as_completed(futures):
            c = futures[fut]
            payload = fut.result()
            if payload is not None:
                results[c] = payload
            done += 1
            if done % 200 == 0 or done == len(todo):
                print(f"  fetched {done}/{len(todo)}")
    return results


def run_pipeline(limit: int | None = None) -> None:
    """End-to-end: scheme list -> matched pairs -> download their NAV history.

    Imports cleaning here (not at module top) to avoid a circular import, since
    cleaning depends only on pandas, not on this module.
    """
    from . import cleaning

    print("1/3  Fetching scheme list ...")
    raw_list = fetch_scheme_list()
    print(f"     {len(raw_list):,} schemes")

    print("2/3  Parsing names and pairing Direct <-> Regular (Growth) ...")
    pairs = cleaning.build_pairs(raw_list)
    if limit is not None:
        pairs = pairs.head(limit)
    paths.PROCESSED.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(paths.PAIRS_CSV, index=False)
    print(f"     {len(pairs):,} matched fund pairs -> {paths.PAIRS_CSV.name}")

    print("3/3  Downloading NAV history for paired schemes ...")
    codes = pairs["direct_code"].tolist() + pairs["regular_code"].tolist()
    download_navs(codes)
    print("Done. Run notebook 02 to build the processed dataset.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download AMFI NAV data for fee-drag analysis.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N matched pairs (smoke test).")
    args = ap.parse_args()
    run_pipeline(limit=args.limit)
