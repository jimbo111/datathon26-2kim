"""Cache-or-call utility — fetch from API or load from local cache.

Every API call in this project MUST use this pattern to ensure the
notebook runs reliably even when APIs are rate-limited or offline.
"""

import hashlib
import json
from pathlib import Path
from typing import Callable

import pandas as pd
from rich.console import Console

console = Console()

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def cache_or_call(
    key: str,
    fetch_fn: Callable[[], pd.DataFrame],
    subdir: str = "raw",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load from parquet cache if available; otherwise call fetch_fn and cache.

    Args:
        key: Unique filename (without extension) for the cached file.
        fetch_fn: Callable that returns a DataFrame when invoked.
        subdir: "raw" or "processed".
        force_refresh: If True, always call the API and overwrite cache.

    Returns:
        DataFrame from cache or fresh fetch.
    """
    base = RAW_DIR if subdir == "raw" else PROCESSED_DIR
    path = base / f"{key}.parquet"

    if path.exists() and not force_refresh:
        console.print(f"[dim]Cache hit:[/] {path.name}")
        return pd.read_parquet(path)

    console.print(f"[cyan]Fetching:[/] {key} ...")
    try:
        df = fetch_fn()
        df.to_parquet(path, index=True)
        console.print(f"[green]Cached:[/] {path.name} ({len(df):,} rows)")
        return df
    except Exception as e:
        # If fetch fails but cache exists, use stale cache
        if path.exists():
            console.print(f"[yellow]Fetch failed ({e}), using stale cache:[/] {path.name}")
            return pd.read_parquet(path)
        raise


def cache_json(key: str, data: dict, subdir: str = "processed") -> Path:
    """Cache a dict as JSON."""
    base = RAW_DIR if subdir == "raw" else PROCESSED_DIR
    path = base / f"{key}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def load_json(key: str, subdir: str = "processed") -> dict | None:
    """Load a cached JSON dict, or None if not found."""
    base = RAW_DIR if subdir == "raw" else PROCESSED_DIR
    path = base / f"{key}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None
