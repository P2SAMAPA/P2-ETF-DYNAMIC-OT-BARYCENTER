"""
data_manager.py  —  Data loading
"""

import os
import sys
import logging
from typing import List, Optional, Tuple
import pandas as pd
from huggingface_hub import HfApi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

logger = logging.getLogger(__name__)

# Macro columns are used only for regime *characterization* in the
# diagnostics layer (e.g. "this regime's days had an average VIX of ~27")
# — never as inputs to the barycenter itself, which is built purely from
# each ETF's own return distribution. See barycenter_model.py.
MACRO_COLS_CORE = ["VIX", "T10Y2Y", "DXY"]
MACRO_COLS_EXTENDED = ["IG_SPREAD", "HY_SPREAD"]


def _all_tickers() -> List[str]:
    seen, result = set(), []
    for tickers in config.UNIVERSES.values():
        for t in tickers:
            if t not in seen:
                seen.add(t)
                result.append(t)
    return result


def load_master_data(hf_token: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    token = hf_token or config.HF_TOKEN or os.environ.get("HF_TOKEN", "")
    if not token:
        raise ValueError("HF_TOKEN not set")

    api = HfApi(token=token)
    logger.info(f"Downloading master parquet from {config.DATA_REPO} …")

    parquet_path = None
    for fname in ["master_data.parquet", "data/master.parquet", "master.parquet"]:
        try:
            parquet_path = api.hf_hub_download(repo_id=config.DATA_REPO, filename=fname, repo_type="dataset", token=token)
            logger.info(f"  → found at '{fname}'")
            break
        except Exception:
            continue

    if parquet_path is None:
        raise RuntimeError(f"Could not locate master parquet in {config.DATA_REPO}.")

    df = pd.read_parquet(parquet_path)
    logger.info(f"Raw parquet: {df.shape[0]} rows × {df.shape[1]} cols")

    if not isinstance(df.index, pd.DatetimeIndex):
        for date_col in ["Date", "date", "DATE", "timestamp"]:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
                break
        else:
            df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    df.index.name = "Date"

    all_tickers = _all_tickers()
    avail_tickers = [t for t in all_tickers if t in df.columns]
    if not avail_tickers:
        raise RuntimeError("No ETF ticker columns found.")

    prices = df[avail_tickers].copy()
    prices = prices.ffill()
    prices = prices.dropna(how="all")

    # Macro is best-effort and optional — the barycenter signal never
    # depends on it, so a missing/incomplete macro block should shrink
    # the diagnostics, not fail the whole run.
    macro = None
    avail_core = [c for c in MACRO_COLS_CORE if c in df.columns]
    avail_ext = [c for c in MACRO_COLS_EXTENDED if c in df.columns]
    avail_all = avail_core + avail_ext
    if avail_all:
        macro = df[avail_all].copy()
        if avail_core:
            before = len(macro)
            macro = macro.dropna(subset=avail_core)
            dropped = before - len(macro)
            if dropped:
                logger.info(f"Dropped {dropped} rows with NaN in core macro cols.")
        if avail_ext:
            macro[avail_ext] = macro[avail_ext].ffill().fillna(0.0)

        common = prices.index.intersection(macro.index)
        if len(common) > 0:
            prices = prices.loc[common]
            macro = macro.loc[common]
        else:
            macro = None

    logger.info(
        f"Dataset ready: {len(prices)} rows | {len(avail_tickers)} ETFs | "
        + (f"{len(avail_all)} macro cols" if macro is not None else "no usable macro cols")
    )
    return prices, macro


def validate_data(prices: pd.DataFrame, macro: Optional[pd.DataFrame]) -> None:
    for ticker in list(prices.columns)[:3]:
        col = prices[ticker].dropna()
        if len(col) > 10 and abs(col.median()) < 0.05:
            logger.warning(f"'{ticker}' median ≈ {col.median():.4f} — looks like returns, not prices!")

    if len(prices) < 252:
        logger.warning(f"Only {len(prices)} rows — less than 1 year of data.")
