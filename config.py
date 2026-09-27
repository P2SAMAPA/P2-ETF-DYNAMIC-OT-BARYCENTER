"""
config.py  —  Configuration for P2 DYNAMIC-OT-BARYCENTER
"""

import os
import numpy as np

HF_TOKEN = os.environ.get("HF_TOKEN")
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
RESULTS_REPO = "P2SAMAPA/p2-dynamic-ot-barycenter-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE"],
    "COMBINED": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV", "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE"]
}

# Multiple regime-fit lookback windows for analysis: how much barycenter
# history feeds the K-means regime discovery + regime-conditional
# next-day-return lookup, at each walk-forward step.
WINDOWS = [126, 252, 504, 756, 1008]

# Quantile levels used to represent every 1-D return distribution (each
# ETF's own local distribution AND the market barycenter alike). 25
# levels from the 2nd to 98th percentile: fine enough to carry shape /
# skew / tail asymmetry, coarse enough to stay numerically stable when
# estimated from a ~21-day local window (see LOCAL_WINDOW below). For
# 1-D distributions the quantile function is exactly the representation
# under which Wasserstein-2 distance becomes ordinary Euclidean distance
# (see barycenter_model.py's module docstring) — that equivalence is
# what makes everything in this engine tractable on free-tier CPU with
# no transport-plan solver anywhere.
QUANTILE_LEVELS = np.linspace(0.02, 0.98, 25).tolist()

# Dynamic-OT-Barycenter configuration
#
# local_window       : trailing days used to build each ETF's own
#                       empirical return distribution mu_ETFi(t) on day
#                       t — the "local" distribution that feeds the
#                       barycenter. 21 trading days (~1 month) is short
#                       enough that the distribution reflects CURRENT
#                       conditions rather than a stale multi-year blend.
# n_regimes          : number of K-means clusters fit over the historical
#                       barycenter series, i.e. how many recurring
#                       "market-distribution shapes" to discover. K-means
#                       on the quantile-function representation is exactly
#                       Wasserstein-space clustering (see barycenter_model.py).
# min_train_samples  : minimum number of (barycenter day -> next-day
#                       return) training pairs required before a regime
#                       fit + lookup is even attempted — needs to be
#                       comfortably larger than n_regimes so every
#                       cluster gets a non-trivial number of days.
# burn_in_fraction   : fraction of a universe's available barycenter
#                       history reserved as burn-in before walk-forward
#                       testing starts. Kept small for the same reason as
#                       the sheaf engine: every test window is still
#                       individually protected by its own `window`-sized
#                       training lookback, so a large global burn-in only
#                       wastes usable out-of-sample days.
BARYCENTER_CONFIG = {
    "local_window": 21,
    "n_regimes": 6,
    "min_train_samples": 60,
    "burn_in_fraction": 0.05,
}

# Small hyperparameter grid searched PER WINDOW, per universe, selecting
# the combination with the best out-of-sample correlation (consistent
# with BEST_WINDOW_METRIC below — prediction quality drives selection,
# not backtested P&L). local_window controls how "local"/recent each
# day's per-ETF distribution is before it enters the barycenter;
# n_regimes controls how finely historical market states are carved up.
#
# IMPORTANT CAVEAT: as with the sheaf engine, searching more combinations
# increases the risk of picking one that looks good by pure chance. The
# trainer reports the FULL comparison table, not just the winner, so this
# can be checked directly.
BARYCENTER_GRID = [
    {"n_regimes": 4, "local_window": 21},
    {"n_regimes": 6, "local_window": 21},
    {"n_regimes": 8, "local_window": 21},
    {"n_regimes": 6, "local_window": 10},
    {"n_regimes": 6, "local_window": 42},
    {"n_regimes": 6, "local_window": 63},
]

TOP_N = 3

# Round-trip trading cost assumption, in basis points, applied to every
# position change (turnover) in the backtest.
TRADING_COST_BPS = 15

# How the "best window" per universe is chosen for Tab 1's live picks.
# "correlation" selects the window whose return PREDICTIONS were most
# accurate historically — a direct measure of prediction quality, for
# the same reason the sheaf engine avoids selecting by Sharpe: backtested
# P&L can look good from a regime assignment that barely explains any
# real variance, simply by riding the test period's market drift.
BEST_WINDOW_METRIC = "correlation"
