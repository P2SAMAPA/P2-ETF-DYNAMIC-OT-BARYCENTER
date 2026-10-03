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
    "EQUITY_SECTORS": ["SPY", "QQQ", "XLK", "XLF", "XLE", "CPER", "COPX", "XLV", "XLI", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE"],
    "COMBINED": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV", "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "CPER", "COPX", "XLI", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE"]
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
# regime_refit_interval : how often (in valid trading days) the K-means
#                       regime set is actually re-fit during walk-forward
#                       testing/live prediction, instead of every single
#                       day. Refitting daily lets cluster labels/centroids
#                       drift noticeably from one day to the next even
#                       when the underlying distribution barely moved,
#                       which flips predicted-return signs (and therefore
#                       triggers turnover cost) for no real informational
#                       gain. 21 days (~1 trading month) matches
#                       local_window's own timescale.
# persistence_margin   : hysteresis threshold (see
#                       barycenter_model.predict_with_hysteresis):
#                       the regime assignment only switches away from the
#                       previously-held regime if the new nearest regime
#                       is closer by more than this fraction. 0.10 means a
#                       10% closer match is required before switching.
# horizon              : PREDICTION TARGET horizon, in trading days — the
#                       barycenter/regime machinery is unchanged, but the
#                       "actual" it's trained and evaluated against is now
#                       the CUMULATIVE forward return over this many days
#                       (see barycenter_model.compute_forward_returns),
#                       not tomorrow's single-day return. The original
#                       design used a horizon of 1 implicitly; two real
#                       backtests showed next-single-day directional
#                       accuracy sitting at or below chance almost
#                       everywhere, which single-day ETF returns are
#                       plausibly just too noisy to predict from a
#                       once-a-month-ish regime signal. 5 trading days
#                       (~1 week) is the new default; BARYCENTER_GRID
#                       below also searches 1/3/10/20 explicitly so this
#                       assumption gets checked, not just asserted.
BARYCENTER_CONFIG = {
    "local_window": 21,
    "n_regimes": 6,
    "min_train_samples": 60,
    "burn_in_fraction": 0.05,
    "regime_refit_interval": 21,
    "persistence_margin": 0.10,
    "horizon": 5,
}

# Small hyperparameter grid searched PER WINDOW, per universe, selecting
# the combination with the best out-of-sample correlation (consistent
# with BEST_WINDOW_METRIC below — prediction quality drives selection,
# not backtested P&L). local_window controls how "local"/recent each
# day's per-ETF distribution is before it enters the barycenter;
# n_regimes controls how finely historical market states are carved up;
# horizon controls how many trading days ahead the return target is
# cumulated over (see BARYCENTER_CONFIG's "horizon" comment above). The
# first 6 combos hold horizon at the new default (5d) and vary
# local_window/n_regimes as before; the last 4 hold local_window/n_regimes
# at their defaults and vary horizon, so the horizon question gets a
# direct, isolated comparison rather than being tangled up with the other
# two hyperparameters.
#
# IMPORTANT CAVEAT: as with the sheaf engine, searching more combinations
# increases the risk of picking one that looks good by pure chance. The
# trainer reports the FULL comparison table, not just the winner, so this
# can be checked directly.
BARYCENTER_GRID = [
    {"n_regimes": 4, "local_window": 21, "horizon": 5},
    {"n_regimes": 6, "local_window": 21, "horizon": 5},
    {"n_regimes": 8, "local_window": 21, "horizon": 5},
    {"n_regimes": 6, "local_window": 10, "horizon": 5},
    {"n_regimes": 6, "local_window": 42, "horizon": 5},
    {"n_regimes": 6, "local_window": 63, "horizon": 5},
    {"n_regimes": 6, "local_window": 21, "horizon": 1},
    {"n_regimes": 6, "local_window": 21, "horizon": 3},
    {"n_regimes": 6, "local_window": 21, "horizon": 10},
    {"n_regimes": 6, "local_window": 21, "horizon": 20},
]

TOP_N = 3

# Round-trip trading cost assumption, in basis points, applied to every
# position change (turnover) in the backtest.
TRADING_COST_BPS = 15

# How the "best window" per universe is chosen for Tab 1's live picks.
#
# Selection is now GATED + BLENDED, not raw correlation alone (an earlier
# version selected purely by correlation, which on real data once picked
# a window with WORSE net Sharpe and WORSE directional accuracy than an
# alternative window, because the correlation gap between them was itself
# within noise -- see trainer._select_best_window's docstring):
#
#   1. Gate: only windows whose out-of-sample directional accuracy meets
#      MIN_DIRECTIONAL_ACCURACY are eligible at all. A window that is
#      wrong more than half the time has no business being "best"
#      regardless of its raw correlation.
#   2. Among eligible windows, rank by a blend of correlation and net
#      Sharpe (rank-summed, since the two are on different scales) so a
#      window doesn't win purely on a fractionally higher correlation
#      while having a clearly worse net-of-cost outcome.
#   3. If NO window clears the gate, the same blended ranking is used as
#      a fallback over every window (a pick still has to be produced),
#      but `gate_passed: false` is recorded in the results, and
#      `_confidence` already forces "Low" confidence whenever a window's
#      own directional accuracy is under 50%, independent of this
#      selection step.
BEST_WINDOW_METRIC = "correlation"   # kept as one half of the blend, alongside net Sharpe
MIN_DIRECTIONAL_ACCURACY = 0.50
