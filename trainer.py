"""
trainer.py  —  P2 DYNAMIC-OT-BARYCENTER Trainer with Multi-Window Backtesting

Builds a dynamic Wasserstein barycenter of each universe's ETF return
distributions, clusters its historical trajectory into recurring "market
regimes," and tests whether today's nearest regime (in Wasserstein /
quantile-function distance) predicts next-day returns — across several
regime-fit window sizes and (local_window, n_regimes) hyperparameter
combinations. See barycenter_model.py for the theory.

Regime persistence: the regime set is only re-fit every
`regime_refit_interval` days (not daily), and the assignment only switches
regimes when the new nearest one is closer by more than `persistence_margin`
(hysteresis) — see barycenter_model.predict_with_hysteresis. Both exist to
curb turnover-cost drag from noise-level day-to-day regime relabeling. Live
runs also carry the previous run's regime assignment forward (best-effort,
via the last pushed results file) so hysteresis works across separate
scheduled runs, not just within one backtest.
"""

import os
import sys
import json
import glob
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_manager import load_master_data, validate_data
from barycenter_model import (
    compute_barycenter_series, RegimeBarycenterModel, nearest_analog_days, predict_with_hysteresis,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def backtest_window(returns: np.ndarray, barycenter_series: np.ndarray, window: int,
                     n_regimes: int, cfg: Dict) -> Dict:
    """
    Walk-forward backtest of the barycenter-regime signal for one
    (regime-fit window, n_regimes) combination against a barycenter series
    already built with a particular local_window (see the caller).

    The regime set is re-fit only every `regime_refit_interval` valid days
    (not on every single day), and between refits the assignment is only
    switched when hysteresis (`persistence_margin`) says the new nearest
    regime is clearly better — see barycenter_model.predict_with_hysteresis.
    This is what keeps predicted-return signs (and therefore turnover cost)
    from flickering on noise-level day-to-day distance changes.
    """
    n_samples = len(returns)
    valid = ~np.isnan(barycenter_series).any(axis=1)
    valid_idx = np.where(valid)[0]
    if len(valid_idx) < window + 50:
        return {"error": "Insufficient data", "window": window}

    burn_in_fraction = cfg.get("burn_in_fraction", 0.05)
    min_train_samples = cfg.get("min_train_samples", 60)
    refit_interval = max(1, cfg.get("regime_refit_interval", 21))
    persistence_margin = cfg.get("persistence_margin", 0.10)
    start_pos = int(len(valid_idx) * burn_in_fraction)

    predictions, actuals, distances, separations, r2s, switches = [], [], [], [], [], []

    model = None
    fit_result = None
    current_regime = None
    steps_since_refit = refit_interval  # forces a fit on the very first iteration

    for pos in range(start_pos, len(valid_idx) - 1):
        i = valid_idx[pos]                      # "today" — must itself have a valid barycenter
        if i + 1 >= n_samples:
            continue

        train_idx = valid_idx[(valid_idx >= i - window) & (valid_idx < i)]
        train_idx = train_idx[train_idx + 1 <= i]
        if len(train_idx) < min_train_samples:
            continue

        try:
            if model is None or steps_since_refit >= refit_interval:
                B_train = barycenter_series[train_idx]
                returns_next_train = returns[train_idx + 1]
                model = RegimeBarycenterModel(n_regimes=n_regimes)
                fit_result = model.fit(B_train, returns_next_train)
                current_regime = None   # unconditional nearest-regime pick right after a refit
                steps_since_refit = 0

            pred = predict_with_hysteresis(model, barycenter_series[i], current_regime, persistence_margin)
            current_regime = pred["regime"]
            steps_since_refit += 1

            actual_returns = returns[i + 1]

            predictions.append(pred["predicted_returns"])
            actuals.append(actual_returns)
            distances.append(pred["distance"])
            separations.append(pred["separation"])
            r2s.append(fit_result["in_sample_r2"])
            switches.append(pred["switched"])
        except Exception:
            continue

    if len(predictions) < 10:
        return {"error": "Not enough predictions", "window": window}

    predictions = np.array(predictions)   # (n_steps, n_tickers)
    actuals = np.array(actuals)

    correlation = np.corrcoef(predictions.flatten(), actuals.flatten())[0, 1]
    mse = np.mean((predictions - actuals) ** 2)

    pred_sign = np.sign(predictions)
    actual_sign = np.sign(actuals)
    directional_accuracy = np.mean(pred_sign.flatten() == actual_sign.flatten())

    gross_returns = actuals * pred_sign

    # Turnover-based trading costs — same convention as the sheaf engine:
    # a cost is paid only when a ticker's position (long/flat/short, from
    # pred_sign) actually changes from one day to the next, not on every
    # day a position is simply held. With regime persistence above, this
    # should now track real forecast changes rather than K-means label
    # noise, and come out noticeably lower than a daily-refit version.
    cost_bps = cfg.get("trading_cost_bps", 15)
    prev_position = np.zeros((1, pred_sign.shape[1]))
    position_history = np.vstack([prev_position, pred_sign])
    turnover = np.abs(np.diff(position_history, axis=0))
    trading_cost = (cost_bps / 10000.0) * turnover

    net_returns = gross_returns - trading_cost

    sharpe_gross = np.mean(gross_returns.flatten()) / (np.std(gross_returns.flatten()) + 1e-8) * np.sqrt(252)
    sharpe_net = np.mean(net_returns.flatten()) / (np.std(net_returns.flatten()) + 1e-8) * np.sqrt(252)

    return {
        "window": window,
        "n_predictions": len(predictions),
        "correlation": float(correlation) if not np.isnan(correlation) else 0.0,
        "mse": float(mse),
        "directional_accuracy": float(directional_accuracy),
        "sharpe": float(sharpe_net),
        "sharpe_gross": float(sharpe_gross),
        "mean_return": float(np.mean(net_returns.flatten())),
        "mean_return_gross": float(np.mean(gross_returns.flatten())),
        "std_return": float(np.std(net_returns.flatten())),
        "avg_daily_cost_bps": float(np.mean(trading_cost.flatten()) * 10000.0),
        "trading_cost_bps_assumed": cost_bps,
        "avg_regime_distance": float(np.mean(distances)),
        "avg_regime_separation": float(np.mean(separations)),
        "avg_in_sample_r2": float(np.mean(r2s)),
        "avg_regime_switch_rate": float(np.mean(switches)),
        "regime_refit_interval": refit_interval,
        "persistence_margin": persistence_margin,
        "distance_series": [round(float(d), 4) for d in distances],
    }


def backtest_window_with_search(returns: np.ndarray, barycenter_series_by_lw: Dict[int, np.ndarray],
                                 window: int, base_cfg: Dict, grid: List[Dict]) -> Dict:
    """
    Try each (local_window, n_regimes) combination in `grid` for this
    window, keep the one with the best out-of-sample correlation
    (consistent with BEST_WINDOW_METRIC). Returns the winner, with the
    winning hyperparameters recorded on it, plus a `hyperparam_search`
    list of every combination tried — so the winner's margin over the
    rest of the grid can be checked for multiple-comparisons risk.
    """
    candidates = []
    for combo in grid:
        lw = combo.get("local_window", base_cfg.get("local_window"))
        nr = combo.get("n_regimes", base_cfg.get("n_regimes"))
        B = barycenter_series_by_lw.get(lw)
        if B is None:
            continue
        result = backtest_window(returns, B, window, nr, base_cfg)
        if "error" not in result:
            result = dict(result)
            result["local_window"] = lw
            result["n_regimes"] = nr
            candidates.append(result)

    if not candidates:
        return {"error": "Not enough predictions", "window": window}

    best = max(candidates, key=lambda r: r.get("correlation", -999))
    best = dict(best)
    best["n_hyperparam_combos_tested"] = len(candidates)
    best["hyperparam_search"] = [
        {
            "local_window": c["local_window"],
            "n_regimes": c["n_regimes"],
            "correlation": round(c["correlation"], 5),
            "sharpe": round(c["sharpe"], 3),
            "n_predictions": c["n_predictions"],
        }
        for c in sorted(candidates, key=lambda r: -r.get("correlation", -999))
    ]
    return best


def _select_best_window(window_results: Dict[int, Dict], min_directional_accuracy: float = 0.50) -> Tuple[int, Dict, bool]:
    """
    Select the best regime-fit window for live picks and reporting.

    Gate: only windows whose out-of-sample directional accuracy clears
    `min_directional_accuracy` are eligible at all — a window that is
    wrong more than half the time has no business being called "best"
    regardless of how its raw correlation compares to other windows this
    close (concretely: on one real run, pure-correlation selection picked
    a window with WORSE net Sharpe and WORSE directional accuracy than an
    alternative window, because the correlation gap between them — 0.030
    vs 0.023 — was itself within noise given a few thousand pooled,
    serially-correlated daily observations).

    Among windows that clear the gate, rank by a blend of correlation and
    net Sharpe (rank-summed, since the two live on different scales) so a
    window doesn't win purely on a fractionally higher correlation while
    having a clearly worse, or negative, net-of-cost outcome.

    If NO window clears the gate, the same blended ranking is applied as
    a fallback over every window (a pick still has to be produced), but
    the returned `gate_passed=False` records that none of them actually
    validated — and downstream, `_confidence` already forces "Low"
    whenever a window's own directional accuracy is under 50%, regardless
    of this selection step, so a failed gate can never surface as a
    confident-looking pick.
    """
    eligible = {w: r for w, r in window_results.items() if r.get("directional_accuracy", 0) >= min_directional_accuracy}
    gate_passed = len(eligible) > 0
    pool = eligible if gate_passed else window_results

    windows = list(pool.keys())
    corr_rank = {w: i for i, w in enumerate(sorted(windows, key=lambda w: pool[w].get("correlation", -999)))}
    sharpe_rank = {w: i for i, w in enumerate(sorted(windows, key=lambda w: pool[w].get("sharpe", -999)))}
    blended = {w: corr_rank[w] + sharpe_rank[w] for w in windows}

    best_w = max(windows, key=lambda w: blended[w])
    return best_w, pool[best_w], gate_passed


def _confidence(oos_correlation: float, oos_directional_accuracy: float, in_sample_r2: float) -> str:
    """
    Confidence is graded primarily on OUT-OF-SAMPLE walk-forward validity
    for the exact window that produced this pick — NOT on in-sample R^2
    alone. In-sample R^2 looks reasonable almost by construction (K-means
    is fit to explain that exact training data), so on its own it is not a
    trustworthy signal: a window whose backtested predictions were actually
    anti-correlated with real subsequent returns must never be labeled
    "High confidence" just because its regime split fit its own training
    data well.
    """
    if oos_correlation <= 0 or oos_directional_accuracy < 0.50:
        return "Low"
    if oos_correlation > 0.02 and oos_directional_accuracy > 0.51 and in_sample_r2 > 0.01:
        return "High"
    if oos_correlation > 0.0:
        return "Medium"
    return "Low"


def compute_ticker_picks(returns: np.ndarray, barycenter_series: np.ndarray, dates: List[str],
                          tickers: List[str], window: int, n_regimes: int, cfg: Dict, top_n: int,
                          oos_correlation: float = 0.0, oos_directional_accuracy: float = 0.5,
                          previous_regime: Optional[int] = None) -> Tuple[List[Dict], Dict, Dict]:
    """
    Fit regimes on the most recent `window` valid days and return the
    top-N ETF picks by regime-conditional predicted next-day return, plus
    per-ticker detail and run diagnostics — including a 2D "regime map"
    for the dashboard, the today-vs-regime-centroid quantile curves, and
    the individual historical days today's distribution most resembles.

    `oos_correlation` / `oos_directional_accuracy` should be this exact
    window's walk-forward backtest metrics (from backtest_window), and
    drive the confidence label — see _confidence.

    `previous_regime` — the regime this same (universe, window,
    n_regimes, local_window) combination was assigned to last run, if
    known — applies the same hysteresis as the backtest so a live pick
    doesn't flip on noise either. Pass None for an unconditional pick
    (e.g. no prior state available, or the hyperparameters changed).
    """
    valid = ~np.isnan(barycenter_series).any(axis=1)
    valid_idx = np.where(valid)[0]
    if len(valid_idx) < 2:
        return [], {}, {}

    today_idx = valid_idx[-1]
    train_idx = valid_idx[(valid_idx >= today_idx - window) & (valid_idx < today_idx)]
    train_idx = train_idx[train_idx + 1 <= today_idx]

    min_train_samples = cfg.get("min_train_samples", 60)
    if len(train_idx) < min_train_samples:
        return [], {}, {}

    try:
        B_train = barycenter_series[train_idx]
        returns_next_train = returns[train_idx + 1]

        model = RegimeBarycenterModel(n_regimes=n_regimes)
        fit_result = model.fit(B_train, returns_next_train)
        persistence_margin = cfg.get("persistence_margin", 0.10)
        pred = predict_with_hysteresis(model, barycenter_series[today_idx], previous_regime, persistence_margin)
    except Exception as e:
        logger.error(f"  Barycenter fit failed (window={window}): {e}")
        return [], {}, {}

    pred_returns = pred["predicted_returns"]
    confidence = _confidence(oos_correlation, oos_directional_accuracy, fit_result["in_sample_r2"])

    ticker_results = {ticker: {"next_return": float(pred_returns[j])} for j, ticker in enumerate(tickers)}

    sorted_picks = sorted(ticker_results.items(), key=lambda x: x[1]["next_return"], reverse=True)
    top_picks = sorted_picks[:top_n]
    picks = [
        {"ticker": ticker, "expected_return": round(info["next_return"] * 100, 2), "confidence": confidence}
        for ticker, info in top_picks
    ]

    # Individual-day analogs — only among days strictly BEFORE today, no leakage.
    hist_idx = valid_idx[valid_idx < today_idx]
    hist_idx = hist_idx[hist_idx + 1 <= today_idx]
    analogs = []
    if len(hist_idx) > 0:
        analogs = nearest_analog_days(
            barycenter_series[today_idx], barycenter_series[hist_idx],
            [dates[k] for k in hist_idx], returns[hist_idx + 1], tickers, top_n=5,
        )

    # 2D regime map for the dashboard: PCA projection of the (small)
    # training barycenter matrix, plus today's point and the fitted
    # regime centers, computed ONCE here and stored in the results JSON
    # so the Streamlit app never has to recompute anything to render it.
    regime_map = None
    try:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(B_train)
        labels = model.kmeans.labels_
        today_coords = pca.transform(barycenter_series[today_idx].reshape(1, -1))[0]
        center_coords = pca.transform(model.kmeans.cluster_centers_)
        regime_map = {
            "points": [
                {"x": round(float(coords[k, 0]), 4), "y": round(float(coords[k, 1]), 4),
                 "regime": int(labels[k]), "date": dates[train_idx[k]]}
                for k in range(len(train_idx))
            ],
            "today": {"x": round(float(today_coords[0]), 4), "y": round(float(today_coords[1]), 4)},
            "centers": [
                {"x": round(float(center_coords[r, 0]), 4), "y": round(float(center_coords[r, 1]), 4), "regime": r}
                for r in range(len(center_coords))
            ],
            "explained_variance": [round(float(v), 3) for v in pca.explained_variance_ratio_],
        }
    except Exception as e:
        logger.warning(f"  Could not build regime map: {e}")

    quantile_curves = {
        "quantile_levels": config.QUANTILE_LEVELS,
        "today": [round(float(v), 5) for v in barycenter_series[today_idx]],
        "nearest_regime_centroid": [round(float(v), 5) for v in model.kmeans.cluster_centers_[pred["regime"]]],
    }

    diagnostics = {
        "regime": pred["regime"],
        "regime_size": pred["regime_size"],
        "regime_distance": round(pred["distance"], 4),
        "regime_separation": round(pred["separation"], 4),
        "in_sample_r2": round(fit_result["in_sample_r2"], 5),
        "n_regimes_used": fit_result["n_regimes_fit"],
        "regime_counts": fit_result["regime_counts"],
        "n_regimes_requested": n_regimes,
        "nearest_analog_days": analogs,
        "regime_map": regime_map,
        "quantile_curves": quantile_curves,
        "previous_regime": previous_regime,
        "switched_from_previous": pred["switched"],
        "oos_correlation_used_for_confidence": round(oos_correlation, 5),
        "oos_directional_accuracy_used_for_confidence": round(oos_directional_accuracy, 4),
    }

    return picks, ticker_results, diagnostics


def _load_previous_results(exclude_path: Optional[str] = None) -> Optional[Dict]:
    """
    Best-effort lookup of the most recent PREVIOUS results file — local
    working directory first, then the HF results dataset — so regime
    persistence (predict_with_hysteresis) has a "yesterday" to compare
    against across separate scheduled runs, not just within one backtest.
    This is advisory only: any failure just means today's run starts with
    an unconditional (fresh) regime pick for every universe/window, same
    as if this feature didn't exist.
    """
    candidates = sorted(glob.glob("barycenter_results_*.json"))
    if exclude_path in candidates:
        candidates.remove(exclude_path)
    if candidates:
        try:
            with open(candidates[-1], "r") as f:
                logger.info(f"Carrying regime state forward from local file: {candidates[-1]}")
                return json.load(f)
        except Exception as e:
            logger.warning(f"Found local previous-results file but couldn't parse it: {e}")

    try:
        from huggingface_hub import HfApi
        token = config.HF_TOKEN or os.environ.get("HF_TOKEN")
        api = HfApi(token=token)
        files = api.list_repo_files(config.RESULTS_REPO, repo_type="dataset")
        result_files = sorted(f for f in files if f.startswith("barycenter_results_") and f.endswith(".json"))
        if not result_files:
            return None
        latest = result_files[-1]
        local_path = api.hf_hub_download(repo_id=config.RESULTS_REPO, filename=latest, repo_type="dataset", token=token)
        with open(local_path, "r") as f:
            logger.info(f"Carrying regime state forward from HF results dataset: {latest}")
            return json.load(f)
    except Exception as e:
        logger.info(f"No previous results found for regime-persistence carryover ({e}); starting fresh.")
        return None


def run_trainer() -> Dict:
    """Main Dynamic-OT-Barycenter trainer with multi-window backtesting."""

    logger.info("🔄 Loading data...")
    try:
        prices_df, macro_df = load_master_data()
        validate_data(prices_df, macro_df)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return {}

    previous_results = _load_previous_results()
    previous_regime_state = (previous_results or {}).get("regime_state", {})

    quantile_levels = np.array(config.QUANTILE_LEVELS)
    run_date = datetime.now().strftime("%Y-%m-%d")
    results = {
        "run_date": run_date,
        "algorithm": "DYNAMIC-OT-BARYCENTER",
        "top_picks": {},
        "backtest_results": {},
        "best_window": {},
        "universes": {},
        "window_picks": {},
        "diagnostics": {},
        "regime_state": {},
    }

    for universe_name, tickers in config.UNIVERSES.items():
        logger.info(f"\n📊 Analyzing {universe_name}...")

        available = [t for t in tickers if t in prices_df.columns]
        if not available:
            continue

        universe_prices_df = prices_df[available]
        valid_mask = ~universe_prices_df.isna().any(axis=1)
        universe_prices_df = universe_prices_df[valid_mask]

        if len(universe_prices_df) < 200:
            logger.warning(f"Not enough data for {universe_name}")
            continue

        returns = np.diff(np.log(universe_prices_df.values), axis=0)
        dates = [d.strftime("%Y-%m-%d") for d in universe_prices_df.index[1:]]

        # Precompute the barycenter time series ONCE per distinct
        # local_window value used anywhere in the grid, instead of
        # recomputing it inside every backtest iteration — the same
        # "compute the expensive rolling thing once, slice it many times"
        # pattern used across the engine suite to stay inside the GitHub
        # Actions free-tier CPU budget.
        distinct_local_windows = sorted({
            combo.get("local_window", config.BARYCENTER_CONFIG["local_window"])
            for combo in config.BARYCENTER_GRID
        })
        barycenter_by_lw = {}
        for lw in distinct_local_windows:
            logger.info(f"  Building barycenter series (local_window={lw})...")
            b_series, _ = compute_barycenter_series(returns, lw, quantile_levels)
            barycenter_by_lw[lw] = b_series

        base_cfg = config.BARYCENTER_CONFIG.copy()
        base_cfg["trading_cost_bps"] = config.TRADING_COST_BPS

        window_results = {}
        window_hyperparams = {}
        for window in config.WINDOWS:
            logger.info(f"  Testing window {window} ({len(config.BARYCENTER_GRID)} hyperparameter combos)...")
            result = backtest_window_with_search(returns, barycenter_by_lw, window, base_cfg, config.BARYCENTER_GRID)

            if "error" not in result:
                window_results[window] = result
                window_hyperparams[window] = {
                    "local_window": result["local_window"],
                    "n_regimes": result["n_regimes"],
                }
                logger.info(f"    Best combo: local_window={result['local_window']}, "
                           f"n_regimes={result['n_regimes']} -> "
                           f"Correlation: {result['correlation']:.4f}, "
                           f"Directional: {result['directional_accuracy']:.2%}, "
                           f"Sharpe (net of {config.TRADING_COST_BPS}bps costs): {result['sharpe']:.2f} "
                           f"(gross: {result['sharpe_gross']:.2f}), "
                           f"n={result['n_predictions']}, "
                           f"In-sample R²: {result['avg_in_sample_r2']:.4f}, "
                           f"Regime switch rate: {result['avg_regime_switch_rate']:.2%}")
            else:
                logger.warning(f"    {result['error']}")

        # Best window selection is GATED + BLENDED (see
        # _select_best_window's docstring), not raw correlation alone —
        # a window whose predictions are wrong more than half the time is
        # never eligible, and among eligible windows the pick blends
        # correlation with net Sharpe rather than trusting a fractional
        # correlation edge that's itself within noise.
        if window_results:
            best_w, best_metrics, gate_passed = _select_best_window(window_results, config.MIN_DIRECTIONAL_ACCURACY)
            results["best_window"][universe_name] = {
                "window": best_w,
                "metrics": best_metrics,
                "selected_by": "directional_accuracy_gate+blended(correlation,sharpe)",
                "gate_passed": gate_passed,
                "min_directional_accuracy_required": config.MIN_DIRECTIONAL_ACCURACY,
            }
            gate_note = "" if gate_passed else " ⚠️ NO window cleared the directional-accuracy gate — falling back to the best of a bad lot; confidence will reflect this."
            logger.info(f"  ✅ Best window for {universe_name}: {best_w} "
                       f"(correlation={best_metrics.get('correlation', 0):.4f}, "
                       f"directional_accuracy={best_metrics.get('directional_accuracy', 0):.4f}, "
                       f"Sharpe: {best_metrics.get('sharpe', 0):.2f}){gate_note}")

        results["backtest_results"][universe_name] = window_results

        best_win = results["best_window"].get(universe_name, {}).get("window", 252)

        results["window_picks"][universe_name] = {}
        results["diagnostics"][universe_name] = {}
        results["regime_state"][universe_name] = {}
        best_win_ticker_results = {}
        best_win_diag = {}

        prev_universe_state = previous_regime_state.get(universe_name, {})

        for window in config.WINDOWS:
            # Use the SAME hyperparameter combination that won this
            # window's backtest, so live picks reflect the regime set
            # that was actually validated, not a different default guess.
            hp = window_hyperparams.get(window, {})
            lw = hp.get("local_window", config.BARYCENTER_CONFIG["local_window"])
            nr = hp.get("n_regimes", config.BARYCENTER_CONFIG["n_regimes"])
            b_series = barycenter_by_lw.get(lw)
            if b_series is None:
                b_series, _ = compute_barycenter_series(returns, lw, quantile_levels)
                barycenter_by_lw[lw] = b_series

            metrics = window_results.get(window, {})

            # Only carry the previous regime forward if the winning
            # hyperparameters for this window haven't changed since last
            # run — otherwise the old regime index doesn't even refer to
            # the same cluster set, so a fresh unconditional pick is
            # correct, not a bug.
            prev_state = prev_universe_state.get(str(window), {})
            previous_regime = None
            if prev_state.get("n_regimes") == nr and prev_state.get("local_window") == lw:
                previous_regime = prev_state.get("regime")

            picks, ticker_results, diag = compute_ticker_picks(
                returns, b_series, dates, available, window, nr, base_cfg, config.TOP_N,
                oos_correlation=metrics.get("correlation", 0.0),
                oos_directional_accuracy=metrics.get("directional_accuracy", 0.5),
                previous_regime=previous_regime,
            )
            results["window_picks"][universe_name][window] = picks
            results["diagnostics"][universe_name][window] = diag
            results["regime_state"][universe_name][str(window)] = {
                "regime": diag.get("regime"),
                "n_regimes": nr,
                "local_window": lw,
            }
            if window == best_win:
                best_win_ticker_results = ticker_results
                best_win_diag = diag

        picks = results["window_picks"][universe_name].get(best_win, [])
        if not best_win_ticker_results:
            hp = window_hyperparams.get(best_win, {})
            lw = hp.get("local_window", config.BARYCENTER_CONFIG["local_window"])
            nr = hp.get("n_regimes", config.BARYCENTER_CONFIG["n_regimes"])
            b_series = barycenter_by_lw.get(lw)
            metrics = window_results.get(best_win, {})
            prev_state = prev_universe_state.get(str(best_win), {})
            previous_regime = prev_state.get("regime") if (
                prev_state.get("n_regimes") == nr and prev_state.get("local_window") == lw
            ) else None
            picks, best_win_ticker_results, best_win_diag = compute_ticker_picks(
                returns, b_series, dates, available, best_win, nr, base_cfg, config.TOP_N,
                oos_correlation=metrics.get("correlation", 0.0),
                oos_directional_accuracy=metrics.get("directional_accuracy", 0.5),
                previous_regime=previous_regime,
            )

        results["top_picks"][universe_name] = picks
        results["universes"][universe_name] = {
            "tickers": available,
            "best_window": best_win,
            "ticker_results": best_win_ticker_results,
            "diagnostics": best_win_diag,
        }

        logger.info(f"  ✅ Top picks for {universe_name}:")
        for pick in picks:
            logger.info(f"     {pick['ticker']}: {pick['expected_return']}% ({pick['confidence']})")

    output_path = f"barycenter_results_{run_date}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\n💾 Saved: {output_path}")

    try:
        from push_results import upload_results
        upload_results(output_path, hf_token=config.HF_TOKEN)
    except Exception as e:
        logger.warning(f"Could not upload results: {e}")

    return results


if __name__ == "__main__":
    run_trainer()
