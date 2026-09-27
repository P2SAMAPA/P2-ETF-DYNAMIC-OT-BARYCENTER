"""
trainer.py  —  P2 DYNAMIC-OT-BARYCENTER Trainer with Multi-Window Backtesting

Builds a dynamic Wasserstein barycenter of each universe's ETF return
distributions, clusters its historical trajectory into recurring "market
regimes," and tests whether today's nearest regime (in Wasserstein /
quantile-function distance) predicts next-day returns — across several
regime-fit window sizes and (local_window, n_regimes) hyperparameter
combinations. See barycenter_model.py for the theory.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_manager import load_master_data, validate_data
from barycenter_model import compute_barycenter_series, RegimeBarycenterModel, nearest_analog_days

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
    """
    n_samples = len(returns)
    valid = ~np.isnan(barycenter_series).any(axis=1)
    valid_idx = np.where(valid)[0]
    if len(valid_idx) < window + 50:
        return {"error": "Insufficient data", "window": window}

    burn_in_fraction = cfg.get("burn_in_fraction", 0.05)
    min_train_samples = cfg.get("min_train_samples", 60)
    start_pos = int(len(valid_idx) * burn_in_fraction)

    predictions, actuals, distances, separations, r2s = [], [], [], [], []

    for pos in range(start_pos, len(valid_idx) - 1):
        i = valid_idx[pos]                      # "today" — must itself have a valid barycenter
        if i + 1 >= n_samples:
            continue

        train_idx = valid_idx[(valid_idx >= i - window) & (valid_idx < i)]
        train_idx = train_idx[train_idx + 1 <= i]
        if len(train_idx) < min_train_samples:
            continue

        try:
            B_train = barycenter_series[train_idx]
            returns_next_train = returns[train_idx + 1]

            model = RegimeBarycenterModel(n_regimes=n_regimes)
            fit_result = model.fit(B_train, returns_next_train)

            pred = model.predict(barycenter_series[i])
            actual_returns = returns[i + 1]

            predictions.append(pred["predicted_returns"])
            actuals.append(actual_returns)
            distances.append(pred["distance"])
            separations.append(pred["separation"])
            r2s.append(fit_result["in_sample_r2"])
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
    # day a position is simply held.
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


def _confidence_from_r2(r2: float) -> str:
    if r2 > 0.01:
        return "High"
    elif r2 > 0.002:
        return "Medium"
    return "Low"


def compute_ticker_picks(returns: np.ndarray, barycenter_series: np.ndarray, dates: List[str],
                          tickers: List[str], window: int, n_regimes: int, cfg: Dict,
                          top_n: int) -> Tuple[List[Dict], Dict, Dict]:
    """
    Fit regimes on the most recent `window` valid days and return the
    top-N ETF picks by regime-conditional predicted next-day return, plus
    per-ticker detail and run diagnostics — including a 2D "regime map"
    for the dashboard, the today-vs-regime-centroid quantile curves, and
    the individual historical days today's distribution most resembles.
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
        pred = model.predict(barycenter_series[today_idx])
    except Exception as e:
        logger.error(f"  Barycenter fit failed (window={window}): {e}")
        return [], {}, {}

    pred_returns = pred["predicted_returns"]
    confidence = _confidence_from_r2(fit_result["in_sample_r2"])

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
    }

    return picks, ticker_results, diagnostics


def run_trainer() -> Dict:
    """Main Dynamic-OT-Barycenter trainer with multi-window backtesting."""

    logger.info("🔄 Loading data...")
    try:
        prices_df, macro_df = load_master_data()
        validate_data(prices_df, macro_df)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return {}

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
                           f"In-sample R²: {result['avg_in_sample_r2']:.4f}")
            else:
                logger.warning(f"    {result['error']}")

        # Best window selected by RETURN-PREDICTION quality (correlation
        # between predicted and actual returns), not by backtested Sharpe
        # — for the same reason as the sheaf engine: Sharpe reflects
        # realized P&L, which can look good even from regime assignments
        # that barely explain any real variance.
        select_metric = config.BEST_WINDOW_METRIC
        if window_results:
            best_window = max(window_results.items(), key=lambda x: x[1].get(select_metric, -999))
            results["best_window"][universe_name] = {
                "window": best_window[0],
                "metrics": best_window[1],
                "selected_by": select_metric,
            }
            logger.info(f"  ✅ Best window for {universe_name}: {best_window[0]} "
                       f"(selected by {select_metric}={best_window[1][select_metric]:.4f}; "
                       f"Sharpe: {best_window[1]['sharpe']:.2f})")

        results["backtest_results"][universe_name] = window_results

        best_win = results["best_window"].get(universe_name, {}).get("window", 252)

        results["window_picks"][universe_name] = {}
        results["diagnostics"][universe_name] = {}
        best_win_ticker_results = {}
        best_win_diag = {}

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

            picks, ticker_results, diag = compute_ticker_picks(
                returns, b_series, dates, available, window, nr, base_cfg, config.TOP_N
            )
            results["window_picks"][universe_name][window] = picks
            results["diagnostics"][universe_name][window] = diag
            if window == best_win:
                best_win_ticker_results = ticker_results
                best_win_diag = diag

        picks = results["window_picks"][universe_name].get(best_win, [])
        if not best_win_ticker_results:
            hp = window_hyperparams.get(best_win, {})
            lw = hp.get("local_window", config.BARYCENTER_CONFIG["local_window"])
            nr = hp.get("n_regimes", config.BARYCENTER_CONFIG["n_regimes"])
            b_series = barycenter_by_lw.get(lw)
            picks, best_win_ticker_results, best_win_diag = compute_ticker_picks(
                returns, b_series, dates, available, best_win, nr, base_cfg, config.TOP_N
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
