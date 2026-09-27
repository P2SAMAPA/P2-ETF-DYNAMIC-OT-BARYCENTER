"""
barycenter_model.py  —  Dynamic Wasserstein-Barycenter regime signal

Theory
------
Instead of asking "what is today's return for each ETF", this engine asks a
distributional question: "what is today's market DISTRIBUTION, and which
historical market state does it most resemble — not in raw Euclidean
feature space, but in the geometry of optimal transport?"

  Per-ETF distribution (mu_ETFi)
        On each day t, ETF i's "local distribution" mu_ETFi(t) is the
        empirical distribution of that ETF's own daily log returns over
        the trailing `local_window` days (e.g. the last 21 trading days).
        It is represented by its quantile function — the value at each of
        a fixed grid of quantile levels (2%, ..., 98%) — because for 1-D
        distributions the quantile function is exactly the representation
        under which Wasserstein geometry becomes linear:

            W2(mu, nu)^2  ==  || Q_mu - Q_nu ||_2^2

        i.e. the (squared) 2-Wasserstein distance between two 1-D
        distributions is just the ordinary Euclidean distance between
        their quantile functions. This single fact is what makes
        everything below tractable on free-tier CPU — no transport-plan
        solver appears anywhere in this engine.

  Barycenter (mu_t)
        The 2-Wasserstein barycenter of a set of 1-D distributions with
        equal weights is, by the same fact, simply the elementwise AVERAGE
        of their quantile functions. So "today's market distribution"

            mu_t = Barycenter( mu_ETF1(t), mu_ETF2(t), ..., mu_ETFn(t) )

        is computed by averaging every ticker's quantile function in the
        universe on day t. It is "dynamic" because it is recomputed every
        trading day from a rolling window — there is no fixed reference
        distribution baked in.

  Historical regimes
        Rolling mu_t back across history gives a time series of daily
        market-distribution barycenters. These are clustered — K-means on
        the quantile-function representation, again exactly equivalent to
        clustering in Wasserstein space — into a small number of regime
        centroids: recurring "shapes" of market-wide return distribution
        (e.g. calm-and-positive, broad risk-off, dispersed/rotational).

  Signal
        Today's mu_t is compared, via the same Euclidean-on-quantiles
        distance, to every regime centroid. The nearest regime directly
        answers the question in the prompt: "which historical market
        state is today's distribution closest to, in distribution
        geometry rather than Euclidean feature space." Each ticker's
        forecast return is that regime's historical average next-day
        return for that ticker — computed ONLY from training-period days
        assigned to the regime, so this is a genuine walk-forward lookup,
        never a look-ahead.

  Regime distance ("distributional energy")
        The Wasserstein distance from today's mu_t to its nearest regime
        centroid is reported alongside the forecast as a market-
        dislocation index: a large distance means today's distribution
        doesn't closely resemble any recurring historical regime — itself
        information (novelty / regime-transition risk) distinct from a
        small distance, which means today looks like a well-worn,
        historically well-characterized market state.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans


def compute_barycenter_series(returns: np.ndarray, local_window: int,
                               quantile_levels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    returns: (n_days, n_tickers) daily log returns.

    Returns:
      barycenter_series : (n_days, n_quantiles) — the equal-weight
                          Wasserstein barycenter of every ticker's local
                          distribution on that day. NaN rows where fewer
                          than `local_window` trailing days are available.
      ticker_quantiles  : (n_days, n_tickers, n_quantiles) — the same, kept
                          per-ticker for diagnostics. Same NaN convention.

    Single pass over days — O(n_days), not O(n_days^2) — since every day's
    quantile function for every ticker is produced by one vectorized
    np.percentile call over that day's trailing window, rather than being
    recomputed inside a nested backtest loop later (see trainer.py).
    """
    n_days, n_tickers = returns.shape
    n_q = len(quantile_levels)
    pct = np.asarray(quantile_levels) * 100.0

    barycenter_series = np.full((n_days, n_q), np.nan)
    ticker_quantiles = np.full((n_days, n_tickers, n_q), np.nan)

    for t in range(local_window - 1, n_days):
        window = returns[t - local_window + 1: t + 1, :]     # (local_window, n_tickers)
        q = np.percentile(window, pct, axis=0).T              # (n_tickers, n_q)
        ticker_quantiles[t] = q
        barycenter_series[t] = q.mean(axis=0)                 # equal-weight barycenter

    return barycenter_series, ticker_quantiles


def wasserstein2_dist(q1: np.ndarray, q2: np.ndarray) -> float:
    """W2 distance between two 1-D distributions given as quantile
    functions == plain Euclidean distance between them (see module docstring)."""
    return float(np.sqrt(np.sum((np.asarray(q1) - np.asarray(q2)) ** 2)))


class RegimeBarycenterModel:
    """Fits historical regime centroids over a window of barycenters, and
    predicts next-day returns for today by nearest-regime lookup."""

    def __init__(self, n_regimes: int = 6, random_state: int = 42):
        self.n_regimes_requested = n_regimes
        self.random_state = random_state
        self.kmeans: Optional[KMeans] = None
        self.regime_avg_return_: Optional[np.ndarray] = None   # (k, n_tickers)
        self.regime_counts_: Optional[np.ndarray] = None       # (k,)
        self.global_avg_return_: Optional[np.ndarray] = None   # (n_tickers,) fallback for empty clusters
        self.in_sample_r2_: float = 0.0

    def fit(self, B_train: np.ndarray, returns_next_train: np.ndarray) -> Dict:
        """
        B_train:            (n_train, n_quantiles) barycenter on each training day.
        returns_next_train: (n_train, n_tickers) the NEXT day's return following
                             each of those training days (already aligned by caller).
        """
        n_train = B_train.shape[0]
        # Degrade the requested cluster count gracefully on small windows,
        # rather than fitting near-empty clusters.
        k = max(2, min(self.n_regimes_requested, n_train // 5))

        self.kmeans = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
        labels = self.kmeans.fit_predict(B_train)

        n_tickers = returns_next_train.shape[1]
        self.global_avg_return_ = returns_next_train.mean(axis=0)
        self.regime_avg_return_ = np.zeros((k, n_tickers))
        self.regime_counts_ = np.zeros(k, dtype=int)

        for r in range(k):
            mask = labels == r
            self.regime_counts_[r] = int(mask.sum())
            if mask.sum() > 0:
                self.regime_avg_return_[r] = returns_next_train[mask].mean(axis=0)
            else:
                self.regime_avg_return_[r] = self.global_avg_return_

        # In-sample R^2: how much of the pooled (ticker, day) next-return
        # variance is explained just by which regime that day was assigned
        # to — the piecewise-constant analogue of sheaf_model.py's pooled
        # regression R^2, here with a regime label standing in for the
        # continuous "gap".
        pred_pooled = self.regime_avg_return_[labels].flatten()
        actual_pooled = returns_next_train.flatten()
        ss_res = np.sum((actual_pooled - pred_pooled) ** 2)
        ss_tot = np.sum((actual_pooled - actual_pooled.mean()) ** 2)
        self.in_sample_r2_ = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

        return {
            "n_regimes_fit": k,
            "regime_counts": self.regime_counts_.tolist(),
            "in_sample_r2": self.in_sample_r2_,
        }

    def predict(self, B_today: np.ndarray) -> Dict:
        """Nearest-regime label + distance, runner-up distance (regime
        separation/confidence), and the predicted next-day return vector."""
        centers = self.kmeans.cluster_centers_
        dists = np.sqrt(np.sum((centers - np.asarray(B_today).reshape(1, -1)) ** 2, axis=1))
        order = np.argsort(dists)
        nearest = int(order[0])
        second = int(order[1]) if len(order) > 1 else nearest

        return {
            "regime": nearest,
            "distance": float(dists[nearest]),
            "second_distance": float(dists[second]),
            "separation": float(dists[second] - dists[nearest]),
            "predicted_returns": self.regime_avg_return_[nearest],
            "regime_size": int(self.regime_counts_[nearest]),
        }


def nearest_analog_days(B_today: np.ndarray, B_history: np.ndarray, dates_history: List[str],
                         returns_next_history: np.ndarray, tickers: List[str],
                         top_n: int = 5) -> List[Dict]:
    """
    Individual-day analogs (not regime centroids): the `top_n` historical
    days whose market-distribution barycenter is closest to today's, under
    the same quantile-space Euclidean/W2 distance — directly answering
    "which specific historical day did today's market distribution look
    most like." Purely descriptive/diagnostic; not used to form the
    tradeable forecast (that comes from the regime lookup above).
    """
    if len(B_history) == 0:
        return []
    dists = np.sqrt(np.sum((B_history - np.asarray(B_today).reshape(1, -1)) ** 2, axis=1))
    order = np.argsort(dists)[:top_n]
    out = []
    for idx in order:
        out.append({
            "date": dates_history[idx],
            "distance": float(dists[idx]),
            "next_day_returns": {t: round(float(returns_next_history[idx, j]), 5)
                                 for j, t in enumerate(tickers)},
        })
    return out


def get_barycenter_predictions(barycenter_series: np.ndarray, returns: np.ndarray,
                                window: int, n_regimes: int, min_train_samples: int = 60) -> Dict:
    """
    Fit a RegimeBarycenterModel on the most recent `window` valid days and
    produce today's (the latest available day's) regime assignment and
    1-step-ahead return forecast for every ticker in the universe.
    """
    valid = ~np.isnan(barycenter_series).any(axis=1)
    valid_idx = np.where(valid)[0]
    if len(valid_idx) < 2:
        raise ValueError("Not enough valid barycenter days.")

    today_idx = valid_idx[-1]
    train_idx = valid_idx[(valid_idx >= today_idx - window) & (valid_idx < today_idx)]
    train_idx = train_idx[train_idx + 1 <= today_idx]

    if len(train_idx) < min_train_samples:
        raise ValueError("Not enough training days for regime fit.")

    B_train = barycenter_series[train_idx]
    returns_next_train = returns[train_idx + 1]

    model = RegimeBarycenterModel(n_regimes=n_regimes)
    fit_result = model.fit(B_train, returns_next_train)

    B_today = barycenter_series[today_idx]
    pred = model.predict(B_today)

    return {
        "model": model,
        "fit_result": fit_result,
        "prediction": pred,
        "today_idx": int(today_idx),
        "train_idx": train_idx,
    }
