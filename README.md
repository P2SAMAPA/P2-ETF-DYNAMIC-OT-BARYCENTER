# P2-ETF-DYNAMIC-OT-BARYCENTER

Regime-detection alpha via **dynamic Wasserstein barycenters** over ETF return
distributions.

## The idea

Instead of asking *"what is today's return for each ETF"*, this engine asks a
distributional question: *"what is today's market DISTRIBUTION, and which
historical market state does it most resemble — not in Euclidean feature
space, but in the geometry of optimal transport?"*

- **Per-ETF distribution (`mu_ETFi`)** — on day `t`, each ETF's own trailing
  21-day (`local_window`) empirical return distribution, represented by its
  **quantile function** (value at each of 25 fixed quantile levels, 2%–98%).
  For 1-D distributions the quantile function is exactly the representation
  under which Wasserstein geometry becomes linear:
  `W2(mu, nu)^2 == ||Q_mu - Q_nu||_2^2` — the squared 2-Wasserstein distance
  between two 1-D distributions is just ordinary Euclidean distance between
  their quantile functions. This single fact is what makes the whole engine
  tractable on free-tier CPU: **no transport-plan solver appears anywhere.**
- **Barycenter (`mu_t`)** — the 2-Wasserstein barycenter of a set of 1-D
  distributions with equal weights is, by the same fact, simply the
  elementwise **average of their quantile functions**. So
  `mu_t = Barycenter(mu_ETF1(t), mu_ETF2(t), ..., mu_ETFn(t))` is today's
  "market distribution," recomputed every trading day from a rolling window
  — there is no fixed reference distribution.
- **Historical regimes** — rolling `mu_t` back across history gives a time
  series of daily market-distribution barycenters. K-means clusters them
  (on the quantile-function representation — again exactly equivalent to
  clustering in Wasserstein space) into a small number of recurring
  "market-distribution shapes."
- **Signal** — today's `mu_t` is compared, via the same Euclidean-on-quantiles
  distance, to every regime centroid. The nearest regime directly answers
  *"which historical market state is today's distribution closest to, in
  distribution geometry rather than Euclidean feature space."* Each ticker's
  forecast return is that regime's historical average next-day return,
  computed only from training-period days assigned to the regime — a genuine
  walk-forward lookup, never a look-ahead.
- **Regime distance ("distributional energy")** — the Wasserstein distance
  from today's `mu_t` to its nearest regime centroid is reported as a
  market-dislocation index: large distance means today doesn't closely
  resemble any recurring historical regime (novelty / regime-transition
  risk); small distance means today looks like a well-worn historical state.

## Repo structure

Same shape as the sister repos (`P2-ETF-SINDY-ETF-DYNAMICS`,
`P2-ETF-SHEAF-ALPHA`), swapping the algorithm:

```
config.py            Universes, windows, HF repo IDs, QUANTILE_LEVELS,
                       BARYCENTER_CONFIG, BARYCENTER_GRID
data_manager.py       Loads the master parquet from HF (unchanged from
                       sister repos); macro loaded only for diagnostics
barycenter_model.py    compute_barycenter_series, RegimeBarycenterModel
                       (K-means regime fit + nearest-regime lookup),
                       nearest_analog_days, get_barycenter_predictions
trainer.py             Walk-forward backtesting across regime-fit window
                       sizes + (local_window, n_regimes) hyperparameter
                       grid, per-window ETF picks, regime map / quantile
                       curve diagnostics for the dashboard
push_results.py        Uploads results JSON to the HF results dataset
                       (unchanged; HfApi.upload_file, never HfFileSystem.open)
us_calendar.py          Trading-day utilities (unchanged)
streamlit_app.py        Dashboard: live picks + backtest diagnostics,
                       including a real 2D regime map (PCA) and a
                       today-vs-regime quantile-curve overlay
requirements.txt        Same dependency set as the sheaf repo, minus networkx
                       (no network graph here — the barycenter's native
                       visual is a distribution/regime map, not a graph)
```

## Running it

```bash
pip install -r requirements.txt
python trainer.py          # walk-forward backtest + live picks, pushes to HF
streamlit run streamlit_app.py
```

`HF_TOKEN` must be set in the environment for `push_results.py` to upload.
`config.RESULTS_REPO` currently points at
`P2SAMAPA/p2-dynamic-ot-barycenter-results` — create that dataset repo on HF
before the first run, or change the value.

> **Note on this delivery**: this sandbox's network access does not include
> `huggingface.co`, so the code here was validated by inspecting the sibling
> `P2-ETF-SHEAF-ALPHA` repo's exact data/push conventions and mirroring them
> line-for-line, plus reasoning through the barycenter math and walk-forward
> indexing by hand — not against the real master parquet or a live HF push.
> Please run `trainer.py` yourself against the real data as the first real
> test, the same way you would for any new engine.

## What the backtest reports, per window per universe

Same metric shape as the sheaf repo (correlation, MSE, directional accuracy,
mean/std return, n_predictions) plus barycenter-specific diagnostics:

- `sharpe` / `sharpe_gross` — Sharpe **net** of trading costs and **before**
  costs, so the cost drag is visible directly.
- `avg_daily_cost_bps` — average trading cost actually paid per day
  (turnover-driven).
- `avg_regime_distance` / `avg_regime_separation` — how far, on average,
  today's barycenter sat from its nearest regime centroid, and how much
  further the 2nd-nearest regime was (a proxy for how "clean" the regime
  assignment was).
- `avg_in_sample_r2` — how much of the pooled next-return variance the
  regime assignment explained in training, averaged across walk-forward
  steps (piecewise-constant analogue of the sheaf engine's pooled-regression
  R²).
- `distance_series` — the full per-day regime-distance time series for the
  best window's test period, plotted in the dashboard as a market-
  dislocation index.
- `local_window` / `n_regimes` — the hyperparameter combination that won
  this window's search.
- `hyperparam_search` — the full comparison table of every combination
  tried for this window, not just the winner.

Additional per-pick diagnostics (`window_picks` / `diagnostics` in the
results JSON, surfaced in the dashboard):

- `regime_map` — a 2D PCA projection of the training barycenter matrix,
  colored by regime, with today's point and every regime centroid marked —
  computed once in `trainer.py` so the dashboard never has to recompute
  anything to render it.
- `quantile_curves` — today's barycenter quantile function vs. its nearest
  regime centroid's, for the "what does today's distribution actually look
  like" overlay chart.
- `nearest_analog_days` — the individual historical days (not just regime
  centroids) whose barycenter is closest to today's, with what each of
  today's picked tickers did the next day on those analog days.

## Trading costs

Same convention as the sheaf engine: `backtest_window` applies a
**turnover-based** cost (`config.TRADING_COST_BPS`, default 15bps), charged
only when a ticker's position actually changes sign from one day to the
next — a position held unchanged costs nothing extra.

## How the "best window" and barycenter hyperparameters are selected

Same rationale as the sheaf engine: **Sharpe is not used to select
anything**, because it reflects realized P&L, which can look good from a
regime assignment that barely explains any real variance if it happens to
ride the test period's market drift.

- **Best window** (`config.BEST_WINDOW_METRIC`, default `"correlation"`):
  selected by predicted-vs-actual return correlation.
- **Barycenter hyperparameters** (`config.BARYCENTER_GRID`): for each
  window, a small grid of `(local_window, n_regimes)` combinations is
  backtested, and the one with the best out-of-sample correlation is used
  both for that window's reported metrics AND for generating that window's
  live picks (so the two are always consistent with each other).

**Multiple-comparisons caveat, stated plainly**: searching more combinations
increases the chance that the "best" one simply got lucky on this
particular test period, even if no combination is actually better than any
other. The full `hyperparam_search` comparison table is kept and surfaced in
the dashboard (Tab 2, per window) for exactly this reason.

## Backtest coverage (`BARYCENTER_CONFIG["burn_in_fraction"]`)

Same reasoning as the sheaf engine: each window is only ever trained on the
`window` days immediately preceding a given test point (enforced by a
per-window guard, regardless of this setting) — `burn_in_fraction` just
controls how early in the dataset's history walk-forward testing is
*allowed* to start. Kept small (0.05) so the out-of-sample sample size isn't
needlessly shrunk, since a large global burn-in only wastes usable days
without reducing any individual window's training data.

## Caveats (same spirit as the sister repos')

- The barycenter is an **equal-weight** average of each ticker's quantile
  function — a wide-universe barycenter (e.g. `COMBINED`, 43 tickers) is
  dominated by whatever's common across the universe, not by any single
  high-vol ticker, by construction. This is a design choice, not a bug: a
  volatility-weighted or ADV-weighted barycenter is a reasonable variant to
  try if equal-weighting proves too blunt.
- K-means on a rolling window re-discovers regimes from scratch at every
  walk-forward step — regime *labels* are not stable across time (regime 2
  in one window isn't "the same" regime 2 in another), only regime
  *behavior* (its historical average forward return) is used downstream.
- A positive backtest Sharpe over a few hundred walk-forward days is weak
  statistical evidence of durable edge, not strong evidence — these are
  daily-correlated observations, not independent trials.
- The multiple-comparisons risk from the hyperparameter grid search (above)
  applies on top of the usual sample-size caveat.
- This is not financial advice, and none of this should be traded live
  without further validation and sanity-checking that results aren't driven
  by a handful of outlier days.
