import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import glob
import os
import plotly.graph_objects as go

st.set_page_config(
    page_title="P2 Dynamic OT-Barycenter",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# Theme
# ----------------------------------------------------------------------------
PRIMARY = "#0d9488"      # teal -- distinct identity from sibling dashboards (violet Sheaf, ...)
POSITIVE = "#16a34a"
NEGATIVE = "#dc2626"
NEUTRAL = "#d97706"
INK = "#0f172a"
SUBTLE = "#64748b"
CARD_BG = "#ffffff"
CARD_BORDER = "#e2e8f0"
PAGE_BG = "#f8fafc"
TODAY_COLOR = "#0f172a"
REGIME_PALETTE = ["#0d9488", "#7c3aed", "#dc2626", "#d97706", "#2563eb", "#db2777", "#65a30d", "#0891b2"]

CONF_COLORS = {"high": POSITIVE, "medium": NEUTRAL, "low": SUBTLE}

st.markdown(f"""
<style>
    .stApp {{ background-color: {PAGE_BG}; }}
    #MainMenu, footer {{visibility: hidden;}}

    .app-header {{ display: flex; align-items: baseline; gap: 0.75rem; margin-bottom: 0; }}
    .app-title {{ font-size: 1.9rem; font-weight: 800; color: {INK}; margin: 0; }}
    .app-subtitle {{ color: {SUBTLE}; font-size: 0.95rem; margin-top: 0.15rem; margin-bottom: 1.25rem; }}

    .universe-heading {{ font-size: 1.15rem; font-weight: 700; color: {INK}; margin: 0 0 0.15rem 0; }}
    .universe-caption {{ color: {SUBTLE}; font-size: 0.85rem; margin-bottom: 0.75rem; }}

    .pick-card {{
        background: {CARD_BG}; border: 1px solid {CARD_BORDER}; border-left: 4px solid var(--accent);
        border-radius: 12px; padding: 1.1rem 1.3rem; margin: 0.35rem 0;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }}
    .pick-ticker {{ font-size: 1.05rem; font-weight: 700; color: {INK}; letter-spacing: 0.02em; }}
    .pick-return {{ font-size: 1.9rem; font-weight: 800; color: {INK}; margin: 0.25rem 0 0.35rem 0; line-height: 1.1; }}
    .pick-badge {{
        display: inline-block; font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.04em; padding: 0.15rem 0.55rem; border-radius: 999px; color: white; background: var(--accent);
    }}

    .kpi-card {{ background: {CARD_BG}; border: 1px solid {CARD_BORDER}; border-radius: 12px; padding: 0.9rem 1.1rem; text-align: center; }}
    .kpi-value {{ font-size: 1.4rem; font-weight: 800; color: {INK}; }}
    .kpi-label {{ font-size: 0.72rem; color: {SUBTLE}; text-transform: uppercase; letter-spacing: 0.04em; margin-top: 0.15rem; }}

    .best-window-banner {{
        background: linear-gradient(90deg, #f0fdfa 0%, #ecfeff 100%); border: 1px solid #99f6e4;
        border-left: 4px solid {PRIMARY}; border-radius: 10px; padding: 0.7rem 1rem;
        font-size: 0.9rem; color: #134e4a; margin: 0.5rem 0 1rem 0;
    }}
    .regime-note {{
        font-size: 0.85rem; color: {SUBTLE}; margin: 0.5rem 0 0.75rem 0;
    }}
    .theory-note {{
        background: #f8fafc; border: 1px dashed {CARD_BORDER}; border-radius: 10px;
        padding: 0.75rem 1rem; font-size: 0.82rem; color: {SUBTLE}; margin: 0.75rem 0 1.25rem 0;
    }}

    div[data-testid="stDataFrame"] {{ border: 1px solid {CARD_BORDER}; border-radius: 10px; overflow: hidden; }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
RESULTS_REPO = "P2SAMAPA/p2-dynamic-ot-barycenter-results"


def _find_latest_hf_result(repo_id: str):
    """List actual files in the HF dataset and return the newest results
    file by name (YYYY-MM-DD sorts correctly lexically), instead of
    guessing that today's date matches when the training run happened."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        files = api.list_repo_files(repo_id, repo_type="dataset")
        result_files = sorted(f for f in files if f.startswith("barycenter_results_") and f.endswith(".json"))
        if not result_files:
            return None, f"No barycenter_results_*.json files found in {repo_id}."
        return result_files[-1], None
    except Exception as e:
        return None, f"Could not list files in {repo_id}: {e}"


@st.cache_data(show_spinner="Loading results...")
def load_data(_cache_key: str = ""):
    """
    Load the latest results, trying in order:
      1. A local barycenter_results_*.json in the working directory (fast
         path right after running trainer.py locally).
      2. The most recent barycenter_results_*.json actually present in the
         HF results dataset (not a guess at "today's" date -- training may
         have run on a different day than this dashboard is being viewed).
    Returns (data, notes) where notes is a list of human-readable strings
    describing what was tried, so a failure is diagnosable instead of a
    silent blank dashboard.
    """
    notes = []

    json_files = glob.glob("barycenter_results_*.json")
    if json_files:
        latest_local = sorted(json_files)[-1]
        try:
            with open(latest_local, "r") as f:
                return json.load(f), [f"Loaded local file: {latest_local}"]
        except Exception as e:
            notes.append(f"Found local file {latest_local} but couldn't parse it: {e}")

    latest_remote, list_err = _find_latest_hf_result(RESULTS_REPO)
    if list_err:
        notes.append(list_err)
    else:
        try:
            url = f"https://huggingface.co/datasets/{RESULTS_REPO}/resolve/main/{latest_remote}"
            headers = {}
            token = os.environ.get("HF_TOKEN")
            if token:
                headers["Authorization"] = f"Bearer {token}"
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                return response.json(), [f"Loaded from HF dataset: {latest_remote}"]
            notes.append(f"Fetching {latest_remote} from HF returned HTTP {response.status_code} "
                         f"(private dataset without a valid HF_TOKEN would show up this way).")
        except Exception as e:
            notes.append(f"Fetching {latest_remote} from HF failed: {e}")

    return None, notes


def conf_color(confidence: str) -> str:
    return CONF_COLORS.get((confidence or "low").lower(), SUBTLE)


def render_pick_cards(picks, key_prefix):
    if not picks:
        st.info("No ETF picks available for this selection.")
        return

    horizon = picks[0].get("horizon_days") if picks else None
    return_label = f"Expected {horizon}-day return" if horizon else "Expected return"

    cols = st.columns(min(len(picks), 3))
    for i, pick in enumerate(picks):
        color = conf_color(pick["confidence"])
        with cols[i % len(cols)]:
            st.markdown(f"""
            <div class="pick-card" style="--accent: {color};">
                <div class="pick-ticker">{pick['ticker']}</div>
                <div class="pick-return">{pick['expected_return']:+.2f}%</div>
                <div style="font-size:0.72rem; color:{SUBTLE}; margin-top:-0.2rem; margin-bottom:0.4rem;">{return_label}</div>
                <span class="pick-badge">{pick['confidence']} confidence</span>
            </div>
            """, unsafe_allow_html=True)


def render_analog_table(analogs, horizon=None):
    if not analogs:
        st.info("No historical analog days available for this selection.")
        return
    fwd_label = f"{horizon}-day fwd %" if horizon else "fwd %"
    rows = []
    for a in analogs:
        row = {"Date": a["date"], "Distributional distance": round(a["distance"], 3)}
        for ticker, ret in a.get("forward_returns", {}).items():
            row[f"{ticker} {fwd_label}"] = round(ret * 100, 2)
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_regime_map(regime_map, key):
    if not regime_map or not regime_map.get("points"):
        st.info("No regime map available for this selection.")
        return

    pts = pd.DataFrame(regime_map["points"])
    fig = go.Figure()

    for regime_id in sorted(pts["regime"].unique()):
        sub = pts[pts["regime"] == regime_id]
        color = REGIME_PALETTE[regime_id % len(REGIME_PALETTE)]
        fig.add_trace(go.Scatter(
            x=sub["x"], y=sub["y"], mode="markers", name=f"Regime {regime_id}",
            marker=dict(size=7, color=color, opacity=0.55),
            text=sub["date"], hovertemplate="%{text}<extra>Regime " + str(regime_id) + "</extra>",
        ))

    for c in regime_map.get("centers", []):
        color = REGIME_PALETTE[c["regime"] % len(REGIME_PALETTE)]
        fig.add_trace(go.Scatter(
            x=[c["x"]], y=[c["y"]], mode="markers",
            marker=dict(size=16, color=color, symbol="diamond", line=dict(width=2, color="white")),
            showlegend=False, hovertemplate=f"Regime {c['regime']} centroid<extra></extra>",
        ))

    today = regime_map.get("today")
    if today:
        fig.add_trace(go.Scatter(
            x=[today["x"]], y=[today["y"]], mode="markers+text", name="Today",
            marker=dict(size=18, color=TODAY_COLOR, symbol="star", line=dict(width=2, color="white")),
            text=["Today"], textposition="top center",
            textfont=dict(size=11, color=TODAY_COLOR, family="Arial Black"),
        ))

    ev = regime_map.get("explained_variance", [0, 0])
    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title=f"PC1 ({ev[0]*100:.0f}% var)", gridcolor="#eef2f7", zeroline=False),
        yaxis=dict(title=f"PC2 ({ev[1]*100:.0f}% var)" if len(ev) > 1 else "PC2", gridcolor="#eef2f7", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"regimemap_{key}")
    st.caption("Each point is one historical trading day's market-distribution barycenter, projected to 2D. "
               "Diamonds are the fitted regime centroids; the star is today. Distance on this map approximates "
               "Wasserstein distance between market-wide return distributions — not Euclidean distance in raw returns.")


def render_quantile_curve(quantile_curves, key):
    if not quantile_curves:
        st.info("No distribution curve available for this selection.")
        return
    levels = quantile_curves["quantile_levels"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=levels, y=quantile_curves["today"], mode="lines+markers", name="Today's barycenter",
        line=dict(color=TODAY_COLOR, width=2.5), marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=levels, y=quantile_curves["nearest_regime_centroid"], mode="lines+markers", name="Nearest regime centroid",
        line=dict(color=PRIMARY, width=2.5, dash="dot"), marker=dict(size=5),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#cbd5e1")
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="Quantile level", tickformat=".0%", gridcolor="#eef2f7"),
        yaxis=dict(title="Daily log return", tickformat=".2%", gridcolor="#eef2f7"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, key=f"qcurve_{key}")
    st.caption("The market-wide return distribution implied by today's barycenter (solid) vs. its nearest "
               "historical regime centroid (dotted). Where the two curves diverge is where today's distribution "
               "shape departs from its closest historical analog — e.g. a fatter left tail, a shifted median.")


def render_distance_chart(distance_series, key):
    if not distance_series:
        return
    x = list(range(len(distance_series)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=distance_series, mode="lines", fill="tozeroy",
        line=dict(color=PRIMARY, width=1.5), fillcolor="rgba(13, 148, 136, 0.12)",
        hovertemplate="Day %{x}: distance %{y:.2f}<extra></extra>",
    ))
    avg = float(np.mean(distance_series))
    fig.add_hline(y=avg, line_dash="dot", line_color=SUBTLE, annotation_text=f"avg {avg:.2f}", annotation_font_size=10)
    fig.update_layout(
        height=220, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="Test period (non-overlapping, spaced by horizon)", gridcolor="#eef2f7"),
        yaxis=dict(title="Distance to nearest regime", gridcolor="#eef2f7"),
        font=dict(color=INK, size=11),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"distance_{key}")


def render_window_comparison(df_results, best_idx, key):
    """Net Sharpe + directional accuracy across regime-fit window sizes -- a line chart, not a bar chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_results["Window"], y=df_results["Net Sharpe"],
        mode="lines+markers", name="Net Sharpe (after costs)",
        line=dict(color=PRIMARY, width=2.5), marker=dict(size=9),
    ))
    fig.add_trace(go.Scatter(
        x=df_results["Window"], y=df_results["Directional Accuracy"],
        mode="lines+markers", name="Directional Accuracy %",
        line=dict(color=POSITIVE, width=2.5, dash="dot"), marker=dict(size=9),
        yaxis="y2",
    ))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="Regime-Fit Window Size (days)", gridcolor="#eef2f7"),
        yaxis=dict(title="Net Sharpe Ratio", gridcolor="#eef2f7"),
        yaxis2=dict(title="Directional Accuracy (%)", overlaying="y", side="right"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        font=dict(color=INK, size=12),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"windowcmp_{key}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    if "cache_bust" not in st.session_state:
        st.session_state.cache_bust = 0

    header_col, refresh_col = st.columns([5, 1])
    with header_col:
        st.markdown('<div class="app-header"><span style="font-size:1.9rem;">🌊</span>'
                     '<span class="app-title">P2 Dynamic OT-Barycenter</span></div>', unsafe_allow_html=True)
        st.markdown(
            "<div class=\"app-subtitle\">Wasserstein barycenters over ETF return distributions "
            "— comparing today's market state to historical regimes</div>",
            unsafe_allow_html=True,
        )
    with refresh_col:
        st.markdown("<div style='margin-top: 0.6rem;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh data", use_container_width=True,
                      help="Clear the cache and re-check for the latest results"):
            st.cache_data.clear()
            st.session_state.cache_bust += 1
            st.rerun()

    data, notes = load_data(_cache_key=str(st.session_state.cache_bust))

    if not data:
        st.error("No results found.")
        with st.expander("Why? (click for details)", expanded=True):
            if notes:
                for n in notes:
                    st.write(f"• {n}")
            else:
                st.write("• No diagnostic information was returned.")
            st.write(
                "Run `python trainer.py` to generate a local `barycenter_results_*.json`, "
                "or confirm the HF dataset repo has a results file and (if private) "
                "that `HF_TOKEN` is set in this app's environment. Then hit **🔄 Refresh data** above."
            )
        return

    run_date = data.get("run_date", "Unknown")
    st.caption(f"🕒 Results generated: **{run_date}**")
    if notes:
        st.caption(f"📡 {notes[0]}")

    tab1, tab2 = st.tabs(["🔮 Live Signal", "🌊 Distribution Diagnostics & Backtest"])

    # ------------------------------------------------------------------ #
    # TAB 1 — Live Signal
    # ------------------------------------------------------------------ #
    with tab1:
        st.markdown("""
        <div class="theory-note">
        Each ETF's trailing return distribution is averaged into a single "market barycenter"
        distribution for today. That barycenter is compared — via Wasserstein distance on
        quantile functions, not Euclidean distance on raw returns — to a library of historical
        regime centroids. The closest regime's historical next-day returns become the forecast below.
        </div>
        """, unsafe_allow_html=True)

        top_picks = data.get("top_picks", {})
        best_window = data.get("best_window", {})
        universes_meta = data.get("universes", {})

        if not top_picks:
            st.warning("No top-pick data available yet.")

        for universe, picks in top_picks.items():
            st.markdown(f'<div class="universe-heading">{universe.replace("_", " ").title()}</div>',
                        unsafe_allow_html=True)

            best = best_window.get(universe, {})
            if best:
                metrics = best.get("metrics", {})
                gate_passed = best.get("gate_passed", True)
                min_acc = best.get("min_directional_accuracy_required", 0.50)
                gate_icon = "✅" if gate_passed else "⚠️"
                gate_line = (
                    f"passed the ≥{min_acc:.0%} directional-accuracy gate"
                    if gate_passed else
                    f"<b>did not clear</b> the ≥{min_acc:.0%} directional-accuracy gate — "
                    f"this is the best of a bad lot, not a validated window"
                )
                st.markdown(f"""
                <div class="best-window-banner">
                    {gate_icon} Best regime-fit window: <b>{best.get('window', 'N/A')} days</b> &nbsp;|&nbsp; Forecast horizon: <b>{metrics.get('horizon_days', '?')} trading days</b>
                    <span style="opacity:0.7;">({gate_line}; ranked among eligible windows by a blend of correlation and net Sharpe)</span>
                    &nbsp;|&nbsp; Correlation: <b>{metrics.get('correlation', 0):.4f}</b>
                    &nbsp;|&nbsp; Directional accuracy: <b>{metrics.get('directional_accuracy', 0):.1%}</b>
                    &nbsp;|&nbsp; In-sample R²: <b>{metrics.get('avg_in_sample_r2', 0):.4f}</b>
                    <br/>
                    Net Sharpe (after {metrics.get('trading_cost_bps_assumed', 15)}bps trading cost):
                    <b>{metrics.get('sharpe', 0):.2f}</b>
                    <span style="opacity:0.7;">(gross, before costs: {metrics.get('sharpe_gross', 0):.2f})</span>
                </div>
                """, unsafe_allow_html=True)

            render_pick_cards(picks, key_prefix=f"picks_{universe}")

            uni_diag = universes_meta.get(universe, {}).get("diagnostics", {})
            if uni_diag:
                prev_regime = uni_diag.get("previous_regime")
                if prev_regime is None:
                    persistence_line = "No prior run state to compare against (cold start)."
                elif uni_diag.get("switched_from_previous"):
                    persistence_line = f"Regime <b>changed</b> since last run (was Regime {prev_regime})."
                else:
                    persistence_line = "Regime <b>unchanged</b> since last run (held via hysteresis)."

                st.markdown(f"""
                <div class="regime-note">
                    Assigned to <b>Regime {uni_diag.get('regime', '?')}</b>
                    ({uni_diag.get('regime_size', '?')} historical days, of {uni_diag.get('n_regimes_used', '?')} regimes fit)
                    &nbsp;|&nbsp; Distance to regime: <b>{uni_diag.get('regime_distance', 0):.3f}</b>
                    &nbsp;|&nbsp; Separation from 2nd-nearest regime: <b>{uni_diag.get('regime_separation', 0):.3f}</b>
                    <br/>{persistence_line}
                </div>
                """, unsafe_allow_html=True)

                with st.expander("Which historical days did today's distribution most resemble?"):
                    render_analog_table(uni_diag.get("nearest_analog_days", []), horizon=uni_diag.get("horizon_days"))

            st.markdown("<div style='margin: 0.5rem 0 1.5rem 0;'></div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------ #
    # TAB 2 — Distribution Diagnostics & Backtest
    # ------------------------------------------------------------------ #
    with tab2:
        backtest_results = data.get("backtest_results", {})
        window_picks = data.get("window_picks", {})
        diagnostics = data.get("diagnostics", {})

        if not backtest_results:
            st.warning("No backtest data available yet.")

        for universe, window_results in backtest_results.items():
            st.markdown(f'<div class="universe-heading">{universe.replace("_", " ").title()}</div>',
                        unsafe_allow_html=True)
            st.markdown(
                '<div class="universe-caption">Regime discovery, distributional-distance diagnostics, '
                'and performance across regime-fit window sizes</div>',
                unsafe_allow_html=True,
            )

            if not window_results:
                st.warning("No backtest results available for this universe.")
                continue

            df_results = pd.DataFrame([
                {
                    "Window": int(w),
                    "Horizon (d)": r.get("horizon_days", 1),
                    "Correlation": r.get("correlation", 0),
                    "Directional Accuracy": r.get("directional_accuracy", 0) * 100,
                    "Net Sharpe": r.get("sharpe", 0),
                    "Gross Sharpe": r.get("sharpe_gross", 0),
                    "In-sample R²": r.get("avg_in_sample_r2", 0),
                    "Avg Cost (bps/rebalance)": r.get("avg_cost_bps_per_rebalance", 0),
                    "Periods": r.get("n_predictions", 0),
                }
                for w, r in window_results.items()
            ]).sort_values("Window").reset_index(drop=True)

            best_idx = df_results["Correlation"].idxmax()
            best_row = df_results.loc[best_idx]
            best_window_val = str(int(best_row["Window"]))
            cost_bps = window_results.get(best_window_val, window_results.get(int(best_window_val), {})).get("trading_cost_bps_assumed", 15)

            best_window_switch_rate = (window_results.get(best_window_val, {})
                                        or window_results.get(int(best_window_val), {})).get("avg_regime_switch_rate")

            kpi_cols = st.columns(6)
            kpi_data = [
                ("Best Window", f"{int(best_row['Window'])}d"),
                ("Correlation", f"{best_row['Correlation']:.4f}"),
                ("Directional Acc.", f"{best_row['Directional Accuracy']:.1f}%"),
                ("Net Sharpe", f"{best_row['Net Sharpe']:.2f}"),
                ("In-sample R²", f"{best_row['In-sample R²']:.4f}"),
                ("Regime Switch Rate", f"{best_window_switch_rate*100:.1f}%" if best_window_switch_rate is not None else "N/A"),
            ]
            for col, (label, value) in zip(kpi_cols, kpi_data):
                with col:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-value">{value}</div>
                        <div class="kpi-label">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.caption(f"💸 Net Sharpe/returns above assume a {cost_bps}bps trading cost per position "
                       f"change, charged once per rebalance (every Horizon days), not per day. Sharpe is "
                       f"annualized by the number of non-overlapping horizon-length periods per year "
                       f"(252/horizon), and each 'Period' is a non-overlapping test point — so a longer "
                       f"horizon means fewer, but statistically independent, observations. Regime Switch "
                       f"Rate is the share of test periods the regime assignment actually changed; see README.")

            st.markdown("<div style='margin-top: 0.9rem;'></div>", unsafe_allow_html=True)

            st.dataframe(
                df_results.style.apply(
                    lambda x: ["background-color: #ccfbf1" if x.name == best_idx else "" for _ in x],
                    axis=1,
                ).format({
                    "Correlation": "{:.4f}",
                    "Directional Accuracy": "{:.1f}%",
                    "Net Sharpe": "{:.2f}",
                    "Gross Sharpe": "{:.2f}",
                    "In-sample R²": "{:.4f}",
                    "Avg Cost (bps/rebalance)": "{:.2f}",
                    "Periods": "{:,.0f}",
                }),
                use_container_width=True,
                hide_index=True,
                column_config={"Window": "Window (days)"},
            )

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("###### Performance across regime-fit windows")
                render_window_comparison(df_results, best_idx, key=f"{universe}")
            with col_b:
                st.markdown("###### Distance to nearest regime (best window, test period)")
                distance_series = window_results.get(int(best_window_val), {}).get("distance_series", [])
                if not distance_series:
                    distance_series = window_results.get(best_window_val, {}).get("distance_series", [])
                render_distance_chart(distance_series, key=f"{universe}")

            best_window_result = window_results.get(best_window_val) or window_results.get(int(best_window_val)) or {}
            hp_search = best_window_result.get("hyperparam_search", [])
            if hp_search:
                st.markdown("###### Hyperparameter search (best window)")
                st.caption(
                    f"local_window / n_regimes / horizon were searched over {len(hp_search)} combinations for the "
                    f"{best_window_val}d window; the winner is highlighted. "
                    f"⚠️ Testing more combinations raises the chance the 'best' one just got lucky — "
                    f"if the winner isn't clearly ahead of the rest, treat its edge as noise."
                )
                hp_df = pd.DataFrame(hp_search)
                winner_mask = (
                    (hp_df["local_window"] == best_window_result.get("local_window"))
                    & (hp_df["n_regimes"] == best_window_result.get("n_regimes"))
                )
                winner_idx = hp_df[winner_mask].index[0] if winner_mask.any() else None
                st.dataframe(
                    hp_df.style.apply(
                        lambda x: ["background-color: #ccfbf1" if x.name == winner_idx else "" for _ in x],
                        axis=1,
                    ).format({"correlation": "{:.4f}", "sharpe": "{:.2f}", "n_predictions": "{:,.0f}"}),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "local_window": "Local window (days)",
                        "horizon_days": "Horizon (days)",
                        "n_regimes": "Regimes",
                        "correlation": "Correlation",
                        "sharpe": "Sharpe",
                        "n_predictions": "Periods",
                    },
                )

            uni_diag = diagnostics.get(universe, {})
            best_diag = uni_diag.get(best_window_val) or uni_diag.get(int(best_window_val)) or {}

            col_c, col_d = st.columns(2)
            with col_c:
                st.markdown("###### Regime map (best window)")
                render_regime_map(best_diag.get("regime_map"), key=f"{universe}")
            with col_d:
                st.markdown("###### Today's distribution vs. nearest regime")
                render_quantile_curve(best_diag.get("quantile_curves"), key=f"{universe}")

            st.markdown("###### ETF picks by window")
            universe_window_picks = window_picks.get(universe, {})
            if not universe_window_picks:
                st.info("No per-window ETF picks in this results file.")
            else:
                available_windows = sorted(universe_window_picks.keys(), key=lambda w: int(w))
                window_tabs = st.tabs([f"{w}d" for w in available_windows])
                for wtab, w in zip(window_tabs, available_windows):
                    with wtab:
                        render_pick_cards(universe_window_picks[w], key_prefix=f"wpicks_{universe}_{w}")

            st.markdown("<hr style='margin: 1.75rem 0; border-color: #e2e8f0;'>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
