"""GitHub Repository Metadata dashboard (Streamlit).

Reads the 16 CSV files produced by the PySpark notebook from the ./data folder.
Run with:  streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="GitHub Repository Metadata", page_icon="📊", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
FILES = [
    "overview", "stars_percentiles", "stars_log_hist", "language_summary", "language_bytes",
    "created_year_cohorts", "recency_buckets", "size_buckets", "collab_buckets", "license_summary",
    "license_flag", "archived_vs_active", "topic_summary", "correlation_log1p", "ml_results",
    "ml_feature_importance_gbt",
]
ACCENT, ACCENT2 = "#2f81f7", "#8250df"


@st.cache_data(show_spinner="Loading data...")
def load_all() -> dict:
    return {f: pd.read_csv(DATA_DIR / f"{f}.csv") for f in FILES}


def fmt(n) -> str:
    if pd.isna(n):
        return ""
    a = abs(n)
    if a >= 1e9:
        return f"{n / 1e9:.2f}B"
    if a >= 1e6:
        return f"{n / 1e6:.2f}M"
    if a >= 1e3:
        return f"{n / 1e3:.1f}K"
    return f"{n:,.0f}" if float(n).is_integer() else f"{n:.3f}"


def pct(x) -> str:
    return f"{x * 100:.1f}%"


def hbar(df, x, y, color=ACCENT, x_title="", height=480):
    fig = px.bar(df, x=x, y=y, orientation="h", color_discrete_sequence=[color])
    fig.update_yaxes(autorange="reversed", title="")
    fig.update_xaxes(title=x_title)
    fig.update_layout(height=height, margin=dict(l=0, r=10, t=10, b=0))
    return fig


def vbar(df, x, y, color=ACCENT, y_title="", x_title="", height=330, log_y=False):
    fig = px.bar(df, x=x, y=y, color_discrete_sequence=[color], log_y=log_y)
    fig.update_xaxes(title=x_title, type="category")
    fig.update_yaxes(title=y_title)
    fig.update_layout(height=height, margin=dict(l=0, r=10, t=10, b=0))
    return fig


def dual(df, x, bar_col, line_col, bar_name, line_name, line_pct=False, height=330):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=df[x].astype(str), y=df[bar_col], name=bar_name, marker_color=ACCENT)
    fig.add_scatter(x=df[x].astype(str), y=df[line_col] * (100 if line_pct else 1),
                    name=line_name, mode="lines+markers", line=dict(color=ACCENT2),
                    secondary_y=True)
    fig.update_yaxes(title_text=bar_name, secondary_y=False)
    fig.update_yaxes(title_text=line_name + (" (%)" if line_pct else ""), secondary_y=True,
                     showgrid=False)
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", y=1.12))
    return fig


# ---------------------------------------------------------------- load
if not DATA_DIR.is_dir() or any(not (DATA_DIR / f"{f}.csv").is_file() for f in FILES):
    missing = [f for f in FILES if not (DATA_DIR / f"{f}.csv").is_file()]
    st.error(
        "Could not find all dashboard CSVs in the `data/` folder next to `app.py`.\n\n"
        f"Missing: {', '.join(missing) if missing else 'data folder'}.\n\n"
        "Copy the 16 CSV files from `/kaggle/working/outputs` into `data/`."
    )
    st.stop()

D = load_all()

# ---------------------------------------------------------------- header
st.title("GitHub Repository Metadata")
st.caption("4.2 million public repositories with 5+ stars: quality, trends and a popularity model")

tabs = st.tabs(["Overview", "Stars", "Languages", "Time", "Repo health", "Topics",
                "Correlations", "Model", "Data explorer"])

# ---------------------------------------------------------------- overview
with tabs[0]:
    o = D["overview"].iloc[0]
    kpis = [
        (fmt(o.repos), "Repositories"),
        (fmt(o.total_stars), "Total stars"),
        (pct(o.share_licensed), "Have a license"),
        (pct(o.share_with_topics), "Have topics"),
        (pct(o.share_archived), "Archived"),
        (pct(o.share_active_365d_of_valid_dates), "Pushed in last year"),
        (f"{o.median_age_days / 365.25:.1f} yrs", "Median repo age"),
    ]
    cols = st.columns(len(kpis))
    for c, (v, l) in zip(cols, kpis):
        c.metric(l, v)
    st.info(
        f"The top **1%** of repositories ({fmt(o.repos_at_or_above_p99)} repos with "
        f">= {int(o.p99_stars_threshold):,} stars) hold "
        f"**{pct(o.share_of_all_stars_at_or_above_p99)}** of all stars."
    )
    c1, c2 = st.columns(2)
    c1.metric("Median days since last push", f"{o.median_days_since_push:,.0f}")
    c2.metric("Share with description", pct(o.share_with_description))

# ---------------------------------------------------------------- stars
with tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Repositories by stars (log-scale bins)")
        st.caption("Each bar is a quarter of a power of ten. A few repos have enormous star "
                   "counts while most have few.")
        h = D["stars_log_hist"].dropna(subset=["log10_stars_lower"]).copy()
        h["stars"] = (10 ** h["log10_stars_lower"]).round(0).astype(int).map(
            lambda v: f"{v / 1000:.1f}K" if v >= 1000 else str(v))
        fig = vbar(h, "stars", "count", y_title="repositories", x_title="stars (lower edge of bin)")
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("Star percentiles")
        st.caption("The typical repo has about a dozen stars.")
        p = D["stars_percentiles"].copy()
        p["pctl"] = p["percentile"].map(lambda q: f"p{q * 100:.1f}" if q >= 0.99 else f"p{q * 100:.0f}")
        st.plotly_chart(vbar(p, "pctl", "stars", color=ACCENT2, y_title="stars (log)", log_y=True),
                        width="stretch")

# ---------------------------------------------------------------- languages
with tabs[2]:
    n_lang = st.slider("Number of languages to show", 5, 30, 15)
    L = D["language_summary"].head(n_lang)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Primary languages by repository count")
        st.caption('"Unknown" = no primary language reported.')
        st.plotly_chart(hbar(L, "repos", "primary_language_clean", x_title="repositories"),
                        width="stretch")
    with c2:
        st.subheader("Language usage across all repo languages")
        st.caption("Total code volume per language, in billions of bytes.")
        LB = D["language_bytes"].head(n_lang).copy()
        LB["billion_bytes"] = LB["total_bytes"] / 1e9
        st.plotly_chart(hbar(LB, "billion_bytes", "language", color=ACCENT2,
                             x_title="billion bytes"), width="stretch")
    st.subheader("Share of repos active in the last year, by language")
    LA = L[L["primary_language_clean"] != "Unknown"].copy()
    LA["active_pct"] = LA["share_active_365d"] * 100
    LA = LA.sort_values("active_pct", ascending=False)
    st.plotly_chart(hbar(LA, "active_pct", "primary_language_clean", x_title="% active"),
                    width="stretch")
    with st.expander("Language table"):
        st.dataframe(D["language_summary"], width="stretch", hide_index=True)

# ---------------------------------------------------------------- time
with tabs[3]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Repositories by creation year")
        st.caption("Line shows the share still active in the last 365 days.")
        C = D["created_year_cohorts"].dropna(subset=["created_year"])
        st.plotly_chart(dual(C, "created_year", "repos", "share_active_365d",
                             "Repos", "Active last 365d", line_pct=True), width="stretch")
        st.caption("The 100% for 2025 is true by construction: the snapshot is September 2025.")
    with c2:
        st.subheader("Median stars by recency of last push")
        R = D["recency_buckets"].sort_values("bucket_order")
        st.plotly_chart(vbar(R, "bucket", "median_stars", color=ACCENT2,
                             y_title="median stars", x_title="time since last push"),
                        width="stretch")
    st.subheader("Median stars and forks by creation year")
    C2 = D["created_year_cohorts"].melt(id_vars="created_year",
                                       value_vars=["median_stars", "median_forks"])
    fig = px.line(C2, x="created_year", y="value", color="variable", markers=True,
                  color_discrete_sequence=[ACCENT, ACCENT2])
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), legend_title="")
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- health
with tabs[4]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Licenses (top 10)")
        Li = D["license_summary"].head(10).copy()
        Li["license"] = Li["license_clean"].str.replace(" License", "", regex=False)
        st.plotly_chart(hbar(Li, "repos", "license", x_title="repositories", height=380),
                        width="stretch")
    with c2:
        st.subheader("Licensed vs unlicensed")
        lf = D["license_flag"].copy()
        lf["has_license"] = lf["has_license"].map(
            lambda v: "Licensed" if str(v).lower() == "true" else "No license")
        st.dataframe(lf, width="stretch", hide_index=True)
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Team size (assignable users)")
        st.caption("Bars: repos. Line: share with pull requests.")
        st.plotly_chart(dual(D["collab_buckets"].sort_values("bucket_order"), "bucket", "repos",
                             "share_with_pull_requests", "Repos", "Has pull requests",
                             line_pct=True), width="stretch")
    with c4:
        st.subheader("Repository size")
        st.caption("Bars: repos. Line: median commits.")
        st.plotly_chart(dual(D["size_buckets"].sort_values("bucket_order"), "bucket", "repos",
                             "median_commits", "Repos", "Median commits"),
                        width="stretch")
    st.subheader("Archived vs active (median values)")
    av = D["archived_vs_active"].copy()
    av["status"] = av["isArchived"].map(lambda v: "Archived" if str(v).lower() == "true" else "Active")
    av = av.drop(columns="isArchived").set_index("status").T
    av.index = [i.replace("median_", "median ").replace("_", " ") for i in av.index]
    st.dataframe(av.map(fmt), width="stretch")

# ---------------------------------------------------------------- topics
with tabs[5]:
    n_top = st.slider("Number of topics to show", 10, 50, 20)
    st.subheader(f"Top {n_top} topics by repository count")
    st.caption("Only about 30% of repos have any topic.")
    T = D["topic_summary"].head(n_top)
    st.plotly_chart(hbar(T, "repos", "topic", color=ACCENT2, x_title="repositories",
                         height=max(420, n_top * 22)), width="stretch")

# ---------------------------------------------------------------- correlations
with tabs[6]:
    st.subheader("Correlations (Pearson on log1p)")
    st.caption("Counts are heavily skewed, so values are log-transformed first. "
               "These are correlations, not causes.")
    cr = D["correlation_log1p"].set_index("variable")
    fig = px.imshow(cr, text_auto=".2f", color_continuous_scale="Blues", zmin=-0.2, zmax=1,
                    aspect="auto")
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- model
with tabs[7]:
    st.subheader("Predicting popular repositories (stars >= 100)")
    st.warning(
        "Set **A** excludes forks and watchers. Set **B** includes them, but they are almost a "
        "proxy for stars, so B's higher scores are inflated. Treat **A** as the less circular "
        "result (issues and pull requests are still partly consequences of popularity). "
        "Random guessing would score a PR-AUC of about 0.098."
    )
    ml = D["ml_results"].copy()
    ml["set"] = ml["feature_set"].str[0]
    ml["label"] = ml["set"] + " / " + ml["model"].replace({"LogisticRegression": "LogReg"})

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**PR-AUC by model**")
        fig = px.bar(ml, x="label", y="pr_auc", color="set",
                     color_discrete_map={"A": ACCENT, "B": ACCENT2})
        fig.add_hline(y=0.098, line_dash="dash", annotation_text="random baseline (0.098)")
        fig.update_yaxes(range=[0, 1], title="PR-AUC")
        fig.update_xaxes(title="")
        fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("**Test-set metrics**")
        show = ml[["feature_set", "model", "precision", "recall", "f1", "pr_auc", "roc_auc",
                   "fit_seconds"]]
        st.dataframe(show, width="stretch", hide_index=True,
                     column_config={c: st.column_config.NumberColumn(format="%.3f")
                                    for c in ["precision", "recall", "f1", "pr_auc", "roc_auc"]})

    st.markdown("**Confusion matrix**")
    opts = ml["feature_set"] + " | " + ml["model"]
    pick = st.selectbox("Model", opts.tolist(), index=1)
    r = ml[opts == pick].iloc[0]
    cm = pd.DataFrame([[r.TN, r.FP], [r.FN, r.TP]],
                      index=["Actual: not popular", "Actual: popular"],
                      columns=["Predicted: not popular", "Predicted: popular"])
    fig = px.imshow(cm, text_auto=",", color_continuous_scale="Blues", aspect="auto")
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch")

    imp = D["ml_feature_importance_gbt"].copy()
    imp["feature"] = imp["feature"].str.replace(r"^log_", "", regex=True)
    c3, c4 = st.columns(2)
    for col, key, name, color in [(c3, "A", "set A (no forks/watchers)", ACCENT),
                                  (c4, "B", "set B (with forks/watchers)", ACCENT2)]:
        with col:
            st.markdown(f"**GBT feature importance: {name}**")
            sub = imp[imp["feature_set"].str.startswith(key)].nlargest(12, "importance")
            st.plotly_chart(hbar(sub, "importance", "feature", color=color, height=420),
                            width="stretch")

# ---------------------------------------------------------------- explorer
with tabs[8]:
    st.subheader("Data explorer")
    name = st.selectbox("File", FILES, format_func=lambda f: f + ".csv")
    df = D[name]
    st.caption(f"{len(df):,} rows x {df.shape[1]} columns")
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button("Download CSV", df.to_csv(index=False).encode(), f"{name}.csv", "text/csv")

st.divider()
st.caption("Built from a PySpark pipeline run on the GitHub Repository Metadata dataset "
           "(pelmers, Kaggle). Data files are in the data/ folder.")
