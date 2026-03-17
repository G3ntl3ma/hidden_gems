"""Interactive analysis dashboard: EDA, sentiment, topics, clustering,
and hidden-gem scoring."""

from __future__ import annotations

import sys
from pathlib import Path
import json
import time

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from models.eda import (
    correlation_matrix,
    missing_data_audit,
    numeric_summary,
    ownership_tiers,
    platform_distribution,
    review_score_distribution,
)
from models.sentiment import (
    add_sentiment_to_reviews,
    aggregate_sentiment_per_game,
)
from models.topic_model import build_review_topics_pipeline
from models.clustering import (
    build_feature_matrix,
    reduce_pca,
    reduce_umap,
    run_dbscan,
    run_hierarchical,
    run_kmeans,
)
from models.hidden_gems import compute_hidden_gem_score, detect_anomalies
from models.analysis_artifacts import (
    DEFAULT_TOPIC_COUNT,
    DEFAULT_TOPIC_METHOD,
    load_or_precompute,
)

# region agent log
_DEBUG_LOG_PATH = Path("/home/user/Programming/hidden_gems/.cursor/debug-7882d0.log")


def _agent_log(*, run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    try:
        payload = {
            "sessionId": "7882d0",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        _DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _isfinite_series(s: pd.Series) -> bool:
    try:
        arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        return bool(np.isfinite(arr).all())
    except Exception:
        return False

# endregion agent log


st.set_page_config(page_title="Game Analysis", layout="wide")
st.title("Game Analysis Pipeline")

# ── data loading ──────────────────────────────────────────────────────────

GAMES_CSV = Path("steam_games_clean.csv")
REVIEWS_CSV = Path("steam_reviews_clean.csv")

# region agent log
_agent_log(
    run_id="pre-fix",
    hypothesis_id="A",
    location="view/pages/2_Analysis.py:page_entry",
    message="Analysis page entry",
    data={
        "server.baseUrlPath": st.get_option("server.baseUrlPath"),
        "server.enableCORS": st.get_option("server.enableCORS"),
        "server.enableXsrfProtection": st.get_option("server.enableXsrfProtection"),
    },
)
# endregion agent log


@st.cache_data(show_spinner="Loading games …")
def load_games() -> pd.DataFrame:
    return pd.read_csv(GAMES_CSV)


@st.cache_data(show_spinner="Loading reviews …")
def load_reviews() -> pd.DataFrame:
    return pd.read_csv(REVIEWS_CSV)


def _file_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


@st.cache_data(show_spinner="Loading analysis artifacts …")
def _load_analysis_artifacts(
    games_sig: tuple[int, int],
    reviews_sig: tuple[int, int],
) -> tuple[dict, bool, str]:
    del games_sig, reviews_sig
    artifacts, loaded_from_cache, cache_dir = load_or_precompute(GAMES_CSV, REVIEWS_CSV)
    return artifacts, loaded_from_cache, str(cache_dir)


if not GAMES_CSV.exists() or not REVIEWS_CSV.exists():
    st.warning(
        "Cleaned CSV data files not found. Run `python scripts/clean_collected_data.py` "
        "and ensure `steam_games_clean.csv` and `steam_reviews_clean.csv` exist "
        "in the project root."
    )
    # region agent log
    _agent_log(
        run_id="pre-fix",
        hypothesis_id="C",
        location="view/pages/2_Analysis.py:missing_csv",
        message="Required cleaned CSVs missing; stopping",
        data={"games_csv_exists": GAMES_CSV.exists(), "reviews_csv_exists": REVIEWS_CSV.exists()},
    )
    # endregion agent log
    st.stop()

games_df = load_games()
reviews_df = load_reviews()

# region agent log
_agent_log(
    run_id="pre-fix",
    hypothesis_id="C",
    location="view/pages/2_Analysis.py:loaded_csv",
    message="Loaded cleaned CSVs",
    data={
        "games_rows": int(len(games_df)),
        "games_cols": int(len(games_df.columns)),
        "reviews_rows": int(len(reviews_df)),
        "reviews_cols": int(len(reviews_df.columns)),
    },
)
# endregion agent log

artifacts: dict | None = None
artifacts_loaded_from_cache = False
artifacts_cache_dir = ""
artifacts_error: str | None = None
try:
    artifacts, artifacts_loaded_from_cache, artifacts_cache_dir = _load_analysis_artifacts(
        _file_signature(GAMES_CSV),
        _file_signature(REVIEWS_CSV),
    )
except Exception as exc:
    artifacts_error = str(exc)

# region agent log
_agent_log(
    run_id="pre-fix",
    hypothesis_id="C",
    location="view/pages/2_Analysis.py:artifacts_status",
    message="Artifact load status",
    data={
        "artifacts_ok": artifacts_error is None and artifacts is not None,
        "loaded_from_cache": bool(artifacts_loaded_from_cache),
        "cache_dir": str(artifacts_cache_dir),
        "error": artifacts_error,
        "keys": sorted(list(artifacts.keys())) if isinstance(artifacts, dict) else None,
    },
)
# endregion agent log

st.sidebar.metric("Games", len(games_df))
st.sidebar.metric("Reviews", len(reviews_df))
if artifacts_error:
    st.sidebar.warning("Analysis artifacts unavailable; using live compute.")
else:
    source = "precomputed cache" if artifacts_loaded_from_cache else "freshly recomputed"
    st.sidebar.caption(f"Analysis artifacts: {source}")
    st.sidebar.caption(f"Artifact directory: {artifacts_cache_dir}")


# ── shared cached helpers (used across multiple tabs) ─────────────────────

@st.cache_data(show_spinner="Running sentiment analysis …")
def _sentiment_reviews(df: pd.DataFrame) -> pd.DataFrame:
    return add_sentiment_to_reviews(df)


@st.cache_data(show_spinner="Aggregating sentiment per game …")
def _sentiment_per_game(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate_sentiment_per_game(df)


@st.cache_data(show_spinner="Fitting topic model …")
def _run_topics(df: pd.DataFrame, n: int, method: str) -> dict:
    return build_review_topics_pipeline(
        df, n_topics=n, method=method, max_features=5000,
    )


@st.cache_data(show_spinner="Building feature matrix …")
def _build_features(
    _games: pd.DataFrame,
    _sent: pd.DataFrame | None,
    _topics: pd.DataFrame | None,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    return build_feature_matrix(_games, _sent, _topics)


@st.cache_data(show_spinner="Computing hidden-gem scores …")
def _gem_scores(
    _games: pd.DataFrame,
    _sent: pd.DataFrame | None,
) -> pd.DataFrame:
    df = _games.copy()
    if _sent is not None:
        df = df.merge(_sent, left_on="id", right_on="gameId", how="left")
    return compute_hidden_gem_score(df)


def _get_english_reviews(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["language"].fillna("").str.lower() == "english"]


def _split_multi_value_cell(value: object) -> set[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    return {part.strip() for part in str(value).split(";") if part.strip()}


def _multi_value_options(series: pd.Series) -> list[str]:
    values: set[str] = set()
    for value in series:
        values.update(_split_multi_value_cell(value))
    return sorted(values)


# ── tabs ──────────────────────────────────────────────────────────────────

tab_eda, tab_sent, tab_topics, tab_clusters, tab_gems = st.tabs(
    ["EDA Overview", "Sentiment", "Topics", "Clusters", "Hidden Gems"],
)

# region agent log
_agent_log(
    run_id="pre-fix",
    hypothesis_id="D",
    location="view/pages/2_Analysis.py:tabs_created",
    message="Tabs created; about to render tab contents",
    data={},
)
# endregion agent log

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 – Exploratory Data Analysis
# ═══════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.header("Exploratory Data Analysis")

    st.subheader("Numeric summary — Games")
    # region agent log
    _t0 = time.time()
    _agent_log(
        run_id="pre-fix",
        hypothesis_id="D",
        location="view/pages/2_Analysis.py:eda_numeric_summary_start",
        message="Starting numeric_summary(games_df)",
        data={"games_rows": int(len(games_df)), "games_cols": int(len(games_df.columns))},
    )
    # endregion agent log
    _num_summary = numeric_summary(games_df)
    # region agent log
    _agent_log(
        run_id="pre-fix",
        hypothesis_id="D",
        location="view/pages/2_Analysis.py:eda_numeric_summary_end",
        message="Finished numeric_summary(games_df)",
        data={"elapsed_s": round(time.time() - _t0, 3), "rows": int(len(_num_summary))},
    )
    # endregion agent log
    st.dataframe(_num_summary, use_container_width=True)

    st.subheader("Missing data audit")
    audit = missing_data_audit(games_df)
    st.dataframe(
        audit[audit["missing_count"] > 0],
        use_container_width=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Review score distribution")
        rsd = review_score_distribution(games_df)
        if not rsd.empty:
            fig, ax = plt.subplots()
            rsd.plot.barh(ax=ax)
            ax.set_xlabel("Count")
            ax.set_ylabel("")
            st.pyplot(fig)
            plt.close(fig)

    with col2:
        st.subheader("Platform support")
        pd_dist = platform_distribution(games_df)
        if not pd_dist.empty:
            fig, ax = plt.subplots()
            pd_dist["count"].plot.bar(ax=ax)
            ax.set_ylabel("Games")
            st.pyplot(fig)
            plt.close(fig)

    st.subheader("Ownership tiers")
    tiers = ownership_tiers(games_df)
    if not tiers.empty:
        st.bar_chart(tiers)

    st.subheader("Correlation matrix")
    # region agent log
    _t1 = time.time()
    _agent_log(
        run_id="pre-fix",
        hypothesis_id="D",
        location="view/pages/2_Analysis.py:eda_corr_start",
        message="Starting correlation_matrix(games_df)",
        data={},
    )
    # endregion agent log
    corr = correlation_matrix(games_df)
    # region agent log
    _agent_log(
        run_id="pre-fix",
        hypothesis_id="D",
        location="view/pages/2_Analysis.py:eda_corr_end",
        message="Finished correlation_matrix(games_df)",
        data={"elapsed_s": round(time.time() - _t1, 3), "empty": bool(getattr(corr, "empty", True))},
    )
    # endregion agent log
    if not corr.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 – Sentiment Analysis
# ═══════════════════════════════════════════════════════════════════════════
with tab_sent:
    st.header("Sentiment Analysis (VADER)")
    # region agent log
    _agent_log(
        run_id="pre-fix",
        hypothesis_id="E",
        location="view/pages/2_Analysis.py:tab_sent_entry",
        message="Entered Sentiment tab block",
        data={},
    )
    # endregion agent log

    if artifacts is not None:
        reviews_sent = artifacts["reviews_sentiment"]
        game_sentiment = artifacts["sentiment_per_game"]
    else:
        reviews_sent = _sentiment_reviews(reviews_df)
        game_sentiment = _sentiment_per_game(reviews_sent)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Compound sentiment distribution")
        fig, ax = plt.subplots()
        ax.hist(reviews_sent["sentiment_compound"], bins=40, edgecolor="black")
        ax.set_xlabel("Compound score")
        ax.set_ylabel("Reviews")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Sentiment by vote direction")
        if "votedUp" in reviews_sent.columns:
            fig, ax = plt.subplots()
            for voted, grp in reviews_sent.groupby("votedUp"):
                label = "Positive" if voted else "Negative"
                ax.hist(
                    grp["sentiment_compound"],
                    bins=30,
                    alpha=0.6,
                    label=label,
                    edgecolor="black",
                )
            ax.legend()
            ax.set_xlabel("Compound score")
            ax.set_ylabel("Reviews")
            st.pyplot(fig)
            plt.close(fig)

    st.subheader("Per-game sentiment")
    display_cols = [
        c for c in [
            "gameId", "sentiment_mean", "sentiment_median",
            "sentiment_std", "sentiment_pos_ratio",
            "avg_review_length", "review_count",
        ] if c in game_sentiment.columns
    ]
    st.dataframe(
        game_sentiment[display_cols].sort_values(
            "sentiment_mean", ascending=False,
        ),
        use_container_width=True,
    )

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 – Topic Modeling
# ═══════════════════════════════════════════════════════════════════════════
with tab_topics:
    st.header("Topic Modeling")
    # region agent log
    _agent_log(
        run_id="pre-fix",
        hypothesis_id="E",
        location="view/pages/2_Analysis.py:tab_topics_entry",
        message="Entered Topics tab block",
        data={},
    )
    # endregion agent log

    n_topics = st.slider("Number of topics", 3, 20, 8, key="n_topics")
    topic_method = st.selectbox("Method", ["lda", "nmf"], key="topic_method")

    topic_result = None
    use_precomputed_topics = (
        artifacts is not None
        and n_topics == DEFAULT_TOPIC_COUNT
        and topic_method == DEFAULT_TOPIC_METHOD
        and artifacts.get("topic_result", {}).get("game_topics") is not None
    )
    if use_precomputed_topics:
        topic_result = artifacts["topic_result"]
        st.caption("Using precomputed topic model output.")
    else:
        english_reviews = _get_english_reviews(reviews_df)
        if len(english_reviews) < 10:
            st.info("Not enough English reviews for topic modeling (need >= 10).")
        else:
            topic_result = _run_topics(english_reviews, n_topics, topic_method)

    if topic_result is not None:
        st.subheader("Discovered topics")
        for tid, words in topic_result["top_words"].items():
            st.markdown(f"**Topic {tid}:** {', '.join(words)}")

        st.subheader("Per-game topic averages")
        st.dataframe(topic_result["game_topics"], use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 – Clustering
# ═══════════════════════════════════════════════════════════════════════════
with tab_clusters:
    st.header("Unsupervised Clustering")
    # region agent log
    _agent_log(
        run_id="pre-fix",
        hypothesis_id="E",
        location="view/pages/2_Analysis.py:tab_clusters_entry",
        message="Entered Clusters tab block",
        data={},
    )
    # endregion agent log

    if len(games_df) < 4:
        st.warning("Need at least 4 games to run meaningful clustering.")
        st.stop()

    if artifacts is not None:
        features = artifacts["features"]
        merged_df = features["merged_df"]
        X = features["X"]
        feature_names = features["feature_names"]
        st.caption("Using precomputed feature matrix.")
    else:
        sent_agg = _sentiment_per_game(reviews_df)
        english_for_topics = _get_english_reviews(reviews_df)
        topic_agg = None
        if len(english_for_topics) >= 10:
            tr = _run_topics(english_for_topics, 8, "lda")
            topic_agg = tr["game_topics"]
        merged_df, X, feature_names = _build_features(games_df, sent_agg, topic_agg)

    st.write(f"Feature matrix: **{X.shape[0]}** games x **{X.shape[1]}** features")
    st.caption(f"Features: {', '.join(feature_names)}")

    # Clustering on the full dataset can be prohibitively slow in Streamlit
    # because all tab blocks execute on page load. Default to a manageable sample.
    max_cluster_rows = st.slider(
        "Max games to cluster (sampled)",
        min_value=500,
        max_value=20000,
        value=5000,
        step=500,
        key="cluster_max_rows",
        help="Clustering cost grows quickly with dataset size; sampling keeps the dashboard responsive.",
    )
    if X.shape[0] > max_cluster_rows:
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], size=max_cluster_rows, replace=False)
        Xc = X[idx]
        merged_df_c = merged_df.iloc[idx].reset_index(drop=True)
        st.info(f"Clustering is running on a random sample of **{max_cluster_rows}** games.")
        # region agent log
        _agent_log(
            run_id="post-fix",
            hypothesis_id="G",
            location="view/pages/2_Analysis.py:clusters_sampling",
            message="Downsampled for clustering",
            data={"full_rows": int(X.shape[0]), "cluster_rows": int(Xc.shape[0]), "cluster_cols": int(Xc.shape[1])},
        )
        # endregion agent log
    else:
        Xc = X
        merged_df_c = merged_df.reset_index(drop=True)
        # region agent log
        _agent_log(
            run_id="post-fix",
            hypothesis_id="G",
            location="view/pages/2_Analysis.py:clusters_sampling",
            message="Using full matrix for clustering",
            data={"full_rows": int(X.shape[0]), "cluster_rows": int(Xc.shape[0]), "cluster_cols": int(Xc.shape[1])},
        )
        # endregion agent log

    # ── K-Means ───────────────────────────────────────────────────────────
    st.subheader("K-Means")
    max_k = min(10, Xc.shape[0] - 1)
    if max_k >= 2:
        # region agent log
        _t_km = time.time()
        _agent_log(
            run_id="pre-fix",
            hypothesis_id="F",
            location="view/pages/2_Analysis.py:clusters_kmeans_start",
            message="Starting run_kmeans",
            data={"n_samples": int(Xc.shape[0]), "n_features": int(Xc.shape[1]), "max_k": int(max_k)},
        )
        # endregion agent log
        km_results = run_kmeans(Xc, k_range=range(2, max_k + 1))
        # region agent log
        _agent_log(
            run_id="pre-fix",
            hypothesis_id="F",
            location="view/pages/2_Analysis.py:clusters_kmeans_end",
            message="Finished run_kmeans",
            data={"elapsed_s": round(time.time() - _t_km, 3)},
        )
        # endregion agent log

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.plot(km_results["k_range"], km_results["inertias"], "o-")
            ax.set_xlabel("k")
            ax.set_ylabel("Inertia")
            ax.set_title("Elbow plot")
            st.pyplot(fig)
            plt.close(fig)
        with col2:
            fig, ax = plt.subplots()
            ax.plot(km_results["k_range"], km_results["silhouettes"], "o-")
            ax.set_xlabel("k")
            ax.set_ylabel("Silhouette")
            ax.set_title("Silhouette score")
            st.pyplot(fig)
            plt.close(fig)

        best_k = km_results["k_range"][
            int(np.argmax(km_results["silhouettes"]))
        ]
        st.info(f"Best k by silhouette: **{best_k}**")
        chosen_labels = km_results["models"][best_k]["labels"]
    else:
        st.info("Not enough games for K-Means analysis.")
        chosen_labels = np.zeros(Xc.shape[0], dtype=int)

    # ── DBSCAN ────────────────────────────────────────────────────────────
    st.subheader("DBSCAN")
    eps = st.slider("eps", 0.1, 5.0, 1.5, 0.1, key="dbscan_eps")
    min_samp = st.slider("min_samples", 2, 10, 2, key="dbscan_min")
    # region agent log
    _t_db = time.time()
    _agent_log(
        run_id="pre-fix",
        hypothesis_id="F",
        location="view/pages/2_Analysis.py:clusters_dbscan_start",
        message="Starting run_dbscan",
        data={"eps": float(eps), "min_samples": int(min_samp)},
    )
    # endregion agent log
    db_result = run_dbscan(Xc, eps=eps, min_samples=min_samp)
    # region agent log
    _agent_log(
        run_id="pre-fix",
        hypothesis_id="F",
        location="view/pages/2_Analysis.py:clusters_dbscan_end",
        message="Finished run_dbscan",
        data={"elapsed_s": round(time.time() - _t_db, 3)},
    )
    # endregion agent log
    st.write(
        f"Clusters: **{db_result['n_clusters']}** | "
        f"Noise points: **{db_result['n_noise']}** | "
        f"Silhouette: **{db_result['silhouette']:.3f}**"
    )

    # ── Hierarchical ──────────────────────────────────────────────────────
    st.subheader("Hierarchical (Agglomerative)")
    n_clust = st.slider(
        "n_clusters", 2, min(10, Xc.shape[0] - 1), min(3, Xc.shape[0] - 1),
        key="hier_k",
    )
    # region agent log
    _t_hc = time.time()
    _agent_log(
        run_id="pre-fix",
        hypothesis_id="F",
        location="view/pages/2_Analysis.py:clusters_hier_start",
        message="Starting run_hierarchical",
        data={"n_clusters": int(n_clust)},
    )
    # endregion agent log
    hc_result = run_hierarchical(Xc, n_clusters=n_clust)
    # region agent log
    _agent_log(
        run_id="pre-fix",
        hypothesis_id="F",
        location="view/pages/2_Analysis.py:clusters_hier_end",
        message="Finished run_hierarchical",
        data={"elapsed_s": round(time.time() - _t_hc, 3)},
    )
    # endregion agent log
    st.write(f"Silhouette: **{hc_result['silhouette']:.3f}**")

    # ── Visualisation ─────────────────────────────────────────────────────
    st.subheader("2-D projection")
    viz_method = st.selectbox("Reduction method", ["PCA", "UMAP"], key="viz_method")

    if viz_method == "PCA":
        # region agent log
        _t_red = time.time()
        _agent_log(
            run_id="pre-fix",
            hypothesis_id="F",
            location="view/pages/2_Analysis.py:clusters_reduce_pca_start",
            message="Starting reduce_pca",
            data={},
        )
        # endregion agent log
        X_2d, _ = reduce_pca(Xc)
        # region agent log
        _agent_log(
            run_id="pre-fix",
            hypothesis_id="F",
            location="view/pages/2_Analysis.py:clusters_reduce_pca_end",
            message="Finished reduce_pca",
            data={"elapsed_s": round(time.time() - _t_red, 3)},
        )
        # endregion agent log
    else:
        try:
            # region agent log
            _t_red = time.time()
            _agent_log(
                run_id="pre-fix",
                hypothesis_id="F",
                location="view/pages/2_Analysis.py:clusters_reduce_umap_start",
                message="Starting reduce_umap",
                data={},
            )
            # endregion agent log
            X_2d, _ = reduce_umap(Xc)
            # region agent log
            _agent_log(
                run_id="pre-fix",
                hypothesis_id="F",
                location="view/pages/2_Analysis.py:clusters_reduce_umap_end",
                message="Finished reduce_umap",
                data={"elapsed_s": round(time.time() - _t_red, 3)},
            )
            # endregion agent log
        except Exception:
            st.warning("UMAP failed; falling back to PCA.")
            X_2d, _ = reduce_pca(Xc)

    cluster_algo = st.selectbox(
        "Color by cluster labels from",
        ["K-Means", "DBSCAN", "Hierarchical"],
        key="color_algo",
    )
    if cluster_algo == "K-Means":
        color_labels = chosen_labels
    elif cluster_algo == "DBSCAN":
        color_labels = db_result["labels"]
    else:
        color_labels = hc_result["labels"]

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=color_labels,
        cmap="tab10",
        edgecolors="black",
        linewidths=0.5,
        s=80,
    )
    if "name" in merged_df_c.columns:
        for i, name in enumerate(merged_df_c["name"]):
            ax.annotate(
                str(name)[:20],
                (X_2d[i, 0], X_2d[i, 1]),
                fontsize=7,
                alpha=0.7,
            )
    ax.set_title(f"{viz_method} — colored by {cluster_algo}")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    st.pyplot(fig)
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 – Hidden Gems
# ═══════════════════════════════════════════════════════════════════════════
with tab_gems:
    st.header("Hidden Gem Ranking")
    # region agent log
    _agent_log(
        run_id="pre-fix",
        hypothesis_id="E",
        location="view/pages/2_Analysis.py:tab_gems_entry",
        message="Entered Hidden Gems tab block",
        data={},
    )
    # endregion agent log

    if artifacts is not None:
        gem_scores = artifacts["gem_scores"].copy()
    else:
        sent_for_gems = _sentiment_per_game(reviews_df)
        gem_scores = _gem_scores(games_df, sent_for_gems)

    # region agent log
    _agent_log(
        run_id="pre-fix",
        hypothesis_id="B",
        location="view/pages/2_Analysis.py:tab_gems_scores",
        message="Computed/loaded gem_scores",
        data={
            "gem_scores_rows": int(len(gem_scores)) if hasattr(gem_scores, "__len__") else None,
            "gem_scores_cols": int(len(getattr(gem_scores, "columns", []))),
            "has_hidden_gem_score": bool(getattr(gem_scores, "columns", []) is not None and "hidden_gem_score" in gem_scores.columns),
            "has_quality_score": bool(getattr(gem_scores, "columns", []) is not None and "quality_score" in gem_scores.columns),
            "has_visibility_score": bool(getattr(gem_scores, "columns", []) is not None and "visibility_score" in gem_scores.columns),
        },
    )
    # endregion agent log

    meta_cols = [
        c
        for c in [
            "id",
            "release_date",
            "genre_names",
            "category_names",
            "windows",
            "mac",
            "linux",
            "reviewScore",
            "metacritic",
        ]
        if c in games_df.columns
    ]
    gem_df = gem_scores.merge(games_df[meta_cols], on="id", how="left")
    gem_df["release_date_parsed"] = pd.to_datetime(gem_df.get("release_date"), errors="coerce")

    st.subheader("Filters")
    date_series = gem_df["release_date_parsed"].dropna()
    date_range = None
    if not date_series.empty:
        date_range = st.date_input(
            "Release date range",
            value=(date_series.min().date(), date_series.max().date()),
        )

    genre_options = _multi_value_options(gem_df.get("genre_names", pd.Series([], dtype=str)))
    selected_genres = st.multiselect("Genres", options=genre_options)

    category_options = _multi_value_options(gem_df.get("category_names", pd.Series([], dtype=str)))
    selected_categories = st.multiselect("Categories", options=category_options)

    platform_options = [p for p in ["windows", "mac", "linux"] if p in gem_df.columns]
    selected_platforms = st.multiselect("Platforms", options=platform_options)

    hg_min = float(gem_df["hidden_gem_score"].min())
    hg_max = float(gem_df["hidden_gem_score"].max())
    hg_range = st.slider("Hidden gem score range", hg_min, hg_max, (hg_min, hg_max))

    q_min = float(gem_df["quality_score"].min())
    q_max = float(gem_df["quality_score"].max())
    q_range = st.slider("Quality score range", q_min, q_max, (q_min, q_max))

    v_min = float(gem_df["visibility_score"].min())
    v_max = float(gem_df["visibility_score"].max())
    v_range = st.slider("Visibility score range", v_min, v_max, (v_min, v_max))

    # region agent log
    _agent_log(
        run_id="pre-fix",
        hypothesis_id="B",
        location="view/pages/2_Analysis.py:tab_gems_ranges",
        message="Hidden gem slider ranges",
        data={
            "hg_min": hg_min,
            "hg_max": hg_max,
            "q_min": q_min,
            "q_max": q_max,
            "v_min": v_min,
            "v_max": v_max,
            "hg_all_finite": _isfinite_series(gem_df["hidden_gem_score"]),
            "q_all_finite": _isfinite_series(gem_df["quality_score"]),
            "v_all_finite": _isfinite_series(gem_df["visibility_score"]),
            "gem_df_rows": int(len(gem_df)),
        },
    )
    # endregion agent log

    rs_range = None
    if "reviewScore" in gem_df.columns:
        review_vals = gem_df["reviewScore"].fillna(0).astype(int)
        rs_min = int(review_vals.min())
        rs_max = int(review_vals.max())
        rs_range = st.slider("Review score range", rs_min, rs_max, (rs_min, rs_max))

    mc_range = None
    if "metacritic" in gem_df.columns:
        meta_vals = gem_df["metacritic"].fillna(0).astype(int)
        mc_min = int(meta_vals.min())
        mc_max = int(meta_vals.max())
        mc_range = st.slider("Metacritic range", mc_min, mc_max, (mc_min, mc_max))

    sort_options = [c for c in ["hidden_gem_score", "quality_score", "visibility_score", "release_date", "name"] if c in gem_df.columns]
    sort_col = st.selectbox("Sort by", options=sort_options, index=0)
    sort_desc = st.checkbox("Descending", value=True)

    filtered_df = gem_df.copy()
    if date_range is not None:
        if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = date_range
            end_date = date_range
        filtered_df = filtered_df[
            filtered_df["release_date_parsed"].between(
                pd.to_datetime(start_date), pd.to_datetime(end_date)
            )
        ]
    if selected_genres:
        selected_genres_set = set(selected_genres)
        filtered_df = filtered_df[
            filtered_df["genre_names"]
            .fillna("")
            .map(lambda v: bool(_split_multi_value_cell(v) & selected_genres_set))
        ]
    if selected_categories:
        selected_categories_set = set(selected_categories)
        filtered_df = filtered_df[
            filtered_df["category_names"]
            .fillna("")
            .map(lambda v: bool(_split_multi_value_cell(v) & selected_categories_set))
        ]
    if selected_platforms:
        platform_mask = np.zeros(len(filtered_df), dtype=bool)
        for platform in selected_platforms:
            platform_mask |= filtered_df[platform].fillna(False).astype(bool).to_numpy()
        filtered_df = filtered_df[platform_mask]

    filtered_df = filtered_df[
        filtered_df["hidden_gem_score"].between(*hg_range)
        & filtered_df["quality_score"].between(*q_range)
        & filtered_df["visibility_score"].between(*v_range)
    ]
    if rs_range is not None:
        filtered_df = filtered_df[
            filtered_df["reviewScore"].fillna(0).astype(int).between(*rs_range)
        ]
    if mc_range is not None:
        filtered_df = filtered_df[
            filtered_df["metacritic"].fillna(0).astype(int).between(*mc_range)
        ]

    if sort_col == "release_date":
        filtered_df = filtered_df.sort_values(
            "release_date_parsed", ascending=not sort_desc, na_position="last"
        )
    else:
        filtered_df = filtered_df.sort_values(sort_col, ascending=not sort_desc, na_position="last")

    st.subheader("Top hidden gems")
    st.caption(f"{len(filtered_df)} games after filters")
    st.dataframe(filtered_df, use_container_width=True)

    st.subheader("Score distributions")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.hist(filtered_df["quality_score"], bins=20, edgecolor="black")
        ax.set_xlabel("Quality score")
        ax.set_ylabel("Games")
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        fig, ax = plt.subplots()
        ax.hist(filtered_df["visibility_score"], bins=20, edgecolor="black")
        ax.set_xlabel("Visibility score (higher = more hidden)")
        ax.set_ylabel("Games")
        st.pyplot(fig)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        filtered_df["quality_score"],
        filtered_df["visibility_score"],
        s=80,
        edgecolors="black",
        linewidths=0.5,
    )
    if "name" in filtered_df.columns:
        for _, row in filtered_df.iterrows():
            ax.annotate(
                str(row["name"])[:20],
                (row["quality_score"], row["visibility_score"]),
                fontsize=7,
                alpha=0.7,
            )
    ax.set_xlabel("Quality")
    ax.set_ylabel("Hidden-ness (inverse visibility)")
    ax.set_title("Quality vs. Visibility — top-right = hidden gem")
    st.pyplot(fig)
    plt.close(fig)

    # Anomaly detection
    st.subheader("Anomaly detection (Isolation Forest)")
    if len(games_df) >= 5:
        if artifacts is not None and len(artifacts["anomaly_labels"]) == len(games_df):
            anomaly_labels = artifacts["anomaly_labels"]
        else:
            sent_for_anom = _sentiment_per_game(reviews_df)
            _, X_anom, _ = _build_features(games_df, sent_for_anom, None)
            anomaly_labels = detect_anomalies(X_anom)
        anomaly_df = (
            games_df[["id", "name"]].copy()
            if "name" in games_df.columns
            else games_df[["id"]].copy()
        )
        anomaly_df["anomaly"] = np.where(
            anomaly_labels == -1, "Outlier", "Normal",
        )
        anomaly_df = anomaly_df[anomaly_df["id"].isin(filtered_df["id"])]
        st.dataframe(anomaly_df, use_container_width=True)
        st.caption(
            "Outliers are games that differ significantly from the majority — "
            "potential hidden gems or unusual titles."
        )
    else:
        st.info("Need at least 5 games for anomaly detection.")
