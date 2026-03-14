"""Interactive analysis dashboard: EDA, sentiment, topics, clustering,
and hidden-gem scoring."""

from __future__ import annotations

import sys
from pathlib import Path

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


st.set_page_config(page_title="Game Analysis", layout="wide")
st.title("Game Analysis Pipeline")

# ── data loading ──────────────────────────────────────────────────────────

GAMES_CSV = Path("steam_games_full.csv")
REVIEWS_CSV = Path("steam_reviews_full.csv")


@st.cache_data(show_spinner="Loading games …")
def load_games() -> pd.DataFrame:
    return pd.read_csv(GAMES_CSV)


@st.cache_data(show_spinner="Loading reviews …")
def load_reviews() -> pd.DataFrame:
    return pd.read_csv(REVIEWS_CSV)


if not GAMES_CSV.exists() or not REVIEWS_CSV.exists():
    st.warning(
        "CSV data files not found. Make sure `steam_games_full.csv` and "
        "`steam_reviews_full.csv` exist in the project root."
    )
    st.stop()

games_df = load_games()
reviews_df = load_reviews()

st.sidebar.metric("Games", len(games_df))
st.sidebar.metric("Reviews", len(reviews_df))


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


# ── tabs ──────────────────────────────────────────────────────────────────

tab_eda, tab_sent, tab_topics, tab_clusters, tab_gems = st.tabs(
    ["EDA Overview", "Sentiment", "Topics", "Clusters", "Hidden Gems"],
)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 – Exploratory Data Analysis
# ═══════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.header("Exploratory Data Analysis")

    st.subheader("Numeric summary — Games")
    st.dataframe(numeric_summary(games_df), use_container_width=True)

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
    corr = correlation_matrix(games_df)
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

    n_topics = st.slider("Number of topics", 3, 20, 8, key="n_topics")
    topic_method = st.selectbox("Method", ["lda", "nmf"], key="topic_method")

    english_reviews = _get_english_reviews(reviews_df)
    if len(english_reviews) < 10:
        st.info("Not enough English reviews for topic modeling (need >= 10).")
    else:
        topic_result = _run_topics(english_reviews, n_topics, topic_method)

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

    if len(games_df) < 4:
        st.warning("Need at least 4 games to run meaningful clustering.")
        st.stop()

    sent_agg = _sentiment_per_game(reviews_df)

    english_for_topics = _get_english_reviews(reviews_df)
    topic_agg = None
    if len(english_for_topics) >= 10:
        tr = _run_topics(english_for_topics, 8, "lda")
        topic_agg = tr["game_topics"]

    merged_df, X, feature_names = _build_features(games_df, sent_agg, topic_agg)

    st.write(f"Feature matrix: **{X.shape[0]}** games x **{X.shape[1]}** features")
    st.caption(f"Features: {', '.join(feature_names)}")

    # ── K-Means ───────────────────────────────────────────────────────────
    st.subheader("K-Means")
    max_k = min(10, X.shape[0] - 1)
    if max_k >= 2:
        km_results = run_kmeans(X, k_range=range(2, max_k + 1))

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
        chosen_labels = np.zeros(X.shape[0], dtype=int)

    # ── DBSCAN ────────────────────────────────────────────────────────────
    st.subheader("DBSCAN")
    eps = st.slider("eps", 0.1, 5.0, 1.5, 0.1, key="dbscan_eps")
    min_samp = st.slider("min_samples", 2, 10, 2, key="dbscan_min")
    db_result = run_dbscan(X, eps=eps, min_samples=min_samp)
    st.write(
        f"Clusters: **{db_result['n_clusters']}** | "
        f"Noise points: **{db_result['n_noise']}** | "
        f"Silhouette: **{db_result['silhouette']:.3f}**"
    )

    # ── Hierarchical ──────────────────────────────────────────────────────
    st.subheader("Hierarchical (Agglomerative)")
    n_clust = st.slider(
        "n_clusters", 2, min(10, X.shape[0] - 1), min(3, X.shape[0] - 1),
        key="hier_k",
    )
    hc_result = run_hierarchical(X, n_clusters=n_clust)
    st.write(f"Silhouette: **{hc_result['silhouette']:.3f}**")

    # ── Visualisation ─────────────────────────────────────────────────────
    st.subheader("2-D projection")
    viz_method = st.selectbox("Reduction method", ["PCA", "UMAP"], key="viz_method")

    if viz_method == "PCA":
        X_2d, _ = reduce_pca(X)
    else:
        try:
            X_2d, _ = reduce_umap(X)
        except Exception:
            st.warning("UMAP failed; falling back to PCA.")
            X_2d, _ = reduce_pca(X)

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
    if "name" in merged_df.columns:
        for i, name in enumerate(merged_df["name"]):
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

    sent_for_gems = _sentiment_per_game(reviews_df)
    gem_df = _gem_scores(games_df, sent_for_gems)

    st.subheader("Top hidden gems")
    st.dataframe(gem_df, use_container_width=True)

    st.subheader("Score distributions")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.hist(gem_df["quality_score"], bins=20, edgecolor="black")
        ax.set_xlabel("Quality score")
        ax.set_ylabel("Games")
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        fig, ax = plt.subplots()
        ax.hist(gem_df["visibility_score"], bins=20, edgecolor="black")
        ax.set_xlabel("Visibility score (higher = more hidden)")
        ax.set_ylabel("Games")
        st.pyplot(fig)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        gem_df["quality_score"],
        gem_df["visibility_score"],
        s=80,
        edgecolors="black",
        linewidths=0.5,
    )
    if "name" in gem_df.columns:
        for _, row in gem_df.iterrows():
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
        st.dataframe(anomaly_df, use_container_width=True)
        st.caption(
            "Outliers are games that differ significantly from the majority — "
            "potential hidden gems or unusual titles."
        )
    else:
        st.info("Need at least 5 games for anomaly detection.")
