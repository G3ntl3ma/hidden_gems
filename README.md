# Hidden Gems

Find underrated Steam games by combining NLP, clustering, and anomaly detection.

- **api/** — Database (Prisma + SQLite) and external API clients
- **models/** — Analysis pipeline (EDA, sentiment, topic modeling, clustering, hidden-gem scoring)
- **view/** — Streamlit dashboard with interactive analysis
- **prisma/** — Prisma schema and generated Python client
- **steam_scraper/** — Steam Store + SteamSpy + reviews ingestion pipeline

---

## Quickstart

**1. Virtual environment and dependencies**

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Environment**

```bash
cp .env.example .env
```

Set `DATABASE_URL` to a local SQLite file (required for Prisma):

```bash
# In .env
DATABASE_URL="file:./dev.db"
```

**3. Database and Prisma client**

```bash
npx prisma@5.17.0 db push --schema prisma/schema.prisma
source .venv/bin/activate && python -m prisma generate --schema prisma/schema.prisma
```

**4. Run the dashboard**

```bash
streamlit run view/app.py
```

---

## Steam data pipeline

### Step 1: Export all Steam app IDs

Requires `STEAM_WEB_API_KEY` in `.env` or the environment.

```bash
python scripts/export_steam_appids.py
```

Creates `steam_appids.csv` in the project root (one `appid` per row).

### Step 2: Scrape game data and reviews

The scraper uses:

- **Steam Store** — appdetails (name, developers, publishers, categories, genres, etc.) and review summaries + up to 100 reviews per game
- **SteamSpy** — owners, average playtime

Outputs:

- **steam_games_full.csv** — one row per game
- **steam_reviews_full.csv** — one row per review (max 100 per game)
- **Prisma DB** (`dev.db`) — `Game`, `Review`, and related tables

**Run (first time or full run):**

```bash
python -m steam_scraper.main
```

**Resume** (skip games already in the DB or in the games CSV; appends to CSVs):

```bash
python -m steam_scraper.main --resume
```

**Options:**

| Option | Description |
|--------|-------------|
| `--limit N` | Process only N appids (e.g. `--limit 100` for testing) |
| `--offset N` | Skip the first N appids in the list |
| `--resume` | Skip appids already in DB / `steam_games_full.csv` and append to CSVs |
| `--appid-csv PATH` | Input CSV of appids (default: `steam_appids.csv`) |
| `--output-csv PATH` | Output games CSV (default: `steam_games_full.csv`) |

Example:

```bash
python -m steam_scraper.main --resume --limit 500
```

### Step 3: Clean collected datasets

Run a post-collection cleaning pass to remove unusable rows, normalize text, and
generate a quality report:

```bash
python scripts/clean_collected_data.py
```

Default outputs:

- **steam_games_clean.csv** — cleaned game rows (invalid/empty rows removed, fields normalized)
- **steam_reviews_clean.csv** — cleaned review rows (tokenless/invalid/duplicate reviews removed)
- **data_quality_report.json** — counters and warnings about detected data quality issues

**Options:**

| Option | Description |
|--------|-------------|
| `--games-in PATH` | Input raw games CSV (default: `steam_games_full.csv`) |
| `--reviews-in PATH` | Input raw reviews CSV (default: `steam_reviews_full.csv`) |
| `--games-out PATH` | Output cleaned games CSV (default: `steam_games_clean.csv`) |
| `--reviews-out PATH` | Output cleaned reviews CSV (default: `steam_reviews_clean.csv`) |
| `--report-out PATH` | Output quality report JSON (default: `data_quality_report.json`) |

---

## Analysis pipeline

Once data collection is complete (i.e. `steam_games_full.csv` and `steam_reviews_full.csv` exist in the project root), the analysis pipeline is ready to use.

### Option A: Streamlit dashboard (recommended)

The easiest way to run the full pipeline is the interactive dashboard. It loads the CSVs, runs every analysis step, and shows the results across five tabs.

```bash
streamlit run view/app.py
```

Navigate to **Analysis** in the sidebar. The page has five tabs:

| Tab | What it does |
|-----|--------------|
| **EDA Overview** | Descriptive statistics, missing-data audit, review-score distribution, platform support, ownership tiers, correlation heatmap |
| **Sentiment** | VADER sentiment analysis on every review, compound-score histogram, sentiment split by vote direction, per-game aggregation table |
| **Topics** | LDA or NMF topic modeling on English reviews (configurable number of topics and method), top words per topic, per-game topic averages |
| **Clusters** | Feature matrix (game metadata + sentiment + topic features, scaled), K-Means with elbow/silhouette plots, DBSCAN, hierarchical clustering, 2-D scatter (PCA or UMAP) colored by cluster |
| **Hidden Gems** | Quality score, visibility score, combined hidden-gem ranking, quality-vs-visibility scatter plot, Isolation Forest anomaly detection |

Results are cached after the first run so switching tabs is instant.

### Option B: Python scripts / notebooks

Every analysis module can be imported and used directly.

**1. Exploratory Data Analysis** (`models/eda.py`)

```python
import pandas as pd
from models.eda import numeric_summary, missing_data_audit, correlation_matrix

games = pd.read_csv("steam_games_full.csv")
print(numeric_summary(games))
print(missing_data_audit(games))
print(correlation_matrix(games))
```

**2. Sentiment Analysis** (`models/sentiment.py`)

Uses VADER (rule-based, no training needed). The VADER lexicon is downloaded automatically on first run.

```python
from models.sentiment import add_sentiment_to_reviews, aggregate_sentiment_per_game

reviews = pd.read_csv("steam_reviews_full.csv")

# Add sentiment_neg/neu/pos/compound columns to each review
reviews_with_sentiment = add_sentiment_to_reviews(reviews)

# Roll up to one row per game: mean, median, std, positive ratio, avg review length
per_game = aggregate_sentiment_per_game(reviews_with_sentiment)
print(per_game)
```

**3. Topic Modeling** (`models/topic_model.py`)

Runs TF-IDF then LDA or NMF to discover latent topics in review text. Works best with English-language reviews.

```python
from models.topic_model import build_review_topics_pipeline

reviews = pd.read_csv("steam_reviews_full.csv")
english = reviews[reviews["language"].str.lower() == "english"]

result = build_review_topics_pipeline(english, n_topics=8, method="lda")

# Top words per topic
for tid, words in result["top_words"].items():
    print(f"Topic {tid}: {', '.join(words)}")

# Per-game topic averages (one row per game, one column per topic)
print(result["game_topics"])
```

**4. Clustering** (`models/clustering.py`)

Builds a feature matrix from game metadata + sentiment + topic features, then runs clustering algorithms.

```python
from models.clustering import build_feature_matrix, run_kmeans, run_dbscan, reduce_pca

games = pd.read_csv("steam_games_full.csv")

# per_game and game_topics come from steps 2 and 3 above
merged_df, X, feature_names = build_feature_matrix(games, per_game, result["game_topics"])

# K-Means with automatic elbow/silhouette evaluation
km = run_kmeans(X)
print("Best silhouette:", max(km["silhouettes"]))

# DBSCAN (noise points = potential outliers)
db = run_dbscan(X, eps=1.5, min_samples=2)
print(f"Clusters: {db['n_clusters']}, noise: {db['n_noise']}")

# 2-D PCA for plotting
X_2d, pca = reduce_pca(X)
```

**5. Hidden-Gem Scoring** (`models/hidden_gems.py`)

Composite score combining quality signals (sentiment, positive ratio, playtime) with inverse-visibility signals (low owner count, few reviews, no metacritic).

```python
from models.hidden_gems import compute_hidden_gem_score, detect_anomalies

# Merge games with per-game sentiment first
enriched = games.merge(per_game, left_on="id", right_on="gameId", how="left")

gems = compute_hidden_gem_score(enriched)
print(gems.head(10))  # top 10 hidden gems

# Isolation Forest anomaly detection on the feature matrix
anomaly_labels = detect_anomalies(X)  # 1 = normal, -1 = outlier
```

### Analysis modules overview

```
models/
  eda.py           Descriptive stats, missing-data audit, correlations, distributions
  sentiment.py     VADER sentiment per review, aggregated per game
  topic_model.py   TF-IDF + LDA/NMF topic modeling, per-game topic distributions
  clustering.py    Feature engineering, K-Means, DBSCAN, hierarchical, PCA, UMAP
  hidden_gems.py   Quality/visibility scoring, Isolation Forest anomaly detection
```

---

## Migrating local DB to Turso / libsql

Data is stored in the SQLite file from `DATABASE_URL` (e.g. `dev.db`). To copy it to a remote Turso (libsql) instance:

**1. Dump SQLite**

```bash
sqlite3 dev.db ".dump" > dump.sql
```

**2. Import into Turso**

Use a libsql-compatible client with your `libsql://...` URL and run the SQL (e.g. `.read dump.sql` in the Turso shell). Prisma and the scraper keep using the local `dev.db`; Turso holds a separate copy for remote use.

---

## Troubleshooting

- **Prisma "url must start with file:"** — Set `DATABASE_URL="file:./dev.db"` in `.env` (SQLite only).
- **"Client hasn't been generated yet"** — Run `python -m prisma generate --schema prisma/schema.prisma` with the venv activated.
- **CSV overwritten on resume** — Use `--resume` so the scraper appends to existing games/reviews CSVs instead of overwriting.
