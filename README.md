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

The sidebar includes **Label curator** (`view/pages/3_Label_Curator.py`): enter a Steam app ID and whether it is a gem or not. Labels are stored in SQLite as the `CuratedSteamLabel` table. After changing [`prisma/schema.prisma`](prisma/schema.prisma), run step 3 again (`db push` and `python -m prisma generate`). The local `dev.db` file is gitignored, so use the page’s **Download labels as CSV** export for backups and for merging with training features (for example with [`models/train.py`](models/train.py)).

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
- **Prisma DB** (`dev.db`) — `Game`, `Review`, `CuratedSteamLabel` (manual training labels), and related tables

`steam_games_full.csv` now includes `genre_names` and `category_names` as
semicolon-separated fields for dashboard filtering.

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

`steam_games_clean.csv` preserves `genre_names` and `category_names`.

---

## Training a gem / not-gem classifier (supervised)

This project’s training labels come from the Streamlit **Label curator** page, exported as `curated_steam_labels.csv` (columns: `appid`, `is_gem`).

**1) Export labels from Streamlit**

Run the dashboard and use **Label curator → Download labels as CSV** to save `curated_steam_labels.csv` in the project root.

**2) Build a training dataset by joining labels to cleaned data**

This step merges:
- `steam_games_clean.csv` (game metadata)
- `steam_reviews_clean.csv` (reviews, used to compute per-game sentiment aggregates)
- `curated_steam_labels.csv` (the target labels)

```bash
python scripts/build_training_dataset.py \
  --games-csv steam_games_clean.csv \
  --reviews-csv steam_reviews_clean.csv \
  --labels-csv curated_steam_labels.csv \
  --out-csv training_dataset.csv
```

**3) Train**

```bash
python -m models.train --csv training_dataset.csv --target is_gem
```

### Step 4: Precompute analysis artifacts (recommended)

Precompute expensive analysis results (sentiment, topics, feature matrix,
hidden-gem scores, anomaly labels) so Streamlit restarts do not recompute
everything:

```bash
python scripts/precompute_analysis.py
```

Default output directory:

- **.cache/analysis/** — pickled artifacts + `manifest.json` used for stale-cache checks

Useful options:

| Option | Description |
|--------|-------------|
| `--games-in PATH` | Input cleaned games CSV (default: `steam_games_clean.csv`) |
| `--reviews-in PATH` | Input cleaned reviews CSV (default: `steam_reviews_clean.csv`) |
| `--cache-dir PATH` | Artifact directory (default: `.cache/analysis`) |
| `--force` | Recompute even if manifest is still fresh |

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

Before running analysis, complete the data flow in order:
`collect -> clean -> analyze`.
Run `python scripts/clean_collected_data.py` after scraping, then ensure
`steam_games_clean.csv` and `steam_reviews_clean.csv` exist in the project root.

### Option A: Streamlit dashboard (recommended)

The easiest way to run the full pipeline is the interactive dashboard. It loads the cleaned CSVs, runs every analysis step, and shows the results across five tabs.

```bash
streamlit run view/app.py
```

Open **Analysis** in the sidebar for the five-tab pipeline, or **Label curator** to record gem / not-gem labels per Steam app ID (stored in the database; export CSV from that page for backups and training).

**Analysis** tabs:

| Tab | What it does |
|-----|--------------|
| **EDA Overview** | Descriptive statistics, missing-data audit, review-score distribution, platform support, ownership tiers, correlation heatmap |
| **Sentiment** | VADER sentiment analysis on every review, compound-score histogram, sentiment split by vote direction, per-game aggregation table |
| **Topics** | LDA or NMF topic modeling on English reviews (configurable number of topics and method), top words per topic, per-game topic averages |
| **Clusters** | Feature matrix (game metadata + sentiment + topic features, scaled), K-Means with elbow/silhouette plots, DBSCAN, hierarchical clustering, 2-D scatter (PCA or UMAP) colored by cluster |
| **Hidden Gems** | Quality score, visibility score, combined hidden-gem ranking, quality-vs-visibility scatter plot, Isolation Forest anomaly detection |

Results are loaded from precomputed artifacts when available and up to date.
If artifacts are missing or stale, the dashboard recomputes and refreshes them.

The **Hidden Gems** tab also includes filters and sorting controls for release
date, genres, categories, platforms, and score ranges.

### Option B: Python scripts / notebooks

Every analysis module can be imported and used directly.

**1. Exploratory Data Analysis** (`models/eda.py`)

```python
import pandas as pd
from models.eda import numeric_summary, missing_data_audit, correlation_matrix

games = pd.read_csv("steam_games_clean.csv")
print(numeric_summary(games))
print(missing_data_audit(games))
print(correlation_matrix(games))
```

**2. Sentiment Analysis** (`models/sentiment.py`)

Uses VADER (rule-based, no training needed). The VADER lexicon is downloaded automatically on first run.

```python
from models.sentiment import add_sentiment_to_reviews, aggregate_sentiment_per_game

reviews = pd.read_csv("steam_reviews_clean.csv")

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

reviews = pd.read_csv("steam_reviews_clean.csv")
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

games = pd.read_csv("steam_games_clean.csv")

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

Composite score combining:

- **Quality core**: positivity ratio, sentiment aggregates, review score, metacritic, lifetime engagement
- **Longevity**: game age + age-adjusted engagement/review activity + persistence of recent playtime
- **Inverse visibility**: low review/activity footprint and low external coverage, with `owners_min` / `owners_max` treated as lower-weight rough signals

The final hidden-gem ranking is quality-dominant, then adjusted by inverse visibility to surface underrated titles.

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
