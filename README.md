# Hidden Gems

Find underrated Steam games by combining NLP, clustering, and anomaly detection.

## Table of Contents
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Data Pipeline](#data-pipeline)
- [Supervised Training Workflow](#supervised-training-workflow)
- [Analysis Dashboard](#analysis-dashboard)
- [Troubleshooting](#troubleshooting)

## Project Structure

```
.
├── api/                 Prisma + DB integration and shared API clients
├── apps/                App entrypoint wrappers
├── artifacts/           Generated analysis/model artifacts (gitignored)
│   ├── analysis/
│   └── models/
├── data/
│   ├── local/           Generated local CSV/JSON outputs (gitignored)
│   └── reference/       Curated reference datasets (tracked)
├── docs/
├── models/              Core analysis and ML modules
├── prisma/              Prisma schema and generated client
├── scripts/             Operational CLI scripts
├── steam_scraper/       Steam ingestion pipeline
└── view/                Streamlit app and pages
```

## Quickstart

Run all commands below from the repository root (`hidden_gems/`) unless noted otherwise.

1) Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Configure environment:

```bash
cp .env.example .env
```

Set `DATABASE_URL` in `.env`:

```env
DATABASE_URL="file:./dev.db"
```

If omitted, the app falls back to a repo-root SQLite path (`file:/.../hidden_gems/dev.db`).

Set Steam API credentials in `.env` before exporting app IDs:

```env
STEAM_WEB_API_KEY="your_steam_web_api_key"
```

3) Sync Prisma schema and generate client:

```bash
npx prisma@5.17.0 db push --schema prisma/schema.prisma
python -m prisma generate --schema prisma/schema.prisma
```

4) Run Streamlit:

```bash
streamlit run view/app.py
```

## Data Pipeline

### Step 1: Export Steam App IDs

```bash
python -m scripts.export_steam_appids
```

Requires `STEAM_WEB_API_KEY` in the environment (for example via `.env`).

Default output: `data/local/steam_appids.csv`

### Step 2: Scrape games and reviews

```bash
python -m steam_scraper.main
```

Resume mode:

```bash
python -m steam_scraper.main --resume
```

Defaults:
- games: `data/local/steam_games_full.csv`
- reviews: `data/local/steam_reviews_full.csv`

### Step 3: Clean collected datasets

```bash
python -m scripts.clean_collected_data
```

Defaults:
- input games: `data/local/steam_games_full.csv`
- input reviews: `data/local/steam_reviews_full.csv`
- output games: `data/local/steam_games_clean.csv`
- output reviews: `data/local/steam_reviews_clean.csv`
- quality report: `data/local/data_quality_report.json`

### Step 4: Precompute analysis artifacts (recommended)

```bash
python -m scripts.precompute_analysis
```

Defaults:
- games input: `data/local/steam_games_clean.csv`
- reviews input: `data/local/steam_reviews_clean.csv`
- cache output: `artifacts/analysis/`

## Supervised Training Workflow

1) Export labels from Streamlit Label Curator:
- Download the CSV from the page and save it as `data/local/curated_steam_labels.csv`.

2) Build the training dataset:

```bash
python -m scripts.build_training_dataset \
  --games-csv data/local/steam_games_clean.csv \
  --reviews-csv data/local/steam_reviews_clean.csv \
  --labels-csv data/local/curated_steam_labels.csv \
  --out-csv data/local/training_dataset.csv
```

3) Train baseline/champion model:

```bash
python -m models.train \
  --csv data/local/training_dataset.csv \
  --target is_gem \
  --out artifacts/models/model.joblib \
  --metrics-out artifacts/models/metrics.json \
  --report-out artifacts/models/run_report.json
```

4) Optional weight learning:

```bash
python -m scripts.learn_hidden_gem_weights \
  --csv data/local/training_dataset.csv \
  --weights-out artifacts/models/hidden_gem_weights.json \
  --report-out artifacts/models/hidden_gem_weight_report.json
```

5) Optional anomaly benchmark:

```bash
python -m scripts.benchmark_anomaly_methods \
  --games-csv data/local/steam_games_clean.csv \
  --reviews-csv data/local/steam_reviews_clean.csv \
  --labels-csv data/local/training_dataset.csv \
  --out artifacts/models/anomaly_benchmark.json
```

6) Predict:

```bash
python -m models.predict \
  --model artifacts/models/model.joblib \
  --csv data/local/training_dataset.csv \
  --out artifacts/models/predictions.json
```

## Analysis Dashboard

Run:

```bash
streamlit run view/app.py
```

Use this entrypoint (`view/app.py`) for multipage navigation stability.

Main pages:
- `view/pages/1_Data_Explorer.py`
- `view/pages/2_Analysis.py`
- `view/pages/3_Label_Curator.py`

The analysis page loads from `artifacts/analysis/` when cache artifacts are fresh, and recomputes automatically when input signatures change.

## Troubleshooting

- Prisma error about URL format: ensure `.env` uses `DATABASE_URL="file:./dev.db"`.
- Missing Prisma client: run `python -m prisma generate --schema prisma/schema.prisma`.
- Missing DB tables (Label Curator fails): run `npx prisma@5.17.0 db push --schema prisma/schema.prisma`.
- Missing Steam API key: ensure `STEAM_WEB_API_KEY` is set before running export.
- Missing cleaned datasets: run `python -m scripts.clean_collected_data`.
- Missing analysis artifacts: run `python -m scripts.precompute_analysis --force`.

