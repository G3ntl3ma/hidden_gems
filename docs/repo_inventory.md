# Repository Inventory and Cleanup Baseline

This inventory records what is source code vs generated/local output and what can be pruned conservatively.

## Source Code (tracked)
- `api/` - database and external API client integration.
- `models/` - analysis and ML pipeline modules.
- `scripts/` - operational CLI entrypoints.
- `steam_scraper/` - ingestion pipeline modules.
- `view/` - Streamlit application and pages.
- `tests/` - test package placeholder.

## Project Configuration (tracked)
- `.gitignore`
- `requirements.txt`
- `prisma/schema.prisma`
- `.env.example`

## Generated / Build Outputs (policy required)
- `prisma/generated/` - generated Prisma client code.
- `artifacts/analysis/` - generated analysis artifacts and manifest.
- `artifacts/models/` - generated model outputs, reports, and intermediate artifacts.

## Local Data Outputs (non-source)
- `data/local/steam_appids.csv`
- `data/local/steam_games_full.csv`
- `data/local/steam_reviews_full.csv`
- `data/local/steam_games_clean.csv`
- `data/local/steam_reviews_clean.csv`
- `data/local/curated_steam_labels.csv`
- `data/local/training_dataset*.csv`
- `data/local/data_quality_report.json`

## Reference Data (tracked)
- `data/reference/steam_hidden_gems/`
- `data/reference/steam_hype_flops/`

## Conservative Cleanup Candidates
- Safe to remove locally: `__pycache__/` directories.
- Keep local data outputs under `data/local/` as the canonical runtime location.
- Keep generated analysis/model artifacts under `artifacts/`.
- Keep `prisma/generated/` tracked for now to avoid breaking environments, but align docs and ignore policy.
