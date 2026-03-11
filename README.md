# Hidden Gems — Python ML Starter

Starter project scaffold with:

- `api/`: database + external API connectors
- `models/`: training / inference code
- `view/`: Streamlit dashboard
- `prisma/`: Prisma schema for SQLite (Prisma Client Python)

## Quickstart

Create a virtual environment, install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy env vars:

```bash
cp .env.example .env
```

Generate Prisma client + create SQLite DB:

```bash
prisma db push --schema prisma/schema.prisma
```

Run the dashboard:

```bash
streamlit run view/app.py
```

## Training a model (CSV)

```bash
python -m models.train --csv path/to/data.csv --target target_column
```

## Exporting all Steam appids to CSV

Set your Steam Web API key in an environment variable:

```bash
export STEAM_WEB_API_KEY="your_steam_web_api_key_here"
```

Then run the export script from the project root:

```bash
python scripts/export_steam_appids.py
```

This will create a `steam_appids.csv` file in the project root containing a single column `appid` with all appids returned by the Steam `IStoreService/GetAppList` endpoint.

