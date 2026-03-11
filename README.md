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

Generate Prisma client + create local SQLite DB:

```bash
# 1) Ensure .env has a local SQLite URL
echo 'DATABASE_URL="file:./dev.db"' >> .env

# 2) Push the Prisma schema to the local DB (Prisma 5.x)
npx prisma@5.17.0 db push --schema prisma/schema.prisma
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

## Scraping full Steam game data (CSV + local DB)

The project includes a scraper that:

- Reads all appids from `steam_appids.csv`
- Calls:
  - Steam Store appdetails: `https://store.steampowered.com/api/appdetails?appids={appid}`
  - SteamSpy appdetails: `https://steamspy.com/api.php?request=appdetails&appid={appid}` ([docs](https://steamspy.com/api.php?appdetails&appid=440))
  - Steam Store review summary: `https://store.steampowered.com/appreviews/{appid}?cursor=*&json=1&...`
- Merges the results
- Writes:
  - `steam_games_full.csv` in the project root
  - Rows into the Prisma-managed SQLite DB (`dev.db`) using the existing schema in `prisma/schema.prisma`

To run the scraper:

```bash

# push the db 
prisma db push --schema prisma/schema.prisma
# From the project root, with the virtualenv activated
python -m steam_scraper.main --limit 100   # small test batch

# Or run for all appids
python -m steam_scraper.main
```

## Migrating data from local SQLite to Turso/libsql

After running the scraper, all data lives in the local SQLite file referenced by
`DATABASE_URL` (e.g. `file:./dev.db`). To move this data into a remote Turso/libsql
database (such as `libsql://hidden-gems-test-g3ntl3ma.aws-eu-west-1.turso.io`),
you can:

1. Dump the local SQLite database to SQL:

   ```bash
   sqlite3 dev.db ".dump" > dump.sql
   ```

2. Use any libsql-compatible client or shell that accepts the `libsql://` URL
   to import `dump.sql`. For example, with a generic libsql shell:

   ```bash
   libsql-shell "libsql://hidden-gems-test-g3ntl3ma.aws-eu-west-1.turso.io"
   # inside the shell:
   .read dump.sql
   ```

This leaves Prisma + the Python client using the local SQLite file for development,
while your Turso instance holds a copy of the same schema and data for remote use.
