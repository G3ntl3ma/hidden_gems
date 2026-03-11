"""
Steam data ingestion pipeline.

This package reads Steam app IDs from `steam_appids.csv`, fetches data from:
- Steam Store appdetails API
- SteamSpy appdetails API
- Steam Store reviews summary API

and writes the merged data both to CSV and into the existing Prisma-backed
SQLite database defined in `prisma/schema.prisma`.
"""

