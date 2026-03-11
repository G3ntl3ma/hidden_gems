from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from prisma.generated import Prisma

from api.db import db_session
from steam_scraper.config import get_api_config, get_paths_config
from steam_scraper.http_client import HttpClient, SteamSpyClient
from steam_scraper.parsers import (
    StoreParsed,
    parse_reviews_list,
    parse_reviews_summary,
    parse_steamspy_appdetails,
    parse_store_appdetails,
)


STORE_APPDETAILS_PATH = "/api/appdetails"
STORE_REVIEWS_PATH_TEMPLATE = "/appreviews/{appid}"


def _read_appids(csv_path: str) -> list[int]:
    path = Path(csv_path)
    appids: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        # Simple CSV with a header "appid" followed by one appid per line.
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("appid")
            if not raw:
                continue
            try:
                appids.append(int(raw))
            except ValueError:
                continue
    return appids


def _fetch_store_appdetails(http: HttpClient, appid: int) -> StoreParsed | None:
    cfg = get_api_config()
    base = cfg.store_base_url.rstrip("/")
    url = f"{base}{STORE_APPDETAILS_PATH}"
    # The appdetails API can take multiple appids, but we keep it simple: one per call.
    import requests

    try:
        raw = http.get_json(url, params={"appids": str(appid)})
    except requests.RequestException:
        return None
    return parse_store_appdetails(appid, raw)


def _fetch_steamspy_appdetails(steamspy: SteamSpyClient, appid: int) -> dict[str, Any] | None:
    raw = steamspy.get_appdetails(appid)
    if raw is None:
        return None
    return parse_steamspy_appdetails(raw)


def _fetch_reviews(
    http: HttpClient, appid: int
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    cfg = get_api_config()
    base = cfg.store_base_url.rstrip("/")
    path = STORE_REVIEWS_PATH_TEMPLATE.format(appid=appid)
    url = f"{base}{path}"
    base_params = {
        "json": 1,
        "day_range": 30,
        "start_date": -1,
        "end_date": -1,
        "date_range_type": "all",
        "filter": "all",  # fetch actual reviews, not just summary sample
        "language": "english,dutch",
        "l": "english",
        "review_type": "all",
        "purchase_type": "all",
        "playtime_filter_min": 0,
        "playtime_filter_max": 0,
        "filter_offtopic_activity": 0,
        "num_per_page": 100,
    }
    import requests

    summary: dict[str, Any] | None = None
    all_reviews: list[dict[str, Any]] = []
    cursor = "*"
    seen_cursors: set[str] = set()

    while True:
        params = dict(base_params)
        params["cursor"] = cursor

        try:
            raw = http.get_json(url, params=params)
        except requests.RequestException:
            break

        if summary is None:
            summary = parse_reviews_summary(raw)

        page_reviews = parse_reviews_list(appid, raw)
        if not page_reviews:
            break
        all_reviews.extend(page_reviews)

        next_cursor = raw.get("cursor")
        if not isinstance(next_cursor, str):
            break
        if next_cursor in seen_cursors or next_cursor == cursor:
            break
        seen_cursors.add(cursor)
        cursor = next_cursor

    return summary, all_reviews


def _merge_record(
    appid: int,
    store_parsed: StoreParsed | None,
    steamspy_parsed: dict[str, Any] | None,
    reviews_parsed: dict[str, Any] | None,
) -> dict[str, Any]:
    # Base record with sane defaults for Game fields.
    record: dict[str, Any] = {
        "id": appid,
        "name": "",
        "required_age": 0,
        "is_free": False,
        "detailed_description": "",
        "about_the_game": "",
        "short_description": "",
        "supported_languages": "",
        "header_image": "",
        "developers": "",
        "publishers": "",
        "windows": False,
        "mac": False,
        "linux": False,
        "metacritic": None,
        "release_date": None,
        "coming_soon": False,
        "owners_min": 0,
        "owners_max": 0,
        "average_2weeks": 0,
        "average_forever": 0,
        # Prisma Game model does not include medians or ccu, but keep them for CSV.
        "median_forever": 0,
        "median_2weeks": 0,
        "ccu": 0,
        "reviewNumReviews": None,
        "reviewScore": None,
        "reviewScoreDesc": None,
        "reviewTotalPositive": None,
        "reviewTotalNegative": None,
        "reviewTotalReviews": None,
        "store_status": "missing",
        "steamspy_status": "missing",
        "reviews_status": "missing",
        "last_updated": datetime.utcnow().isoformat(timespec="seconds"),
    }

    if store_parsed is not None:
        record.update(store_parsed.game_data)
        record["store_status"] = "ok"

    if steamspy_parsed is not None:
        record["owners_min"] = steamspy_parsed.get("owners_min", 0)
        record["owners_max"] = steamspy_parsed.get("owners_max", 0)
        record["average_2weeks"] = steamspy_parsed.get("average_2weeks", 0)
        record["average_forever"] = steamspy_parsed.get("average_forever", 0)
        record["median_forever"] = steamspy_parsed.get("median_forever", 0)
        record["median_2weeks"] = steamspy_parsed.get("median_2weeks", 0)
        record["ccu"] = steamspy_parsed.get("ccu", 0)
        record["steamspy_status"] = "ok"

    if reviews_parsed is not None:
        record.update(reviews_parsed)
        record["reviews_status"] = "ok"

    return record


def _ensure_developer_relations(db: Prisma, game_id: int, developer_names: Iterable[str]) -> None:
    for name in developer_names:
        name = name.strip()
        if not name:
            continue

        existing_link = db.gamedeveloper.find_unique(
            where={"gameId_developerName": {"gameId": game_id, "developerName": name}}
        )
        if existing_link is None:
            # If this game-developer pair is new, create it and increment developer counter.
            db.developer.upsert(
                where={"name": name},
                data={
                    "create": {"name": name, "amount_games_published": 1},
                    "update": {"amount_games_published": {"increment": 1}},
                },
            )
            db.gamedeveloper.create(
                data={
                    "gameId": game_id,
                    "developerName": name,
                }
            )


def _ensure_publisher_relations(db: Prisma, game_id: int, publisher_names: Iterable[str]) -> None:
    for name in publisher_names:
        name = name.strip()
        if not name:
            continue

        existing_link = db.gamepublisher.find_unique(
            where={"gameId_publisherName": {"gameId": game_id, "publisherName": name}}
        )
        if existing_link is None:
            db.publisher.upsert(
                where={"name": name},
                data={
                    "create": {"name": name, "amount_games_published": 1},
                    "update": {"amount_games_published": {"increment": 1}},
                },
            )
            db.gamepublisher.create(
                data={
                    "gameId": game_id,
                    "publisherName": name,
                }
            )


def _ensure_category_relations(db: Prisma, game_id: int, categories: Iterable[dict[str, Any]]) -> None:
    for cat in categories:
        cid = cat.get("id")
        name = (cat.get("name") or "").strip()
        if cid is None:
            continue
        try:
            cid_int = int(cid)
        except (TypeError, ValueError):
            continue

        db.category.upsert(
            where={"id": cid_int},
            data={
                "create": {"id": cid_int, "name": name or str(cid_int)},
                "update": {"name": name or str(cid_int)},
            },
        )

        existing_link = db.gamecategory.find_unique(
            where={"gameId_categoryId": {"gameId": game_id, "categoryId": cid_int}}
        )
        if existing_link is None:
            db.gamecategory.create(
                data={
                    "gameId": game_id,
                    "categoryId": cid_int,
                }
            )


def _ensure_genre_relations(db: Prisma, game_id: int, genres: Iterable[dict[str, Any]]) -> None:
    for gen in genres:
        gid = gen.get("id")
        name = (gen.get("name") or "").strip()
        if gid is None:
            continue
        try:
            gid_int = int(gid)
        except (TypeError, ValueError):
            continue

        db.genre.upsert(
            where={"id": gid_int},
            data={
                "create": {"id": gid_int, "name": name or str(gid_int)},
                "update": {"name": name or str(gid_int)},
            },
        )

        existing_link = db.gamegenre.find_unique(
            where={"gameId_genreId": {"gameId": game_id, "genreId": gid_int}}
        )
        if existing_link is None:
            db.gamegenre.create(
                data={
                    "gameId": game_id,
                    "genreId": gid_int,
                }
            )


def _upsert_game(db: Prisma, record: dict[str, Any]) -> None:
    """
    Insert or update a Game row from the merged record.
    """
    game_data = {
        "id": record["id"],
        "name": record["name"],
        "required_age": record["required_age"],
        "is_free": record["is_free"],
        "detailed_description": record["detailed_description"],
        "about_the_game": record["about_the_game"],
        "short_description": record["short_description"],
        "supported_languages": record["supported_languages"],
        "header_image": record["header_image"],
        "developers": record["developers"],
        "publishers": record["publishers"],
        "windows": record["windows"],
        "mac": record["mac"],
        "linux": record["linux"],
        "metacritic": record.get("metacritic"),
        "release_date": record.get("release_date"),
        "coming_soon": record.get("coming_soon", False),
        "owners_min": record["owners_min"],
        "owners_max": record["owners_max"],
        "average_2weeks": record["average_2weeks"],
        "average_forever": record["average_forever"],
        "reviewNumReviews": record.get("reviewNumReviews"),
        "reviewScore": record.get("reviewScore"),
        "reviewScoreDesc": record.get("reviewScoreDesc"),
        "reviewTotalPositive": record.get("reviewTotalPositive"),
        "reviewTotalNegative": record.get("reviewTotalNegative"),
        "reviewTotalReviews": record.get("reviewTotalReviews"),
    }

    db.game.upsert(
        where={"id": record["id"]},
        data={
            "create": game_data,
            "update": game_data,
        },
    )


def _upsert_reviews(db: Prisma, reviews: list[dict[str, Any]]) -> None:
    """
    Insert or update Review rows for a single game.
    """
    for r in reviews:
        db.review.upsert(
            where={"id": r["id"]},
            data={
                "create": r,
                "update": r,
            },
        )


def _csv_headers() -> list[str]:
    return [
        "id",
        "name",
        "required_age",
        "is_free",
        "detailed_description",
        "about_the_game",
        "short_description",
        "supported_languages",
        "header_image",
        "developers",
        "publishers",
        "windows",
        "mac",
        "linux",
        "metacritic",
        "release_date",
        "coming_soon",
        "owners_min",
        "owners_max",
        "average_2weeks",
        "average_forever",
        "median_forever",
        "median_2weeks",
        "ccu",
        "reviewNumReviews",
        "reviewScore",
        "reviewScoreDesc",
        "reviewTotalPositive",
        "reviewTotalNegative",
        "reviewTotalReviews",
        "store_status",
        "steamspy_status",
        "reviews_status",
        "last_updated",
    ]


def _review_csv_headers() -> list[str]:
    return [
        "id",
        "gameId",
        "authorSteamId",
        "authorPlaytimeForever",
        "authorPlaytimeAtReview",
        "authorLastPlayed",
        "language",
        "review",
        "votedUp",
        "votesUp",
        "votesFunny",
        "weightedVoteScore",
        "writtenDuringEarlyAccess",
        "timestampCreated",
        "timestampUpdated",
    ]

def run_pipeline(
    *,
    appids_csv: str | None = None,
    output_csv: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> None:
    paths_cfg = get_paths_config()
    csv_in = appids_csv or paths_cfg.appids_csv
    csv_out = output_csv or paths_cfg.output_csv
    reviews_out = paths_cfg.reviews_csv

    all_appids = _read_appids(csv_in)
    if offset:
        all_appids = all_appids[offset:]
    if limit is not None:
        all_appids = all_appids[:limit]

    http = HttpClient()
    steamspy = SteamSpyClient()

    csv_path = Path(csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    reviews_path = Path(reviews_out)
    reviews_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        db_session() as db,
        csv_path.open("w", encoding="utf-8", newline="") as f_out,
        reviews_path.open("w", encoding="utf-8", newline="") as f_reviews,
    ):
        writer = csv.DictWriter(f_out, fieldnames=_csv_headers())
        writer.writeheader()

        reviews_writer = csv.DictWriter(f_reviews, fieldnames=_review_csv_headers())
        reviews_writer.writeheader()

        for idx, appid in enumerate(all_appids, start=1):
            store_parsed = _fetch_store_appdetails(http, appid)
            steamspy_parsed = _fetch_steamspy_appdetails(steamspy, appid)
            reviews_summary, reviews_list = _fetch_reviews(http, appid)

            record = _merge_record(appid, store_parsed, steamspy_parsed, reviews_summary)

            # Persist to DB
            _upsert_game(db, record)
            if reviews_list:
                _upsert_reviews(db, reviews_list)

            # Maintain relations if we have store data.
            if store_parsed is not None:
                _ensure_developer_relations(db, appid, store_parsed.developers)
                _ensure_publisher_relations(db, appid, store_parsed.publishers)
                _ensure_category_relations(db, appid, store_parsed.categories)
                _ensure_genre_relations(db, appid, store_parsed.genres)

            # CSV row: ensure release_date is serializable.
            row = dict(record)
            rd = row.get("release_date")
            if isinstance(rd, datetime):
                row["release_date"] = rd.date().isoformat()
            writer.writerow(row)

            # Write reviews to separate CSV, normalizing datetimes.
            for review in reviews_list:
                r_row = dict(review)
                for key in ("authorLastPlayed", "timestampCreated", "timestampUpdated"):
                    val = r_row.get(key)
                    if isinstance(val, datetime):
                        r_row[key] = val.isoformat()
                reviews_writer.writerow(r_row)

