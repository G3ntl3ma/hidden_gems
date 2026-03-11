from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import html
import re


@dataclass
class StoreParsed:
    game_data: dict[str, Any]
    categories: list[dict[str, Any]]
    genres: list[dict[str, Any]]
    developers: list[str]
    publishers: list[str]


_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(value: str) -> str:
    """
    Remove HTML tags and unescape entities from a string.
    """
    if not value:
        return ""
    # Drop tags, then unescape HTML entities.
    no_tags = _TAG_RE.sub("", value)
    return html.unescape(no_tags).strip()


def _parse_release_date(release_info: dict[str, Any] | None) -> tuple[datetime | None, bool]:
    if not release_info:
        return None, False
    coming_soon = bool(release_info.get("coming_soon"))
    date_str = (release_info.get("date") or "").strip()
    if not date_str:
        return None, coming_soon

    # Try a few common Steam date formats; fall back to None if parsing fails.
    for fmt in ("%b %d, %Y", "%d %b, %Y", "%b %Y", "%Y"):
        try:
            return datetime.strptime(date_str, fmt), coming_soon
        except ValueError:
            continue
    return None, coming_soon


def parse_store_appdetails(appid: int, raw_json: Any) -> StoreParsed | None:
    """
    Parse the Steam Store appdetails response for a single app.

    Expected shape:
    {
      "440": {
        "success": true,
        "data": {
          "name": "...",
          ...
        }
      }
    }
    """
    if not isinstance(raw_json, dict):
        return None

    entry = raw_json.get(str(appid))
    if not isinstance(entry, dict) or not entry.get("success"):
        return None

    data = entry.get("data") or {}
    if not isinstance(data, dict):
        return None

    name = _strip_html(data.get("name") or "")
    required_age = int(data.get("required_age") or 0)
    is_free = bool(data.get("is_free"))
    detailed_description = _strip_html(data.get("detailed_description") or "")
    about_the_game = _strip_html(data.get("about_the_game") or "")
    short_description = _strip_html(data.get("short_description") or "")
    # Supported languages is HTML with commas; strip tags and replace comma-separators
    # with semicolons so that CSV tools treat the whole field as one logical list.
    supported_languages_raw = _strip_html(data.get("supported_languages") or "")
    supported_languages = supported_languages_raw.replace(", ", "; ")
    header_image = data.get("header_image") or ""

    developers_raw = data.get("developers") or []
    if isinstance(developers_raw, str):
        developers_list = [developers_raw]
    elif isinstance(developers_raw, list):
        developers_list = [str(x) for x in developers_raw]
    else:
        developers_list = []
    developers = [d.strip() for d in developers_list if d and isinstance(d, str)]

    publishers_raw = data.get("publishers") or []
    if isinstance(publishers_raw, str):
        publishers_list = [publishers_raw]
    elif isinstance(publishers_raw, list):
        publishers_list = [str(x) for x in publishers_raw]
    else:
        publishers_list = []
    publishers = [p.strip() for p in publishers_list if p and isinstance(p, str)]

    platforms = data.get("platforms") or {}
    windows = bool(platforms.get("windows"))
    mac = bool(platforms.get("mac"))
    linux = bool(platforms.get("linux"))

    metacritic_data = data.get("metacritic") or {}
    metacritic_score = metacritic_data.get("score")
    if metacritic_score is not None:
        try:
            metacritic_score = int(metacritic_score)
        except (TypeError, ValueError):
            metacritic_score = None

    release_date_raw = data.get("release_date") or {}
    release_dt, coming_soon = _parse_release_date(release_date_raw if isinstance(release_date_raw, dict) else None)

    categories_raw = data.get("categories") or []
    categories: list[dict[str, Any]] = []
    if isinstance(categories_raw, list):
        for c in categories_raw:
            if not isinstance(c, dict):
                continue
            cid = c.get("id")
            cname = c.get("description") or ""
            try:
                cid_int = int(cid)
            except (TypeError, ValueError):
                continue
            categories.append({"id": cid_int, "name": cname})

    genres_raw = data.get("genres") or []
    genres: list[dict[str, Any]] = []
    if isinstance(genres_raw, list):
        for g in genres_raw:
            if not isinstance(g, dict):
                continue
            gid = g.get("id")
            gname = g.get("description") or g.get("description") or g.get("name") or ""
            try:
                gid_int = int(gid)
            except (TypeError, ValueError):
                continue
            genres.append({"id": gid_int, "name": gname})

    game_data: dict[str, Any] = {
        "id": appid,
        "name": name,
        "required_age": required_age,
        "is_free": is_free,
        "detailed_description": detailed_description,
        "about_the_game": about_the_game,
        "short_description": short_description,
        "supported_languages": supported_languages,
        "header_image": header_image,
        "developers": ", ".join(developers),
        "publishers": ", ".join(publishers),
        "windows": windows,
        "mac": mac,
        "linux": linux,
        "metacritic": metacritic_score,
        "release_date": release_dt,
        "coming_soon": coming_soon,
    }

    return StoreParsed(
        game_data=game_data,
        categories=categories,
        genres=genres,
        developers=developers,
        publishers=publishers,
    )


def parse_steamspy_appdetails(raw_json: Any) -> dict[str, Any] | None:
    """
    Parse SteamSpy appdetails response for a single app.

    Example (from https://steamspy.com/api.php?appdetails&appid=440):
    {
      "appid":440,
      "name":"Team Fortress 2",
      "owners":"50,000,000 .. 100,000,000",
      "average_forever":22316,
      "average_2weeks":517,
      "median_forever":4575,
      "median_2weeks":109,
      "ccu":43819,
      ...
    }
    """
    if not isinstance(raw_json, dict):
        return None

    owners_raw = str(raw_json.get("owners") or "").strip()
    owners_min = 0
    owners_max = 0
    if owners_raw:
        # format: "low .. high"
        parts = owners_raw.split("..")
        if len(parts) == 2:
            low_str = parts[0].replace(",", "").strip()
            high_str = parts[1].replace(",", "").strip()
            try:
                owners_min = int(low_str)
                owners_max = int(high_str)
            except ValueError:
                owners_min = 0
                owners_max = 0

    def _intval(key: str) -> int:
        try:
            return int(raw_json.get(key) or 0)
        except (TypeError, ValueError):
            return 0

    return {
        "owners_min": owners_min,
        "owners_max": owners_max,
        "average_forever": _intval("average_forever"),
        "average_2weeks": _intval("average_2weeks"),
        "median_forever": _intval("median_forever"),
        "median_2weeks": _intval("median_2weeks"),
        "ccu": _intval("ccu"),
    }


def parse_reviews_summary(raw_json: Any) -> dict[str, Any] | None:
    """
    Parse the Steam Store reviews summary endpoint response.

    Example shape (truncated):
    {
      "success": 1,
      "query_summary": {
        "num_reviews": 10,
        "review_score": 8,
        "review_score_desc": "Very Positive",
        "total_positive": 656669,
        "total_negative": 75279,
        "total_reviews": 731948,
        ...
      },
      "reviews": [...]
    }
    """
    if not isinstance(raw_json, dict):
        return None
    if not raw_json.get("success"):
        return None
    summary = raw_json.get("query_summary")
    if not isinstance(summary, dict):
        return None

    def _intval(key: str) -> int | None:
        value = summary.get(key)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    return {
        "reviewNumReviews": _intval("num_reviews"),
        "reviewScore": _intval("review_score"),
        "reviewScoreDesc": summary.get("review_score_desc"),
        "reviewTotalPositive": _intval("total_positive"),
        "reviewTotalNegative": _intval("total_negative"),
        "reviewTotalReviews": _intval("total_reviews"),
    }


def parse_reviews_list(appid: int, raw_json: Any) -> list[dict[str, Any]]:
    """
    Parse individual reviews from the reviews API response into a list of dicts
    compatible with the Prisma `Review` model.
    """
    if not isinstance(raw_json, dict):
        return []
    if not raw_json.get("success"):
        return []

    reviews_raw = raw_json.get("reviews") or []
    if not isinstance(reviews_raw, list):
        return []

    parsed: list[dict[str, Any]] = []
    for r in reviews_raw:
        if not isinstance(r, dict):
            continue
        author = r.get("author") or {}
        if not isinstance(author, dict):
            author = {}

        recommendationid = str(r.get("recommendationid") or "")
        if not recommendationid:
            continue

        def _intval(v: Any, default: int = 0) -> int:
            try:
                return int(v)
            except (TypeError, ValueError):
                return default

        def _floatval(v: Any, default: float = 0.0) -> float:
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        ts_created = _intval(r.get("timestamp_created"), 0)
        ts_updated = _intval(r.get("timestamp_updated"), 0)
        last_played = _intval(author.get("last_played"), 0)

        parsed.append(
            {
                "id": recommendationid,
                "gameId": appid,
                "authorSteamId": str(author.get("steamid") or ""),
                "authorPlaytimeForever": _intval(author.get("playtime_forever"), 0),
                "authorPlaytimeAtReview": _intval(author.get("playtime_at_review"), 0),
                "authorLastPlayed": datetime.fromtimestamp(last_played) if last_played > 0 else None,
                "language": r.get("language") or "",
                "review": r.get("review") or "",
                "votedUp": bool(r.get("voted_up")),
                "votesUp": _intval(r.get("votes_up"), 0),
                "votesFunny": _intval(r.get("votes_funny"), 0),
                "weightedVoteScore": _floatval(r.get("weighted_vote_score"), 0.0),
                "writtenDuringEarlyAccess": bool(r.get("written_during_early_access")),
                "timestampCreated": datetime.fromtimestamp(ts_created) if ts_created > 0 else None,
                "timestampUpdated": datetime.fromtimestamp(ts_updated) if ts_updated > 0 else None,
            }
        )

    return parsed

