"""Curate Steam app IDs as gem / not-gem for model training (stored in SQLite)."""

from __future__ import annotations

import csv
import io
import importlib.util
from pathlib import Path

import pandas as pd
import streamlit as st

_BOOTSTRAP_PATH = Path(__file__).resolve().parents[1] / "_bootstrap.py"
_BOOTSTRAP_SPEC = importlib.util.spec_from_file_location("view_bootstrap", _BOOTSTRAP_PATH)
if _BOOTSTRAP_SPEC is None or _BOOTSTRAP_SPEC.loader is None:
    raise ImportError(f"Could not load bootstrap helper from {_BOOTSTRAP_PATH}")
_BOOTSTRAP_MODULE = importlib.util.module_from_spec(_BOOTSTRAP_SPEC)
_BOOTSTRAP_SPEC.loader.exec_module(_BOOTSTRAP_MODULE)
PROJECT_ROOT = _BOOTSTRAP_MODULE.ensure_project_root(Path(__file__))

from api.db import db_session

st.title("Training label curator")
st.caption(
    "Record whether each Steam app ID is a hidden gem or not. Labels are stored in "
    "the Prisma database (`CuratedSteamLabel`). Download CSV for backups and save it "
    "as `data/local/curated_steam_labels.csv` for training scripts."
)


def _load_labels_with_names() -> pd.DataFrame:
    with db_session() as db:
        rows = db.curatedsteamlabel.find_many(order={"appId": "asc"})
        if not rows:
            return pd.DataFrame(columns=["appid", "is_gem", "updated_at", "name_in_db"])

        ids = [r.appId for r in rows]
        games = db.game.find_many(where={"id": {"in": ids}})
        id_to_name = {g.id: g.name for g in games}

        records = []
        for r in rows:
            records.append(
                {
                    "appid": r.appId,
                    "is_gem": r.isGem,
                    "updated_at": r.updatedAt,
                    "name_in_db": id_to_name.get(r.appId, ""),
                }
            )
        return pd.DataFrame(records)


def _labels_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["appid", "is_gem"])
    for _, row in df.iterrows():
        w.writerow([int(row["appid"]), bool(row["is_gem"])])
    return buf.getvalue().encode("utf-8")


st.subheader("Add or update a label")
col_a, col_b, col_c = st.columns([1, 1, 1])
with col_a:
    app_id = st.number_input("Steam app ID", min_value=1, value=1, step=1, format="%d")
with col_b:
    label_choice = st.radio(
        "Label",
        options=["Gem", "Not a gem"],
        horizontal=True,
    )
with col_c:
    st.write("")
    st.write("")
    save = st.button("Save", type="primary")

if save:
    is_gem = label_choice == "Gem"
    try:
        with db_session() as db:
            db.curatedsteamlabel.upsert(
                where={"appId": int(app_id)},
                data={
                    "create": {"appId": int(app_id), "isGem": is_gem},
                    "update": {"isGem": is_gem},
                },
            )
            game = db.game.find_unique(where={"id": int(app_id)})
        st.success(f"Saved app {int(app_id)} as {'gem' if is_gem else 'not a gem'}.")
        if game is not None:
            st.info(f"In database as: **{game.name}**")
        else:
            st.info("This app ID is not in the `Game` table yet (ok for pre-scrape labels).")
    except Exception as e:
        st.error("Could not save label. Is the database configured?")
        st.exception(e)

st.subheader("Existing labels")
try:
    table_df = _load_labels_with_names()
except Exception as e:
    st.error("Could not load labels from the database.")
    st.exception(e)
    st.stop()

if table_df.empty:
    st.info("No labels yet. Add one above.")
else:
    display_df = table_df.copy()
    if "updated_at" in display_df.columns:
        display_df["updated_at"] = display_df["updated_at"].astype(str)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    exp = table_df[["appid", "is_gem"]].copy()
    st.download_button(
        label="Download labels as CSV",
        data=_labels_to_csv_bytes(exp),
        file_name="curated_steam_labels.csv",
        mime="text/csv",
        help=(
            "Columns: appid, is_gem. Save as data/local/curated_steam_labels.csv "
            "for `python -m scripts.build_training_dataset`."
        ),
    )

st.subheader("Delete a label")
del_id = st.number_input("App ID to remove", min_value=1, value=1, step=1, format="%d", key="del_app")
if st.button("Delete label for this app ID"):
    try:
        with db_session() as db:
            n = db.curatedsteamlabel.delete_many(where={"appId": int(del_id)})
        if n == 0:
            st.warning(f"No label found for app ID {int(del_id)}.")
        else:
            st.success(f"Removed label for app ID {int(del_id)}.")
    except Exception as e:
        st.error("Could not delete label.")
        st.exception(e)
