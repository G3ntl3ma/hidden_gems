from __future__ import annotations

from pathlib import Path

import streamlit as st

from _bootstrap import ensure_project_root

PROJECT_ROOT = ensure_project_root(Path(__file__))

from api.config import get_settings
from api.db import db_session
from view.shared import render_csv_preview


st.set_page_config(page_title="Hidden Gems ML Dashboard", layout="wide")

st.title("Hidden Gems — ML Dashboard")
st.caption("Streamlit dashboard wired for Prisma + SQLite.")

settings = get_settings()
st.sidebar.header("Config")
st.sidebar.code(f"DATABASE_URL={settings.database_url}")

st.subheader("Database connectivity")
try:
    with db_session() as db:
        st.success("Connected to database via Prisma.")
        st.write(
            {
                "client": "Prisma(sync)",
                "note": (
                    "Add models in prisma/schema.prisma, then run "
                    "`npx prisma@5.17.0 db push --schema prisma/schema.prisma` "
                    "and `python -m prisma generate --schema prisma/schema.prisma`."
                ),
            }
        )
except Exception as e:
    st.error(
        "Could not connect. Did you run Prisma db push and Prisma generate "
        "after schema changes?"
    )
    st.exception(e)

st.subheader("Quick data preview (CSV)")
render_csv_preview(label="Upload a CSV to preview")

