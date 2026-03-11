from __future__ import annotations

import pandas as pd
import streamlit as st

from api.config import get_settings
from api.db import db_session


st.set_page_config(page_title="Hidden Gems ML Dashboard", layout="wide")

st.title("Hidden Gems — ML Dashboard")
st.caption("Starter Streamlit app wired for Prisma + SQLite.")

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
                "note": "Add models in prisma/schema.prisma then run `prisma db push`.",
            }
        )
except Exception as e:
    st.error("Could not connect. Did you run Prisma generation / set DATABASE_URL?")
    st.exception(e)

st.subheader("Quick data preview (CSV)")
uploaded = st.file_uploader("Upload a CSV to preview", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.dataframe(df, use_container_width=True)

