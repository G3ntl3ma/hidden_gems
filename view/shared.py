from __future__ import annotations

import pandas as pd
import streamlit as st


def render_csv_preview(
    *,
    label: str,
    empty_message: str | None = None,
) -> pd.DataFrame | None:
    uploaded = st.file_uploader(label, type=["csv"])
    if uploaded is None:
        if empty_message:
            st.info(empty_message)
        return None

    df = pd.read_csv(uploaded)
    st.dataframe(df, use_container_width=True)
    return df

