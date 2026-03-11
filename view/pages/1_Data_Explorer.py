from __future__ import annotations

import pandas as pd
import streamlit as st


st.title("Data Explorer")
st.caption("Drop in a CSV and explore basic stats.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV to get started.")
else:
    df = pd.read_csv(uploaded)
    st.dataframe(df, use_container_width=True)
    st.subheader("Describe")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

