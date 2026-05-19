from __future__ import annotations

import importlib.util
from pathlib import Path

import streamlit as st

_BOOTSTRAP_PATH = Path(__file__).resolve().parents[1] / "_bootstrap.py"
_BOOTSTRAP_SPEC = importlib.util.spec_from_file_location("view_bootstrap", _BOOTSTRAP_PATH)
if _BOOTSTRAP_SPEC is None or _BOOTSTRAP_SPEC.loader is None:
    raise ImportError(f"Could not load bootstrap helper from {_BOOTSTRAP_PATH}")
_BOOTSTRAP_MODULE = importlib.util.module_from_spec(_BOOTSTRAP_SPEC)
_BOOTSTRAP_SPEC.loader.exec_module(_BOOTSTRAP_MODULE)
PROJECT_ROOT = _BOOTSTRAP_MODULE.ensure_project_root(Path(__file__))

from view.shared import render_csv_preview


st.title("Data Explorer")
st.caption("Drop in a CSV and explore basic stats.")

df = render_csv_preview(label="Upload CSV", empty_message="Upload a CSV to get started.")
if df is not None:
    st.subheader("Describe")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

