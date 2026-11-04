"""Baseline pricing and archived embedding experiments."""
import json
from pathlib import Path
import streamlit as st


def render(root):
    root = Path(root)
    st.header("Experimental pricing")
    st.info(
        "Baseline retained: vehicle facts and the MiniLM predicted text price "
        "feed XGBoost AFT, followed by the Bayesian residual correction."
    )
    st.write(
        "The raw ELECTRA experiment was not adopted: its measured historical "
        "MAE improvement was about 0.20%, too small to justify the additional encoder."
    )
    st.markdown("### Experiment history")
    for name, relative in [
        ("Raw ELECTRA in XGBoost · not adopted", "reports/electra-raw-xgboost-experiment/comparison.json"),
        ("ELECTRA/UMAP in the Bayesian hierarchy", "reports/electra-hierarchy-experiment/comparison.json"),
        ("BGE/UMAP in the Bayesian hierarchy", "reports/hierarchy-umap-experiment/comparison.json"),
        ("BGE/UMAP in XGBoost", "reports/bge-umap-experiment/comparison.json"),
    ]:
        previous = root / relative
        if previous.exists():
            try:
                report = json.loads(previous.read_text())
                with st.expander(name):
                    st.dataframe(report["comparison"], hide_index=True)
                    st.caption(report["limits"])
            except (OSError, ValueError, KeyError, TypeError):
                st.caption(f"{name}: report unavailable.")
    st.markdown("### Price your car")
    st.write("Use Price my car for the baseline calculator.")
