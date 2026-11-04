"""Controlled BGE comparison, separate from the deployed pricing calculator."""
import json
from pathlib import Path

import streamlit as st

RUN_URL = "https://github.com/botsandbolts/Pricing/actions/runs/41876358303"


def render(root):
    st.header("Experimental pricing · The extra brain cell")
    st.write(
        "Can description embeddings improve our asking-price estimates? "
        "This experiment compares the same Lemon Squeeze cohort and XGBoost AFT "
        "pipeline, with and without 32 BGE/PCA features in XGBoost."
    )
    st.info(
        "BGE/PCA enters XGBoost only. The Bayesian hierarchy keeps its existing "
        "structure and predictors; each arm refits that same hierarchy to its "
        "own out-of-fold residuals. Wholesale pricing uses the existing model."
    )
    st.caption(
        "Descriptions are encoded in 20 jobs. PCA is fitted separately inside "
        "each training fold. Explicit money amounts are removed from the text."
    )
    st.link_button("View training progress on GitHub", RUN_URL)
    path = Path(root) / "reports/bge-experiment/comparison.json"
    if not path.exists():
        st.info(
            "Evaluation is pending. The completed workflow will publish its "
            "comparison here automatically. No accuracy gain has been established."
        )
    else:
        try:
            report = json.loads(path.read_text())
            rows = report["comparison"]
            if not isinstance(rows, list) or not rows:
                raise ValueError("Missing comparison rows")
        except (OSError, ValueError, KeyError, TypeError):
            st.warning("The comparison report could not be loaded. Check the workflow for details.")
        else:
            st.subheader("Measured historical results")
            st.dataframe(rows, hide_index=True, use_container_width=True)
            arms = {row.get("Model"): row for row in rows}
            baseline, bge = arms.get("baseline"), arms.get("bge")
            if baseline is not None and bge is not None:
                base_mae = float(baseline["MAE ($)"])
                bge_mae = float(bge["MAE ($)"])
                saved = base_mae - bge_mae
                st.metric("BGE change in mean absolute error", f"{bge_mae - base_mae:+,.2f} USD")
                st.caption(
                    "Negative means lower error with BGE; positive means higher error. "
                    "This difference alone does not establish statistical significance."
                )
                if base_mae > 0:
                    st.write(f"BGE reduces historical MAE by {100 * saved / base_mae:+.2f}% relative to the paired baseline.")
            st.caption(report.get("limits", "Historical research comparison."))
            with st.expander("Business and technical receipt"):
                st.json(report)
    st.markdown("### Price your car")
    st.write(
        "Use Price my car for the deployed calculator. Experimental weights "
        "remain separate until training, evaluation and inference checks finish."
    )
