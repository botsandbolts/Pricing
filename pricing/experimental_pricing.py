"""Controlled BGE comparison, separate from the deployed pricing calculator."""
import json
from pathlib import Path

import streamlit as st

RUN_URL = "https://github.com/botsandbolts/Pricing/actions/runs/41899042018"


def render(root):
    st.header("Experimental pricing · The extra brain cell")
    st.write(
        "Can description embeddings improve our asking-price estimates? "
        "This experiment compares the same Lemon Squeeze cohort and XGBoost AFT "
        "predictions, with and without 32 BGE/UMAP features in the Bayesian residual model."
    )
    st.info(
        "XGBoost uses vehicle facts and the MiniLM predicted text price in both arms. "
        "BGE/UMAP enters only the Bayesian residual model, as 32 scaled coordinates "
        "with shrinkage. Both arms use identical saved XGBoost predictions and residuals."
    )
    st.caption(
        "Reuses the 20 completed encoding batches. Cosine UMAP and scaling are fitted "
        "on calibration descriptions only; validation and test are transformed only. "
        "Explicit money amounts are removed from the text."
    )
    st.link_button("View training progress on GitHub", RUN_URL)
    path = Path(root) / "reports/hierarchy-umap-experiment/comparison.json"
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
            baseline, bge = arms.get("baseline"), arms.get("hierarchy_umap")
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
            paired = report.get("paired_improvement")
            if paired:
                low, high = paired["group_bootstrap_95_usd"]
                st.write(f"Paired 95% confidence interval for MAE savings: {low:,.2f} to {high:,.2f} USD.")
                st.caption("Positive savings favour UMAP in the hierarchy. An interval spanning zero leaves the improvement uncertain.")
            previous = report.get("previous_xgboost_umap")
            if previous:
                with st.expander("Earlier UMAP-in-XGBoost comparison"):
                    st.dataframe(previous["comparison"], hide_index=True)
                    st.caption(previous["limits"])
            st.caption(report.get("limits", "Historical research comparison."))
            with st.expander("Business and technical receipt"):
                st.json(report)
    st.markdown("### Price your car")
    st.write(
        "Use Price my car for the deployed calculator. Experimental weights "
        "remain separate until training, evaluation and inference checks finish."
    )
