"""Raw ELECTRA XGBoost experiment and earlier embedding comparisons."""
import json
from pathlib import Path
import streamlit as st

RUN_URL = "https://github.com/botsandbolts/Pricing/actions/workflows/electra-raw-xgboost.yml"


def render(root):
    root = Path(root)
    st.header("Experimental pricing · The extra brain cell")
    st.write(
        "Can a differently trained small transformer add information to MiniLM? "
        "We retrain XGBoost on matched folds, with and without all 256 raw "
        "ELECTRA coordinates, then refit the same Bayesian residual correction."
    )
    st.info(
        "XGBoost still uses vehicle facts and the MiniLM predicted text price. "
        "ELECTRA-Small is frozen: its token vectors are averaged with attention "
        "masking and normalized. All 256 coordinates enter XGBoost directly. "
        "The Bayesian hierarchy uses its existing predictors and is refitted on each arm residuals."
    )
    st.caption(
        "Reuses the 25 completed encoding jobs. No UMAP, PCA or dimension reduction. "
        "Explicit money amounts are removed from descriptions. MiniLM price heads "
        "are refitted inside each XGBoost training fold. ELECTRA pooling remains experimental."
    )
    st.link_button("View ELECTRA training progress", RUN_URL)
    path = root / "reports/electra-raw-xgboost-experiment/comparison.json"
    if not path.exists():
        st.info(
            "ELECTRA evaluation is pending. The workflow will publish its measured "
            "comparison here automatically. No ELECTRA accuracy gain has been established."
        )
    else:
        try:
            report = json.loads(path.read_text())
            rows = report["comparison"]
            arms = {row["Model"]: row for row in rows}
            baseline = float(arms["baseline"]["MAE ($)"])
            electra = float(arms["electra_raw_xgboost"]["MAE ($)"])
        except (OSError, ValueError, KeyError, TypeError):
            st.warning("The ELECTRA report could not be loaded. Check the workflow for details.")
        else:
            st.subheader("Measured historical results")
            st.dataframe(rows, hide_index=True, use_container_width=True)
            st.metric("ELECTRA change in mean absolute error", f"{electra - baseline:+,.2f} USD")
            st.caption("Negative means lower error with ELECTRA; positive means higher error.")
            if baseline > 0:
                st.write(f"Relative MAE change: {100 * (electra - baseline) / baseline:+.2f}%.")
            paired = report.get("paired_improvement")
            if paired:
                low, high = paired["group_bootstrap_95_usd"]
                st.write(f"Paired 95% confidence interval for MAE savings: {low:,.2f} to {high:,.2f} USD.")
                st.caption("Positive savings favour ELECTRA. An interval spanning zero leaves the improvement uncertain.")
            st.caption(report["limits"])
            with st.expander("Business and technical receipt"):
                st.json(report)
    for name, relative in [
        ("Earlier ELECTRA/UMAP in the Bayesian hierarchy", "reports/electra-hierarchy-experiment/comparison.json"),
        ("Earlier BGE/UMAP in the Bayesian hierarchy", "reports/hierarchy-umap-experiment/comparison.json"),
        ("Earlier BGE/UMAP in XGBoost", "reports/bge-umap-experiment/comparison.json"),
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
    st.write(
        "Use Price my car for the deployed calculator. Experimental weights "
        "remain separate until evaluation and inference checks finish."
    )
