"""Read-only, evidence-first neural bench in the existing Model lab panel."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ARCH = {
    "resmlp": "Residual MLP · gated depth",
    "deepcross": "Deep & Cross · low-rank interactions",
    "fttransformer": "FT-Transformer · feature attention",
}
LOSS = {"huber": "Robust Huber",
        "quantile50": "Median · pinball τ=0.5",
        "aft": "Log-normal AFT · exact observations"}
COHORT = {"full": "All original training cars",
          "lemon": "Lemon Squeeze · 75% training retained"}


def render(root):
    root = Path(root)
    st.markdown("## 🧠 Neural pricing proving ground")
    st.caption("Experimental asking prices · no hierarchical correction · completely separate from the live calculator")
    report_path = root / "reports/neural-ask-benchmark/comparison.json"
    if not report_path.exists():
        st.info("The 18-arm neural benchmark is not yet published. The original XGBoost prices remain active; no neural accuracy claim is being made.")
        cols = st.columns(3)
        for col, (name, label) in zip(cols, ARCH.items()):
            with col:
                st.markdown("**" + label + "**")
                st.caption({"resmlp": "Deep residual blocks, layer normalization and controlled dropout.",
                            "deepcross": "Explicit cross terms plus nonlinear representation learning.",
                            "fttransformer": "Attention across structured, categorical and description features."}[name])
        st.write("Every architecture is tested using Huber, log-median quantile 0.5, and exact-observation log-normal AFT. Each is trained with all eligible training data and again with the frozen Lemon train-only exclusion.")
        st.caption("Canonical holdouts: 31,668 validation cars and 31,682 test cars. No seller descriptions, labels or test prices are used to fit text SVD.")
        st.markdown("[GitHub experiment workflow](https://github.com/botsandbolts/Pricing/actions/workflows/neural-ask-sweep.yml)")
        return
    try:
        report = json.loads(report_path.read_text())
        if report.get("status") != "complete" or len(report.get("results", [])) != 36:
            raise ValueError("Incomplete results")
        df = pd.DataFrame(report["results"])
    except (OSError, ValueError, TypeError, KeyError):
        st.warning("A complete verified neural benchmark is not available.")
        return
    winner = report["winner_by_original_validation"]
    validation = df.loc[df.partition.eq("validation")].sort_values("validation_rank").copy()
    test = df.loc[df.partition.eq("test")].copy()
    joined = validation.merge(test, on=["architecture", "loss", "train_cohort"],
                              suffixes=("_val", "_test"), validate="one_to_one")
    c1, c2, c3 = st.columns(3)
    c1.metric("Experimental candidates", f"{report['tested_arms']}")
    c2.metric("Best original validation MAE", f"USD {winner['validation_mae_usd']:,.0f}")
    best_test = joined.loc[joined.validation_rank_val.eq(1)].iloc[0]
    c3.metric("Selected model · original test MAE", f"USD {best_test.mae_usd_test:,.0f}")
    st.caption("Champion selected by ORIGINAL validation MAE only. Test results are displayed for auditing, not used to pick the champion.")

    display = pd.DataFrame({
        "Validation rank": joined.validation_rank_val.astype(int),
        "Architecture": joined.architecture.map(ARCH),
        "Objective": joined.loss.map(LOSS),
        "Training cohort": joined.train_cohort.map(COHORT),
        "Validation MAE ($)": joined.mae_usd_val.round(0),
        "Test MAE ($)": joined.mae_usd_test.round(0),
        "Test RMSE ($)": joined.rmse_usd_test.round(0),
        "Test bias ($)": joined.mean_bias_usd_test.round(0),
        "Δ test MAE vs XGB Lemon ($)": joined.delta_mae_vs_xgboost_usd_test.round(0),
        "Neural win fraction": joined.fraction_listings_beating_xgboost_test.round(3),
    })
    st.dataframe(display.sort_values("Validation rank"), hide_index=True, use_container_width=True)
    st.caption("Δ = neural MAE minus existing Lemon XGBoost AFT MAE on the exact same original test cars. Negative is better.")

    with st.expander("📈 Show loss × architecture × outlier-policy comparisons", expanded=True):
        bars = joined[["architecture", "loss", "train_cohort",
                       "mae_usd_val", "mae_usd_test"]].copy()
        bars["candidate"] = bars.architecture.map(ARCH) + " · " + bars.loss.map(LOSS)
        bars["training"] = bars.train_cohort.map(COHORT)
        chart = bars.pivot_table(index="candidate", columns="training",
                                 values="mae_usd_val", aggfunc="first")
        st.markdown("**Validation MAE (USD), full vs Lemon training**")
        st.bar_chart(chart)
        st.caption("These bars are from the heldout validation set, not training error.")
    with st.expander("🧪 Exact XGBoost comparators and statistical uncertainty"):
        ref = report["frozen_xgboost_lemon_reference"]
        full = report["published_full_xgboost_reference"]
        base_rows = [
            {"XGBoost": "Lemon AFT · matched per row",
             "Validation MAE ($)": ref["validation"]["mae_usd"],
             "Test MAE ($)": ref["test"]["mae_usd"]},
            {"XGBoost": "Full-data AFT · published aggregate",
             "Validation MAE ($)": full["validation"]["mae_usd"],
             "Test MAE ($)": full["test"]["mae_usd"]},
        ]
        st.dataframe(pd.DataFrame(base_rows), hide_index=True)
        st.caption("The frozen Lemon comparator has matching car IDs and asking-price labels. The full-data published XGBoost reference has matching split sizes but lacks a verified row-level artifact; its comparison is descriptive, not paired.")
        for _, r in joined.sort_values("validation_rank_val").head(3).iterrows():
            st.write(f"**{ARCH[r.architecture]} · {LOSS[r.loss]} · {COHORT[r.train_cohort]}**")
            interval = r.paired_delta_95pct_bootstrap_usd_test
            st.caption(f"Paired test MAE difference: USD {r.delta_mae_vs_xgboost_usd_test:,.0f}; 95% listing bootstrap CI USD {interval[0]:,.0f} to USD {interval[1]:,.0f}.")
    with st.expander("Methodology, limitations and machine-readable scores"):
        st.markdown("**Training-only controls:** numerical robust scaling, categorical vocabularies, TF-IDF/SVD description projection, dropout, weight decay, residual layers or feature attention, gradient clipping, input noise, cosine learning-rate scheduling and temporal train-only early stopping.")
        st.write("AFT here is log-normal likelihood on **exact observed asking prices**. The data contain no verified censoring intervals; none are invented. The published results exclude all hierarchical/Bayesian corrections.")
        st.write("All original validation/test rows remain intact for both outlier policies. The lemon comparison uses the archived XGBoost training membership, not a fresh test-informed squeeze.")
        st.caption("The full-data benchmark is not a price on today's market. Network checkpoints are retained in GitHub Actions artifacts, not installed as live pricing models.")
        st.download_button("Download 18-arm benchmark JSON", json.dumps(report, indent=2),
                           "neural-asking-comparison.json", "application/json",
                           key="neural_ask_json")
