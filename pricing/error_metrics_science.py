"""Evidence-based error metric literacy and read-only RMSE-first benchmark cards.

No inference, model fitting, or external network requests. Papers are linked
directly by persistent DOI. Values are read only from verified local reports.
"""
import json
from pathlib import Path

ACADEMIC_PAPERS = (
    {
        "authors": "Chai & Draxler (2014)",
        "title": "Root mean square error (RMSE) or mean absolute error (MAE)? – Arguments against avoiding RMSE in the literature",
        "journal": "Geoscientific Model Development 7, 1247–1250",
        "doi": "https://doi.org/10.5194/gmd-7-1247-2014",
        "message": "RMSE adds information about error variability and is valuable when larger mistakes matter; authors recommend a combination of accuracy measures.",
    },
    {
        "authors": "Hodson (2022)",
        "title": "Root-mean-square error (RMSE) or mean absolute error (MAE): when to use them or not",
        "journal": "Geoscientific Model Development 15, 5481–5487",
        "doi": "https://doi.org/10.5194/gmd-15-5481-2022",
        "message": "Neither metric is universally superior: RMSE aligns with Gaussian-error assumptions, MAE with Laplacian errors; other error distributions may favor other metrics.",
    },
    {
        "authors": "Willmott & Matsuura (2005)",
        "title": "Advantages of the mean absolute error (MAE) over the root mean square error (RMSE) in assessing average model performance",
        "journal": "Climate Research 30, 79–82",
        "doi": "https://doi.org/10.3354/cr030079",
        "message": "Useful counterpoint: RMSE should not be described as the typical or average absolute error. MAE is more directly interpretable as average absolute dollar miss.",
    },
)


def complete_reports(root):
    """Ignore absent or non-complete benchmarks; never promise a result."""
    root=Path(root)
    records=[]
    for study,label in (
        ("ft29","FT-Transformer, 29 configurations"),
        ("fm10","FM, FFM and kernel approximations, 10 configurations"),
        ("fm26","FM family full vs Lemon, 26 configurations"),
    ):
        file=root/"reports"/"bid-rmse"/study/"comparison.json"
        if not file.is_file():
            continue
        try:
            report=json.loads(file.read_text())
            if (report.get("status")!="complete" or report.get("study")!=study or
                report.get("primary_selection_metric")!="original_holdout_validation_RMSE_USD" or
                report.get("trained_objectives_unchanged") is not True or
                report.get("original_test_used_only_after_val_selection") is not True):
                continue
            winner=report["standalone_selected_on_validation"]
            blend=report["blend_selected_on_validation"]
            reference=report["matched_lemon_xgboost"]["test"]
            records.append({
                "Study":label,
                "Selected candidate":winner["trial"],
                "Original test RMSE · $":round(winner["test"]["rmse_usd"],2),
                "Original test MAE · $":round(winner["test"]["mae_usd"],2),
                "Test |error| P99 · $":round(winner["test"]["absolute_error_p99_usd"],2),
                "Validation-selected blend test RMSE · $":round(blend["test"]["rmse_usd"],2),
                "Matched Lemon XGBoost test RMSE · $":round(reference["rmse_usd"],2),
            })
        except (ValueError,TypeError,KeyError,OSError):
            continue
    return records


def render(root=".", *, compact=False):
    import streamlit as st

    if compact:
        with st.expander("📚 Why RMSE? Big mistakes, MAE, and scientific evidence"):
            _body(root,show_reports=False)
    else:
        st.markdown("## 🎯 When a $20,000 pricing miss matters more than twenty $500 misses")
        st.caption("Evidence-based evaluation · dollar RMSE first · original holdouts · training objectives unchanged")
        _body(root,show_reports=True)


def _body(root,*,show_reports):
    import streamlit as st

    left,right=st.columns(2)
    with left:
        st.markdown("**RMSE — spotlight on large errors**")
        st.latex(r"\mathrm{RMSE}_\$\ =\ \sqrt{\frac{1}{n}\sum_{i=1}^{n}(\hat P_i-P_i)^2}")
        st.write("Squaring dollar errors gives larger misses more influence. "
                 "That helps us compare models when a badly wrong wholesale estimate "
                 "could be financially costly.")
    with right:
        st.markdown("**MAE — the average absolute miss**")
        st.latex(r"\mathrm{MAE}_\$\ =\ \frac{1}{n}\sum_{i=1}^{n}|\hat P_i-P_i|")
        st.write("MAE stays useful: it answers how many dollars a prediction "
                 "misses by on average, without disproportionately weighting the largest errors.")
    st.info("The model-training losses are **unchanged**: robust Huber/log-price, AFT, "
            "and quantile training where previously used. RMSE is evaluated on "
            "original VIN-disjoint validation and test predictions in **real dollar units**, "
            "not optimized as a new neural or FM loss.")
    st.markdown("**Why not use RMSE alone?**")
    st.write("A few unusually expensive or misrecorded vehicles can dominate RMSE. "
             "It does not impose a maximum-error guarantee and may hide systematic "
             "overvaluation or poor performance on ordinary cars. We therefore also "
             "report MAE, signed bias, the 95th/99th/99.9th percentile absolute errors, "
             "and the share of predictions missing by more than $5k, $10k or $20k. "
             "In a time-based model comparison, training data and holdout populations "
             "must be distinguished; filtered test observations produce optimistic results.")
    st.markdown("**What the peer-reviewed literature says**")
    for paper in ACADEMIC_PAPERS:
        st.markdown(
            f"- **[{paper['authors']} · {paper['title']}]({paper['doi']})**  \n"
            f"  {paper['journal']}. {paper['message']}")
    st.caption("Research is about properties of evaluation statistics, not proof that "
               "RMSE is always the economically correct objective for an auction. "
               "Error costs and the price distribution determine the business choice.")

    if show_reports:
        st.markdown("### Read-only original-auction RMSE comparison")
        reports=complete_reports(root)
        if reports:
            st.dataframe(reports,hide_index=True,use_container_width=True)
            st.caption("The original test was scored only after the model or blend was "
                       "preselected by validation RMSE. Standalone and blend test "
                       "results have different configurations; they must not be used "
                       "to silently reselect a champion. All values come from saved "
                       "original-row evaluations, not live-user valuations.")
        else:
            st.caption("RMSE-ranked bid experiment reports have not been published yet. "
                       "Existing production pricing models have not changed.")
