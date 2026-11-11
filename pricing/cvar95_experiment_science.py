"""Research and business guide to CVaR95 of absolute USD car-pricing error.

The top-line loss is the mean of the largest 5% ABSOLUTE ORIGINAL USD
prediction errors, NOT a dealer's realized financial P&L and not a model
training objective. MAE/RMSE remain secondary diagnostics.
"""
import json
from pathlib import Path

PAPERS=(
    ("Rockafellar & Uryasev (2000)",
     "Optimization of Conditional Value-at-Risk",
     "The Journal of Risk, 2(3), 21–41",
     "https://doi.org/10.21314/JOR.2000.038",
     "Introduces CVaR as an optimization/evaluation framework for tail risk."),
    ("Acerbi & Tasche (2002)",
     "On the coherence of expected shortfall",
     "Journal of Banking & Finance, 26(7), 1487–1503",
     "https://doi.org/10.1016/S0378-4266(02)00283-2",
     "Explains why the average of the worst losses is often more informative than a tail threshold alone; discrete samples need careful definitions."),
    ("Chai & Draxler (2014)",
     "Root mean square error (RMSE) or mean absolute error (MAE)? – Arguments against avoiding RMSE in the literature",
     "Geoscientific Model Development, 7, 1247–1250",
     "https://doi.org/10.5194/gmd-7-1247-2014",
     "Discusses RMSE versus MAE and supports keeping complementary error metrics."),
    ("Hodson (2022)",
     "Root-mean-square error (RMSE) or mean absolute error (MAE): when to use them or not",
     "Geoscientific Model Development, 15, 5481–5487",
     "https://doi.org/10.5194/gmd-15-5481-2022",
     "Highlights that metric choice depends on the error distribution and the intended decision."),
)

def evidence(root="."):
    """Only completely verified historical benchmark rows; no invented forecasts."""
    path=Path(root)/"reports/cvar95/previously-selected/comparison.json"
    if not path.is_file():return {}
    try:
        d=json.loads(path.read_text())
        if (d.get("status")!="complete" or
            d.get("metric")!="CVaR95 absolute dollar error" or
            d.get("original_holdout_sizes")!={"wholesale":83603,"asking":31682} or
            d.get("test_used_for_reselection") is not False or
            d.get("no_retraining") is not True):
            return {}
        return d
    except (ValueError,KeyError,TypeError,OSError):
        return {}


def all_validation(root,market):
    """CVaR-selected models may differ from previously MAE/RMSE-selected ones."""
    if market not in ("bid","ask"):
        raise ValueError("Unknown wholesale/asking market")
    path=Path(root)/"reports/cvar95/all-candidates"/market/"comparison.json"
    if not path.is_file():return {}
    try:
        data=json.loads(path.read_text())
        if (data.get("status")=="complete" and
            data.get("market")==market and
            data.get("primary_metric")=="validation_absolute_dollar_CVaR95" and
            data.get("training_objectives_unchanged") is True and
            data.get("models_chosen_by_test") is False):
            return data
    except (OSError,TypeError,ValueError,KeyError):
        pass
    return {}


def _number(n):
    return "$"+format(float(n),",.0f")


def model_kpis(root,market):
    """Live matched receipt: filtered Lemon Ask validation or legacy test."""
    from pricing.ray_lemon75_ask_live import manifest as lemon_manifest
    if market=="ask":
        clean=lemon_manifest(root)
        if clean:
            return clean["filtered_validation"]
    from pricing.overnight_bohb_live import available as bohb_available,FOLDERS
    if bohb_available(root,market):
        m=json.loads((Path(root)/FOLDERS[market]/"manifest.json").read_text())
        return m["historical_test"]
    selected=all_validation(root,market)
    if market=="bid" and selected:
        choice=selected.get("chosen_blend",{})
        if (choice.get("model"),choice.get("neural_weight"),
            choice.get("xgb_lemon_weight"))==("deepfm16_full",.75,.25):
            v=choice.get("original_test",{})
            if v.get("CVaR95_absolute_USD",0)>0:
                return v
    if market=="ask" and selected:
        choice=selected.get("CVaR95_selection",{})
        if (choice.get("model")=="bge-256" and
            choice.get("test_status")=="scored_after_validation_selection"):
            v=choice.get("original_test") or {}
            if v.get("CVaR95_absolute_USD",0)>0:
                return v
    # An older test-score champion must never silently substitute for the
    # independently CVaR-selected checkpoint when that selection is missing.
    return None


def render(root):
    import streamlit as st
    st.markdown("### 🛡️ Why CVaR95? The expensive end of being wrong")
    st.write("A car valuation isn't only about being right **on average**. "
             r"A rare \$20,000 miss can be a much more memorable business meeting "
             r"than twenty \$200 misses. Nobody wants an unexpectedly collectible lemon.")
    st.latex(r"\mathrm{CVaR}_{95}(|e|)=\operatorname{mean}\left(\mathrm{largest\ 5\%}\ \left|\widehat{P}_i-P_i\right|\right)")
    st.caption("The input is absolute model error in historical US dollars. "
               "It is **not** a 95% prediction interval, worst-case bound, "
               "a guarantee, or an estimate of dealer profit-and-loss.")

    a,b,c=st.columns(3)
    a.metric("CVaR95 · PRIMARY","Worst 5% mean")
    b.metric("RMSE · secondary","Big errors squared")
    c.metric("MAE · secondary","Average $ miss")
    st.markdown("**For the garage / dealership**")
    st.write("CVaR95 asks: *When our prediction is among the five percent most "
             "wrong, how much are we wrong by on average?* "
             "This supports inventory risk screening, escalation rules, "
             "human appraisals and comparing models that occasionally make "
             "very costly mistakes.")
    st.markdown("**For the scientist**")
    st.write("We calculate the empirical expected shortfall of **absolute USD "
             "residuals** by averaging the largest ceil(0.05 × N) observations. "
             "Original chronological or VIN-disjoint validation determines "
             "selection; the untouched test estimates generalization. "
             "We report the VaR95 threshold, RMSE and MAE alongside CVaR95.")
    st.warning("High-priced cars, mislabeled records, inflation, market shifts "
               "and outliers can dominate the tail. CVaR95 of **absolute price "
               "error** is not automatically the real dollar loss when a dealer "
               "buys, sells or finances a vehicle. Business decisions should "
               "also check overpricing versus underpricing and vehicle segments.")
    st.markdown("**Research foundations**")
    for author,title,journal,url,meaning in PAPERS:
        st.markdown("- **["+author+" · "+title+"]("+url+")** — "+journal+". "+meaning)
    st.markdown("**The regulatory parallel (not a claim of regulatory compliance)**")
    st.write("The Basel Committee's market-risk framework uses expected "
             "shortfall because tail outcomes matter. Our pricing-error "
             "application borrows the statistical idea, not its capital rules.")
    st.markdown("[Basel Committee, Minimum capital requirements for market risk (2019)](https://www.bis.org/publications/201901-standards-minimum-capital-requirements-market-risk)")
    st.divider()
    st.markdown("#### 🧪 What was actually evaluated?")
    d=evidence(root)
    if not d:
        st.info("The verified original CVaR95 archive is not installed.")
        return
    a,b=st.columns(2)
    ask=model_kpis(root,"ask")
    bid=model_kpis(root,"bid")
    with a:
        if ask:
            from pricing.ray_lemon75_ask_live import manifest as lemon_manifest
            clean=lemon_manifest(root)
            label="2021 asking · Lemon75 VALIDATION" if clean else "Earlier Ask TEST"
            st.metric(label,_number(ask["CVaR95_absolute_USD"]))
            st.caption(
                "23,751 outcome-filtered historical validation listings; "
                "conditional, optimistic for new vehicles."
                if clean else
                "Older original historical test, not a clean Lemon75 audit.")
    with b:
        if bid:st.metric("2015 wholesale · CVaR-selected DeepFM16 blend",_number(bid["CVaR95_absolute_USD"]))
        st.caption("83,603 completed historical auctions, 2015-indexed USD.")
    st.caption("Cross-market CVaR values are not comparable because vehicle prices, "
               "targets, dates and cohorts differ. Previously selected models "
               "were chosen on MAE/RMSE, not the new CVaR95 objective.")
    candidates={k:all_validation(root,k) for k in ("bid","ask")}
    if any(candidates.values()):
        st.markdown("#### Overnight validation-CVaR model audit")
        for market,report in candidates.items():
            if not report:continue
            if market=="bid":
                champ=report["chosen_standalone"];blend=report["chosen_blend"]
                st.write("**Wholesale:** "+champ["model"]+" selected by validation "
                    "CVaR95; best validation blend: "+blend["model"]+
                    " (neural/FM weight "+str(blend["neural_weight"])+").")
            else:
                choice=report["CVaR95_selection"]
                st.write("**Asking:** "+choice["model"]+
                    " selected by validation CVaR95 ("+choice["test_status"]+").")
        st.caption("These are validation-based research decisions, not automatic "
                   "authorization to swap or deploy unverified serving weights.")
