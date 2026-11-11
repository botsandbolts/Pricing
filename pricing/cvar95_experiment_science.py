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
    """Only eligible original, unfiltered historical validation cohorts.

    Never present a retrospectively outcome-selected subset's performance
    as prospective full-original-population validation accuracy.
    """
    root=Path(root)
    if market=="bid":
        from pricing.us_bid_resmlp_live import available
        if available(root):
            m=json.loads((root/"models/us-bid-resmlp-20261111/manifest.json").read_text())
            return m["original_validation"]
        from pricing.ray_bid_winner_live import available as ray_available
        if ray_available(root):
            m=json.loads((root/"models/ray-bid-9am-winner/manifest.json").read_text())
            return m["original_validation"]
    elif market=="ask":
        from pricing.granite_aft_ask_live import manifest
        m=manifest(root)
        if m and m.get("validation",{}).get("rows")==31668:
            return m["validation"]
    else:
        raise ValueError("Unknown historical car price market")
    return None


def render(root):
    import streamlit as st
    st.markdown("### 🛡️ CVaR95 · the expensive end of being wrong")
    st.write("Most estimates are close; the costly mistakes are the ones "
             "that can change a decision. CVaR95 measures the mean absolute "
             "dollar error in the worst five percent of held-out cars.")
    st.latex(r"\mathrm{CVaR}_{95}(|e|)=\operatorname{mean}(\text{largest }5\%\text{ of }|\widehat P-P|)")
    st.caption("This is an empirical historical validation diagnostic, "
               "not an uncertainty interval, purchase offer, or guarantee.")
    st.markdown("**Research foundations**")
    for author,title,journal,url,meaning in PAPERS:
        st.markdown("- **["+author+" · "+title+"]("+url+")** — "+journal+". "+meaning)
    st.markdown("#### ✅ Original, unfiltered validation comparisons")
    a,b=st.columns(2)
    ask=model_kpis(root,"ask")
    bid=model_kpis(root,"bid")
    with a:
        st.markdown("**✍ Seller Ask · 2021 advertised prices**")
        if ask:
            st.metric("CVaR95",_number(ask["CVaR95_absolute_USD"]))
            st.caption(f'Original {ask["rows"]:,} validation listings, '
                       'all included. Granite DCNv2 AFT uses the recorded year, '
                       'odometer, categories and frozen text embeddings.')
            st.caption("MAE "+_number(ask["MAE_USD"])+
                       " · RMSE "+_number(ask["RMSE_USD"]))
        else:
            st.info("No independently matched whole-cohort Ask validation "
                    "receipt is currently installed.")
    with b:
        st.markdown("**🔨 Wholesale Bid · 2014–15 completed auctions**")
        if bid:
            st.metric("CVaR95",_number(bid["CVaR95_absolute_USD"]))
            st.caption(f'Original {bid["rows"]:,} validation auctions, '
                       '2015-indexed USD; different label and cohort from Ask.')
            st.caption("MAE "+_number(bid["MAE_USD"])+
                       " · RMSE "+_number(bid["RMSE_USD"]))
        else:
            st.info("No independently matched original Bid validation receipt.")
    st.warning("The two targets are not directly comparable: asking ads are "
               "not completed sales, and an historical auction-price model "
               "does not verify the value of a car in November 2026.")
