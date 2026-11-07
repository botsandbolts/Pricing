"""RAM-light Explainable AI panel for the new native Ask/Bid main calculator.

No Captum, SHAP, transformers, numpy, pandas or plotly imports here.
Actual models run in the same short-lived memory-isolated inference workers
as pricing. No new model weights, no full-cohort allocations, no model copies.
"""
import hashlib
import json


def _money(x):
    return "${:,.0f}".format(float(x))


def _signed(x):
    return ("+" if float(x)>=0 else "−")+_money(abs(float(x)))


def _token(saved):
    payload=[
        saved.get("fields"),saved.get("bid_fields"),saved.get("notes",""),
        saved.get("date"),[
            (s,saved[s].get("price"),saved[s].get("model"))
            for s in ("ask","bid") if s in saved]]
    return hashlib.sha256(json.dumps(payload,default=str,sort_keys=True).encode()).hexdigest()


def _clock(saved,adjusted):
    from pricing.lightweight_price_xai_ui import dollar_bridge
    import streamlit as st
    st.markdown("**From historical research to your selected date**")
    for market,label in (("ask","🏷 Seller Ask"),("bid","🔨 Wholesale Bid")):
        if market not in adjusted:continue
        try:
            parts=dollar_bridge(adjusted[market])
            historical=float(saved[market]["price"])
            if abs(parts["historical_anchor_usd"]-historical)>max(1.,historical*.001):
                raise ValueError("Anchor and served model differ")
        except (ValueError,TypeError,KeyError) as exc:
            st.warning(label+": an exact date bridge isn't available.");continue
        with st.container(border=True):
            st.markdown("**"+label+"**")
            a,b,c=st.columns(3)
            a.metric("Historical model",_money(parts["historical_anchor_usd"]))
            b.metric("Vehicle ageing",_signed(parts["vehicle_age_effect_usd"]))
            c.metric("Market index",_signed(parts["market_index_effect_usd"]))
            st.caption("Equals "+_money(parts["selected_date_usd"])+
                       " on your selected date. This is arithmetic accounting, "
                       "not a model-feature attribution.")
    st.caption("Historical validation is not evidence of accuracy on November "
               "2026 transactions. Retail CPI is a proxy for wholesale transport.")


def _bid_voices(saved):
    import streamlit as st
    if "bid" not in saved:return
    r=saved["bid"]
    if not {"deepfm_price","xgboost_lemon_price"}<=set(r):
        st.caption("The active Bid model did not provide separate price components.")
        return
    a=float(r.get("neural_weight",.75))
    deep=float(r["deepfm_price"])
    tree=float(r["xgboost_lemon_price"])
    combined=a*deep+(1-a)*tree
    published=float(r["price"])
    if not (.0<=a<=1.) or abs(combined-published)>max(.5,published*.0001):
        st.warning("The neural/tree blend receipt did not reconcile. "
                   "Individual model components will not be presented.");return
    st.markdown("**Two different models, one wholesale number**")
    x,y=st.columns(2)
    x.metric("🧠 DeepFM · "+str(round(a*100))+"% weight",_money(deep))
    y.metric("🌳 Lemon XGBoost · "+str(round((1-a)*100))+"% weight",_money(tree))
    st.caption(str(round(a*100))+"% × DeepFM + "+
               str(round((1-a)*100))+"% × Lemon XGBoost = "+
               _money(published)+". These models learn different patterns; "
               "their disagreement is worth inspecting, not ignoring.")


def _scenarios(root,saved,market,token):
    import streamlit as st
    from pricing.neural_isolated_serving import run,IsolatedModelError
    label="Seller Ask" if market=="ask" else "Wholesale Bid"
    st.markdown("#### "+("🏷 " if market=="ask" else "🔨 ")+label)
    st.caption("A few changed car facts, evaluated with the SAME native "
               "neural weights. No surrogate model or text-generation service.")
    key="lite_neural_xai_"+market
    if st.button("🔍 Explain "+label+" with 3 real scenarios",
                 key=key+"_go",type="secondary",use_container_width=True):
        try:
            inputs=saved["fields"] if market=="ask" else saved["bid_fields"]
            with st.spinner("Running one bounded, isolated model check…"):
                data=run(root,market,inputs,saved.get("notes","") if market=="ask" else "",
                         operation="sensitivity",expected=saved[market]["price"])
            st.session_state[key]={"token":token,"data":data}
        except (ValueError,IsolatedModelError,OSError,RuntimeError) as exc:
            st.session_state.pop(key,None)
            st.warning("This optional explanation is unavailable: "+str(exc))
    cache=st.session_state.get(key)
    if cache and cache["token"]==token:
        data=cache["data"]
        st.metric("Historical reference prediction",_money(data["baseline_usd"]))
        for example in data["changes"]:
            st.metric(example["scenario"],
                      _money(example["historical_price_usd"]),
                      delta=_signed(example["difference_usd"]))
        st.caption("These are **model sensitivities**, NOT causal market effects, "
                   "offers or proof that changing a title or odometer changes the "
                   "actual value. Never misrepresent vehicle facts.")


def render(root,saved,adjusted):
    import streamlit as st
    if "ask" not in saved and "bid" not in saved:return
    token=_token(saved)
    st.markdown("### 🔎 Why this price? · The lightweight evidence garage")
    st.caption("Two heavyweight pricing brains. One featherweight explanation. "
               "The initial appraisal triggers NO additional explanation inference.")
    overview,scenarios,science=st.tabs([
        "💡 Plain English","🧪 Try a few changes","⚙️ The science"])
    with overview:
        st.write("**Ask** estimates the number a seller might put in an advert. "
                 "**Bid** estimates a different market: completed wholesale "
                 "auction sales, not a live dealer offer.")
        _bid_voices(saved)
        with st.expander("📅 Show exactly how the date changed the dollars"):
            _clock(saved,adjusted)
        st.caption("A price gap is not dealership profit. Taxes, transport, "
                   "conditioning and market selection are missing from that simple subtraction.")
    with scenarios:
        for market in ("ask","bid"):
            if market in saved and (market=="ask" or "bid_fields" in saved):
                _scenarios(root,saved,market,token)
    with science:
        st.markdown("**Ask:** the pinned frozen BGE language encoder reads "
                    "the description; a complete neural pricing network "
                    "combines its embedding with vehicle facts.")
        st.markdown("**Bid:** DeepFM learns combinations of auction features. "
                    "Lemon XGBoost offers an independent tree-based price "
                    "estimate. The displayed blend comes from the two real predictions.")
        st.markdown("**Explainability:** we show the original Bid ensemble parts, "
                    "exact date arithmetic and up to three one-at-a-time "
                    "full-model sensitivity tests. The tests **do not add up** "
                    "to an attribution waterfall, and they aren't causal.")
        st.caption("Memory budget: zero new dependencies, no Captum, no SHAP "
                   "reference matrices, no extra BGE instance, and each requested "
                   "explanation runs in a short-lived isolated process.")
        st.markdown("[Scientific background and limitations]"
                    "(https://github.com/botsandbolts/Pricing/blob/main/docs/EXPLAINABLE_AI_PRICE_GARAGE.md)")
