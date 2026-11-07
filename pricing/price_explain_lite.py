"""Small honest explanations for the *already loaded* Car Price Studio models.

NO third-party XAI runtime, Captum, duplicate encoders, subprocesses, 64
coalitions or eager what-if inference. Ask reuses the frozen BGE embedding
already prepared for the saved price; Bid reuses the same cached DeepFM/XGB.
Native single-row XGBoost TreeSHAP is offered only on explicit click.
"""
import hashlib
import json
import math


def money(x):
    return "$"+format(float(x),",.0f")


def delta(x):
    x=float(x)
    return ("+" if x>=0 else "−")+money(abs(x))


def _id(saved):
    data={
        "fields":saved.get("fields"),"bid_fields":saved.get("bid_fields"),
        "notes":saved.get("notes"),"date":saved.get("date"),
        "results":{s:{"price":saved[s].get("price"),
                        "model":saved[s].get("model")}
                   for s in ("ask","bid") if s in saved}}
    return hashlib.sha256(json.dumps(data,sort_keys=True,default=str).encode()).hexdigest()


def explain_clock(receipt,actual_source_price):
    """Exact historical anchor -> age -> index accounting; never pretend SHAP."""
    base=float(receipt["anchor_price_reference_usd"])
    age=float(receipt["age_factor"])
    market=float(receipt["market_factor"])
    target=float(receipt["price"])
    original=float(actual_source_price)
    if not all(map(math.isfinite,(base,age,market,target,original))):
        raise ValueError("Historical receipt contains invalid factors.")
    if min(base,age,market,target,original)<=0:
        raise ValueError("Historical receipt contains nonpositive factors.")
    if abs(base-original)>max(1.,original*.0001):
        raise ValueError("Date-adjustment anchor does not match the served model.")
    age_effect=base*(age-1.)
    market_effect=target-base-age_effect
    if abs(base*age*market-target)>max(1.,target*.0001):
        raise ValueError("Date-adjustment bridge does not reconcile.")
    return dict(anchor=base,age_effect=age_effect,
                market_effect=market_effect,selected=target)


def _bid_ensemble(receipt):
    neural=float(receipt["deepfm_price"])
    tree=float(receipt["xgboost_lemon_price"])
    price=float(receipt["price"])
    alpha=float(receipt.get("neural_weight",.75))
    if not 0<=alpha<=1 or not all(map(math.isfinite,(neural,tree,price))):
        raise ValueError("Invalid model component values")
    if abs(alpha*neural+(1-alpha)*tree-price)>max(.5,price*.0001):
        raise ValueError("Wholesale ensemble does not reconcile")
    return dict(neural=neural,tree=tree,weight=alpha,combined=price)


def _change(fields,choice,market):
    new=dict(fields)
    if choice=="10,000 fewer miles":
        new["odometer"]=max(0.,float(fields["odometer"])-10000.)
    elif choice=="10,000 more miles":
        new["odometer"]=min(2_000_000.,float(fields["odometer"])+10000.)
    elif choice=="Compare a clean title":
        new["title_status"]="clean"
    elif choice=="Compare a rebuilt title":
        new["title_status"]="rebuilt"
    elif choice=="Compare pre-sale grade 4" and market=="bid":
        new["auction_grade"]="4"
    else:
        raise ValueError("Unsupported one-feature scenario")
    return new


def one_scenario(root,saved,market,choice,ask_champion=False,bid_champion=False):
    """Exactly ONE native pricing pass on the existing cached model."""
    if market not in saved:
        raise ValueError("The selected model did not produce a verified price.")
    source=float(saved[market]["price"])
    if not math.isfinite(source) or source<=0:
        raise ValueError("Invalid original model price.")
    fields=dict(saved["fields"] if market=="ask" else saved["bid_fields"])
    whatif=_change(fields,choice,market)
    if market=="ask":
        if ask_champion:
            from pricing.neural_champion_live import load_ask
            model=load_ask(root)
        else:
            from pricing.neural_ask_embedding_live import load
            model=load(root)
        prepared=saved.get("ask_prepared")
        if prepared is None:
            # Only for receipts created before this UI release. The pinned
            # globally cached BGE encoder is reused; it is never instantiated
            # again solely for explanation.
            prepared=model.prepare_text(saved.get("notes",""))
        new=float(model.predict_prepared(whatif,prepared)["price"])
    else:
        if bid_champion:
            from pricing.neural_champion_live import load_bid
            model=load_bid(root)
        else:
            from pricing.neural_bid_blend_live import load
            model=load(root)
        new=float(model.predict(whatif)["price"])
    if not math.isfinite(new) or not 100<=new<=1e6:
        raise ValueError("The model could not evaluate that comparison.")
    return {
        "market":market,"reference_usd":source,"comparison_usd":new,
        "difference_usd":new-source,"change":choice,
        "interpretation":"One-variable model sensitivity, not a causal resale effect",
        "model":saved[market].get("model","native model"),
    }


def native_bid_tree_drivers(root,saved,*,bid_champion=False):
    """Native XGBoost TreeSHAP on ONE row; no shap pip package or new model.

    Reconciles with the already displayed XGBoost component before showing
    anything. These are log-price contributions of the tree BRANCH only.
    """
    import numpy as np
    from pricing.neural_bid_blend_live import CATS,_norm
    from pricing.hierarchy_sparse_live import design_matrix
    if "bid_fields" not in saved or "bid" not in saved:
        raise ValueError("Original auction-feature receipt is unavailable.")
    if bid_champion:
        from pricing.neural_champion_live import load_bid
        model=load_bid(root)
    else:
        from pricing.neural_bid_blend_live import load
        model=load(root)
    original=getattr(model,"original",model)
    fields=saved["bid_fields"]
    _,_,raw,_=original.features(fields)
    pack={**{k:_norm(fields.get(k)) for k in CATS[:8]},
          "odometer":float(fields["odometer"]),
          "auction_grade":float(raw[-1]) if np.isfinite(raw[-1]) else None}
    booster=original.xgb
    row,_=design_matrix(booster,original.xgb_schema,None,pack,float(raw[1]))
    parts=np.asarray(booster.predict(row,pred_contribs=True),dtype=float)
    margin=float(booster.predict(row,output_margin=True)[0])
    if parts.ndim!=2 or parts.shape[0]!=1 or parts.shape[1]<2:
        raise ValueError("Wrong native feature-contribution dimensions.")
    if not math.isfinite(margin) or abs(float(parts[0].sum())-margin)>.002:
        raise ValueError("Tree contributions do not reconcile.")
    target=float(math.exp(min(13.815510557964274,max(4.605170186,margin))))
    expected=float(saved["bid"]["xgboost_lemon_price"])
    if abs(target-expected)>max(.50,.0001*expected):
        raise ValueError("TreeSHAP was not computed on the saved model price.")
    names=row.feature_names or [f"Feature {i+1}" for i in range(parts.shape[1]-1)]
    values=parts[0,:-1]
    if len(names)!=len(values) or not np.isfinite(values).all():
        raise ValueError("Tree feature names or contributions are invalid.")
    indices=np.argsort(np.abs(values))[::-1][:7]
    result=[{"Feature":str(names[int(i)]),
             "Direction":"Raises tree estimate" if values[i]>0 else
                         "Lowers tree estimate" if values[i]<0 else "Neutral",
             "Effect (log price)":round(float(values[i]),4)}
            for i in indices]
    return {"tree_price_usd":target,"log_reconciliation":float(parts[0].sum())-margin,
            "top_drivers":result,"scope":"Native Lemon XGBoost branch, not DeepFM or full Bid"}


def render(root,saved,adjusted,*,ask_champion=False,bid_champion=False):
    import streamlit as st
    if "ask" not in saved and "bid" not in saved:return
    token=_id(saved)
    st.markdown("""
    <div style="background:linear-gradient(115deg,#193b54,#265b66);
    border-radius:17px;padding:18px 22px;color:#fff;margin:18px 0 9px">
      <div style="letter-spacing:.12em;font-size:.75rem;font-weight:700;color:#a9eeda">
        CAR PRICE STUDIO · EXPLAINABLE AI · MADE IN GERMANY
      </div>
      <div style="font-size:1.55rem;font-weight:800;color:white;margin:7px 0">
        🔎 Why this price?
      </div>
      <div style="line-height:1.5;color:#e8f3f8">
        The model has opinions. Here are the receipts, not a 500 MB PowerPoint.
      </div>
    </div>
    """,unsafe_allow_html=True)
    basic,clock,whatif,science=st.tabs([
        "💡 The story","📅 Price time machine",
        "🧪 Change one detail","🧠 Under the hood"])
    with basic:
        st.markdown("**🏷 Seller Ask:** an estimated advertised asking price. "
                    "BGE reads the text and the full neural model combines it "
                    "with vehicle facts. It is not a completed retail sale.")
        st.markdown("**🔨 Wholesale Bid:** an estimated completed historical "
                    "auction-sale amount, not a current dealer offer.")
        if "bid" in saved:
            try:
                r=_bid_ensemble(saved["bid"])
                st.markdown("**The Bid model has two expert opinions**")
                x,y=st.columns(2)
                x.metric("🧠 DeepFM · "+str(round(100*r["weight"]))+"% weight",
                         money(r["neural"]))
                y.metric("🌳 Lemon XGBoost · "+
                         str(round(100*(1-r["weight"])))+"% weight",
                         money(r["tree"]))
                st.caption("Weighted together: "+money(r["combined"])+
                           ". Two different algorithms, same auction question. "
                           "The dealership's coffee machine was not consulted.")
            except (ValueError,TypeError,KeyError):
                st.caption("The active Bid model has no reconciled "
                           "component breakdown available.")
        st.info("Ask minus Bid is not dealer profit: repairs, risks, "
                "transport, taxes, fees and different markets all matter.")
    with clock:
        if not adjusted:
            st.info("Historical-to-selected-date adjustments are not available.")
        for side,name in (("ask","Seller Ask"),("bid","Wholesale Bid")):
            if side not in adjusted or side not in saved:continue
            try:bridge=explain_clock(adjusted[side],saved[side]["price"])
            except (KeyError,TypeError,ValueError):
                st.warning(name+" date effects could not be reconciled.");continue
            st.markdown("**"+name+" · exact arithmetic**")
            a,b,c=st.columns(3)
            a.metric("Original model",money(bridge["anchor"]))
            b.metric("Age effect",delta(bridge["age_effect"]))
            c.metric("Market effect",delta(bridge["market_effect"]))
            st.caption("Result on chosen date: "+money(bridge["selected"])+
                       ". Original + age + market = displayed estimate.")
        st.caption("The historical clock uses an aggregate market index and "
                   "vehicle-age curves. It is not an individual 2026 sale or "
                   "a model-feature SHAP explanation.")
    with whatif:
        choices=["10,000 fewer miles","10,000 more miles",
                 "Compare a clean title","Compare a rebuilt title"]
        sides=[s for s in ("ask","bid") if s in saved and
               (s=="ask" or "bid_fields" in saved)]
        if not sides:
            st.info("Calculate a verified price first.")
        else:
            with st.form("lite_inprocess_whatif",clear_on_submit=False):
                side=st.radio("Which price?",sides,horizontal=True,
                    format_func=lambda s:"🏷 Seller Ask" if s=="ask" else "🔨 Wholesale Bid")
                opts=choices+(
                    ["Compare pre-sale grade 4"] if side=="bid" else [])
                chosen=st.selectbox("Try one hypothetical change",opts)
                do=st.form_submit_button("⚡ Ask the actual model",
                    type="primary",use_container_width=True)
            if do:
                try:
                    with st.spinner("One quick inference from the existing cached model…"):
                        r=one_scenario(root,saved,side,chosen,
                            ask_champion=ask_champion,bid_champion=bid_champion)
                    st.session_state["cps_lite_whatif"]={"id":token,"result":r}
                except (ValueError,TypeError,OSError,RuntimeError,KeyError,
                        MemoryError) as exc:
                    st.session_state.pop("cps_lite_whatif",None)
                    st.warning("This optional model comparison is unavailable. "
                               "Your existing valuation is unchanged.")
            previous=st.session_state.get("cps_lite_whatif")
            if previous and previous.get("id")==token:
                r=previous["result"]
                st.metric("What the model predicts for the comparison car",
                          money(r["comparison_usd"]),delta(r["difference_usd"]))
                st.caption(r["change"]+" · original historical price "+
                           money(r["reference_usd"])+".")
                st.warning("This is the model's response to changing an input. "
                           "It does NOT prove you could change the actual sale price, "
                           "vehicle title, odometer history or physical condition.")
    with science:
        st.markdown("**No additional XAI model is loaded.** We reuse the "
                    "same globally cached neural and tree models that produced "
                    "your price. No Captum, 64-scenario sweeps, neural retraining, "
                    "subprocesses, loading/unloading tricks or extra BGE encoding "
                    "for new estimates.")
        st.markdown("**Accuracy context:** CVaR95 is the average dollar error "
                    "in the worst 5% of historical predictions. MAE and RMSE "
                    "show different aspects of those mistakes. "
                    "None guarantees accuracy for today's vehicle market.")
        st.markdown("**Model explanations are not causal claims.** The date "
                    "decomposition is exact accounting; the what-if is one "
                    "real-model sensitivity; the wholesale breakdown is the "
                    "actual weighted ensemble.")
        if "bid" in saved and "bid_fields" in saved:
            with st.expander("🌳 Native one-row XGBoost feature contributions"):
                st.caption("A precise local tree explanation with XGBoost's "
                           "built-in TreeSHAP. No extra package or network call. "
                           "These numbers are in **log-price units**, and explain "
                           "only the tree's share of the Bid estimate.")
                if st.button("Inspect the existing XGBoost model",key="lite_tree_go"):
                    try:
                        with st.spinner("Explaining one row of the cached tree…"):
                            data=native_bid_tree_drivers(root,saved,
                                                       bid_champion=bid_champion)
                        st.session_state["cps_lite_tree"]={"id":token,"data":data}
                    except (OSError,ValueError,TypeError,KeyError,RuntimeError,
                            MemoryError):
                        st.session_state.pop("cps_lite_tree",None)
                        st.warning("Native tree contributions could not be "
                                   "verified for this auction estimate.")
                report=st.session_state.get("cps_lite_tree")
                if report and report.get("id")==token:
                    result=report["data"]
                    st.metric("Original XGBoost component",
                              money(result["tree_price_usd"]))
                    st.table(result["top_drivers"])
                    st.caption("The signed contributions add up to the native "
                               "log-price prediction with bias; these top "
                               "factors are not dollar discounts, causation, "
                               "or an explanation of the DeepFM component.")
        st.caption("Open source. No registration. On-demand predictions; "
                   "no additional external explanation service or tracking.")
