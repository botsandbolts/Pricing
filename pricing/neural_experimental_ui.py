"""Experimental model workshop: two independent parity-gated neural engines.

Ask: description -> pinned BGE embedding -> entire DeepCross/AFT price model.
Bid: full-data DeepFM16 (75%) + matched Lemon XGBoost (25%).
Research models are never substituted by the primary calculator's predictions.
"""
import json
import time
from pathlib import Path
from html import escape
import numpy as np
import pandas as pd
import streamlit as st
from pricing.neural_ask_embedding_live import available as ask_available
from pricing.neural_bid_blend_live import available as bid_available
from pricing.neural_champion_live import (
    ask_available as champion_ask_available,
    bid_available as champion_bid_available,
    load_ask as load_champion_ask,
    load_bid as load_champion_bid,
)

CSS="""
<style>
.exp-hero{background:radial-gradient(circle at 80% 10%,#6172b4,transparent 39%),
linear-gradient(120deg,#102039,#294067 55%,#17443e);color:white;
border:1px solid #a5aaca;border-radius:24px;
padding:clamp(24px,4vw,46px);margin:10px 0 24px}
.exp-hero h1{color:white!important;font-size:clamp(2.1rem,4.3vw,3.5rem);
letter-spacing:-.055em;line-height:1.07;margin:8px 0 15px}
.exp-hero p{color:#e2ebf5;max-width:720px;line-height:1.6}
.exp-hero small{color:#b9efd9;letter-spacing:.13em;font-weight:800}
.exp-chip{display:inline-block;color:white;background:#ffffff1d;
border-radius:99px;border:1px solid #e0f0ff77;padding:.3rem .65rem;
margin:4px;font-size:.8rem}
.exp-result{background:linear-gradient(125deg,#142944,#38466f);color:white;
border-radius:20px;padding:22px;border:1px solid #9aadd1;margin:12px 0}
.exp-result strong{display:block;color:white;font-size:clamp(2rem,4vw,3.4rem);
letter-spacing:-.05em}
.exp-result small{color:#b5efd5;letter-spacing:.1em}
.exp-result p{color:#e9edf7;margin:0}
</style>
"""


def _report(root,name):
    try:return json.loads((Path(root)/name).read_text())
    except (OSError,ValueError,TypeError):return {}


def _vehicle(prefix,bid=False,ready=False,compact=False):
    """One unified, keyboard-accessible form. Main pricing has no scan/BART step."""
    with st.form(prefix+"_vehicle",clear_on_submit=False):
        st.markdown("### 🚗 02 / Tell us about the car" if compact
                    else "#### 01 / Describe the vehicle")
        if compact:
            st.caption("No VIN upload, no special instructions and no degree in "
                       "automotive engineering required. The description is optional "
                       "for Bid but useful for Ask.")
        a,b,c=st.columns(3)
        year=a.number_input("Model year",1900,2021,2015,key=prefix+"_year")
        make=b.text_input("Make","Toyota",key=prefix+"_make")
        model=c.text_input("Model / trim","Corolla",key=prefix+"_model")
        a,b,c=st.columns(3)
        miles=a.number_input("Mileage (miles)",0,2000000,95000,
                             step=1000,key=prefix+"_mileage")
        state=b.text_input("US state code","ca",max_chars=2,
                           key=prefix+"_state")
        body=c.selectbox("Body style",[
              "sedan","suv","truck","pickup","hatchback","coupe",
              "wagon","van","convertible","other","unknown"],
              key=prefix+"_body")
        if compact:
            st.caption("Three useful facts; everything else is recorded as "
                       "'unknown', not guessed.")
            a,b,d=st.columns(3)
            condition=a.selectbox("Condition",[
                "unknown","good","excellent","fair","like new","new","salvage"],
                key=prefix+"_condition")
            title=b.selectbox("Title status",[
                "unknown","clean","rebuilt","salvage","lien","missing","parts only"],
                key=prefix+"_title")
            transmission=d.selectbox("Transmission",[
                "unknown","automatic","manual","other"],
                key=prefix+"_transmission")
            fuel=st.session_state.get(prefix+"_fuel","unknown")
            drive=st.session_state.get(prefix+"_drive","unknown")
            cylinders=st.session_state.get(prefix+"_cylinders","unknown")
            paint=st.session_state.get(prefix+"_paint","unknown")
        else:
            with st.expander("Optional condition, engine and ownership facts"):
                a,b,d=st.columns(3)
                condition=a.selectbox("Condition",[
                    "good","excellent","fair","like new","new","salvage","unknown"],
                    key=prefix+"_condition")
                fuel=b.selectbox("Fuel",[
                    "gas","diesel","hybrid","electric","other","unknown"],
                    key=prefix+"_fuel")
                transmission=d.selectbox("Transmission",[
                    "automatic","manual","other","unknown"],
                    key=prefix+"_transmission")
                a,b,d=st.columns(3)
                drive=a.selectbox("Drivetrain",[
                    "fwd","rwd","4wd","unknown"],key=prefix+"_drive")
                title=b.selectbox("Title",[
                    "clean","rebuilt","salvage","lien","missing","parts only","unknown"],
                    key=prefix+"_title")
                cylinders=d.selectbox("Cylinders",[
                    "4 cylinders","6 cylinders","8 cylinders",
                    "3 cylinders","5 cylinders","unknown"],
                    key=prefix+"_cylinders")
                paint=a.selectbox("Paint",[
                    "silver","white","black","grey","blue","red","green",
                    "brown","other","unknown"],key=prefix+"_paint")
        fields={"year":int(year),"manufacturer":make,"model":model,
                "odometer":float(miles),"state":state,"type":body,
                "condition":condition,"fuel":fuel,
                "transmission":transmission,"drive":drive,
                "title_status":title,"cylinders":cylinders,
                "paint_color":paint}
        if bid:
            st.markdown("#### Historical wholesale auction")
            a,b=st.columns(2)
            auction=a.date_input("Auction reference date (2014–2015)",
                  pd.Timestamp("2015-01-15").date(),
                  min_value=pd.Timestamp("2014-01-01").date(),
                  max_value=pd.Timestamp("2015-12-31").date(),
                  key=prefix+"_auction_date")
            grade=b.selectbox("Pre-sale auction grade",[
                  "unknown","1","2","3","4","5"],
                  index=3,key=prefix+"_grade")
            fields["auction_date"]=str(auction)
            fields["auction_grade"]=grade
            notes=""
        else:
            st.markdown("### ✍ 03 / What does the advert say?" if compact
                        else "#### 02 / The car's entire story")
            notes=st.text_area("Seller's description",
                "Well maintained. Clean title. New tires.",
                max_chars=8000,height=125 if compact else 140,
                key=prefix+"_description",
                help="The original BGE text encoder interprets the advert. "
                     "A full neural network combines it with the vehicle facts.")
        submitted=st.form_submit_button(
            "🚗 Calculate seller Ask + wholesale Bid" if compact else
            "🚀 Run complete "+("wholesale" if bid else "asking-price neural")+" model",
            type="primary",disabled=not ready,use_container_width=True)
    return fields,notes,submitted

def _price(label,price,subtitle):
    if not np.isfinite(float(price)) or float(price)<=0:
        raise ValueError("Invalid price; cannot display")
    st.markdown('<section class="exp-result" aria-label="'+escape(label)+'">'
                '<small>'+escape(label.upper())+'</small><strong>$'+f"{price:,.0f}"+
                '</strong><p>'+escape(subtitle)+'</p></section>',
                unsafe_allow_html=True)


def _ask(root):
    st.markdown("### ✍️ Asking prices · BGE → DeepCross → AFT")
    st.caption("No MiniLM Ridge, no text-only price head and no TF-IDF/SVD price scalar. "
               "All 384 BGE embedding dimensions feed the learned neural projection.")
    from pricing.cvar95_experiment_science import model_kpis
    from pricing.cvar95_experiment_science import all_validation
    kpi=model_kpis(root,"ask") if all_validation(root,"ask") else None
    if kpi:
        a,c,e=st.columns(3)
        a.metric("🛡️ CVaR95 · PRIMARY", "$"+f"{kpi['CVaR95_absolute_USD']:,.0f}")
        c.metric("RMSE · secondary","$"+f"{kpi['RMSE_USD']:,.0f}")
        e.metric("MAE · secondary","$"+f"{kpi['MAE_USD']:,.0f}")
        st.caption("Worst 5% average absolute error across 31,682 original 2021 Craigslist test listings. The CVaR result describes a previously validated model; a CVaR-first re-ranking is published separately.")
    else:
        st.info("No verified CVaR95 test archive is currently installed.")
    ready=ask_available(root)
    if not ready:
        st.warning("Full BGE/DeepCross serving has not passed its original "
                   "holdout-parity gate. No individual price will be fabricated.")
        st.markdown("[Inspect verification workflow](https://github.com/botsandbolts/Pricing/actions/workflows/experimental-two-model-release.yml)")
    fields,notes,go=_vehicle("bge_ask",ready=ready)
    if go:
        start=time.monotonic()
        try:
            from pricing.neural_ask_embedding_live import load
            with st.spinner("Encoding the full description and running DeepCross/AFT…"):
                model=load(root)
                result=model.predict(notes,fields)
            st.session_state["bge_complete_receipt"]={
                "fields":fields,"notes":notes,"result":result,
                "elapsed":time.monotonic()-start}
        except (OSError,RuntimeError,ValueError,TypeError,KeyError) as exc:
            st.session_state.pop("bge_complete_receipt",None)
            st.error("Unable to verify the full asking prediction: "+str(exc))
    saved=st.session_state.get("bge_complete_receipt") if ready else None
    if saved:
        st.markdown("#### 03 / Price from the entire neural network")
        _price("Historical seller asking · May 2021 USD",
               saved["result"]["price"],
               "Not a completed sale, wholesale bid or current market appraisal")
        st.caption("Entire BGE + DeepCross/AFT completed in "+
                   f"{saved['elapsed']:.2f} seconds. Prices are historical anchors.")
        with st.expander("✍ Rewrite the listing · compare two FULL network prices",expanded=True):
            st.write("Vehicle details stay fixed. BGE re-embeds the new wording; "
                     "the complete DeepCross/AFT network predicts the new price. "
                     "This does **not** display any separate Ridge text-price estimate.")
            alt=st.text_area("Alternative wording",value=saved["notes"],
                             height=125,max_chars=8000,key="bge_whatif")
            if st.button("Compare complete neural estimates",type="primary",key="bge_whatif_run"):
                try:
                    from pricing.neural_ask_embedding_live import load
                    new=load(root).predict(alt,saved["fields"])["price"]
                    original=saved["result"]["price"]
                    a,b,c=st.columns(3)
                    a.metric("Original full-network","$"+f"{original:,.0f}")
                    b.metric("Rewritten full-network","$"+f"{new:,.0f}")
                    c.metric("Difference","$"+f"{new-original:+,.0f}")
                    st.caption("Model response to language, not causal proof that "
                               "editing an advertisement changes a transaction.")
                except (OSError,RuntimeError,ValueError,TypeError,KeyError) as exc:
                    st.warning("Whole-model wording comparison unavailable: "+str(exc))
        with st.expander("📉 Mileage sensitivity · full-network reruns"):
            try:
                from pricing.neural_ask_embedding_live import load
                model=load(root);prepared=model.prepare_text(saved["notes"]);table=[]
                for change in (-20000,-10000,0,10000,20000):
                    variant={**saved["fields"],
                             "odometer":max(0,saved["fields"]["odometer"]+change)}
                    price=model.predict_prepared(variant,prepared)["price"]
                    table.append({"Mileage":int(variant["odometer"]),"Asking USD":round(price)})
                st.line_chart(pd.DataFrame(table).set_index("Mileage")["Asking USD"])
            except (OSError,RuntimeError,ValueError,TypeError,KeyError) as exc:
                st.info("Mileage comparison temporarily unavailable: "+str(exc))
        public={"model":"BGE256 + entire DeepCross AFT",
                "historical_price_usd":saved["result"]["price"],
                "vehicle":saved["fields"],"description":"omitted for privacy"}
        st.download_button("Download asking research receipt",
               json.dumps(public,indent=2),file_name="bge-experimental-ask.json",
               mime="application/json",key="bge_download")


def _bid(root):
    st.markdown("### 🚘 Wholesale · DeepFM + Lemon XGBoost")
    st.caption("Historical completed 2014–15 wholesale auction sales, expressed "
               "in January 2015-indexed dollars. No current-year price guarantee.")
    from pricing.cvar95_experiment_science import model_kpis,evidence
    from pricing.cvar95_experiment_science import all_validation
    kpi=model_kpis(root,"bid") if all_validation(root,"bid") else None
    d=evidence(root)
    baseline=next((x for x in d.get("bid",[]) if x.get("model")=="Lemon XGBoost historical baseline"),None)
    if kpi:
        a,b,c=st.columns(3)
        a.metric("🛡️ CVaR95 · PRIMARY","$"+f"{kpi['CVaR95_absolute_USD']:,.0f}")
        b.metric("RMSE · secondary","$"+f"{kpi['RMSE_USD']:,.0f}")
        c.metric("MAE · secondary","$"+f"{kpi['MAE_USD']:,.0f}")
        if baseline:
            st.caption("Worst 5% average absolute miss reduced by "+
                       f"{(1-kpi['CVaR95_absolute_USD']/baseline['CVaR95_absolute_USD'])*100:.1f}%"+
                       " versus matched Lemon XGBoost on 83,603 2015-indexed-dollar auctions.")
        st.caption("The 75/25 blend was originally selected by validation RMSE; overnight CVaR-first selection is independent.")
    else:
        st.info("No verified CVaR95 original wholesale holdout archive is installed.")
    ready=bid_available(root)
    if not ready:
        st.warning("The wholesale DeepFM/XGBoost release has not passed "
                   "original test prediction parity. It cannot issue an experimental price.")
        st.markdown("[Inspect bid serving verification](https://github.com/botsandbolts/Pricing/actions/workflows/experimental-two-model-release.yml)")
    fields,notes,go=_vehicle("deepfm_bid",bid=True,ready=ready)
    if go:
        try:
            from pricing.neural_bid_blend_live import load
            with st.spinner("Evaluating DeepFM16 and matched Lemon XGBoost…"):
                result=load(root).predict(fields)
            st.session_state["deepfm_complete_receipt"]={"fields":fields,"result":result}
        except (OSError,RuntimeError,ValueError,TypeError,KeyError) as exc:
            st.session_state.pop("deepfm_complete_receipt",None)
            st.error("Unable to reproduce historical wholesale valuation: "+str(exc))
    saved=st.session_state.get("deepfm_complete_receipt") if ready else None
    if saved:
        r=saved["result"]
        _price("Historical wholesale sale · 2015-indexed USD",
               r["price"],"Not a dealer bid, guaranteed sale, or November 2026 appraisal")
        a,b=st.columns(2)
        a.metric("DeepFM16 · 75%","$"+f"{r['deepfm_price']:,.0f}")
        b.metric("Lemon XGBoost · 25%","$"+f"{r['xgboost_lemon_price']:,.0f}")
        if r["unrecognized_categories"]:
            st.info("These features use original unknown-category fallbacks: "+
                    ", ".join(r["unrecognized_categories"]))
        st.download_button("Download wholesale research receipt",
                json.dumps({**r,"vehicle":saved["fields"]},indent=2),
                file_name="deepfm-experimental-bid.json",
                mime="application/json",key="deepfm_download")


def _explanation(root):
    st.markdown("### 🥇 The models with fewer nightmare valuation surprises")
    st.write("**Asking:** BGE-256 word embeddings feed the WHOLE DeepCross/AFT "
             "neural network. It is not the old MiniLM-to-Ridge standalone text price. "
             "On the historical 2021 original test, its worst-five-percent "
             "average absolute dollar error was about **$19,985**, against "
             "**$21,453** for MiniLM-scalar DeepCross.")
    st.write("**Wholesale:** DeepFM learns vehicle feature interactions and a small "
             "neural network branch. A 75% DeepFM16 full-data / 25% Lemon XGBoost "
             "blend recorded **$8,860 CVaR95**, versus **$11,140** for the matched "
             "Lemon XGBoost, on historical 2015-normalized auction sales.")
    st.info("These are historical matched research comparisons, NOT proof that either "
            "experimental model is superior to the primary calculator for a "
            "November 2026 car. The primary price pipeline uses a different "
            "market-date transport and possibly Bayesian residual adjustments.")
    from pricing.cvar95_experiment_science import render as show_cvar_science
    show_cvar_science(root)
    st.markdown("[Original CVaR95 audit](https://github.com/botsandbolts/Pricing/blob/main/reports/cvar95/previously-selected/comparison.json)"
                " · [Full 109-model overnight validation audit](https://github.com/botsandbolts/Pricing/actions/workflows/overnight-cvar95-all-models.yml)")


def _apply_example():
    from pricing.description_examples import load_examples
    n=st.session_state.get("experimental_example",0)
    if not n:return
    e=load_examples()[n-1]
    keys={"year":"year","manufacturer":"make","model":"model","odometer":"mileage",
          "state":"state","type":"body","condition":"condition","fuel":"fuel",
          "transmission":"transmission","drive":"drive","title_status":"title",
          "cylinders":"cylinders","paint_color":"paint"}
    options={"body":["sedan","suv","truck","pickup","hatchback","coupe","wagon","van","convertible","other","unknown"],
             "condition":["good","excellent","fair","like new","new","salvage","unknown"],
             "fuel":["gas","diesel","hybrid","electric","other","unknown"],
             "transmission":["automatic","manual","other","unknown"],
             "drive":["fwd","rwd","4wd","unknown"],
             "title":["clean","rebuilt","salvage","lien","missing","parts only","unknown"],
             "cylinders":["4 cylinders","6 cylinders","8 cylinders","3 cylinders","5 cylinders","unknown"],
             "paint":["silver","white","black","grey","blue","red","green","brown","other","unknown"]}
    for name,key in keys.items():
        v=e.get("fields",{}).get(name)
        if v is None:continue
        if key in options:
            v=str(v).lower().strip()
            if v not in options[key]:v="unknown"
        if key=="state":v=str(v).lower()[:2]
        st.session_state["bge_ask_"+key]=v
    st.session_state["bge_ask_description"]=e.get("text","")[:8000]
    st.session_state.pop("experimental_combined_receipt",None)


class _FixedHistoricalPrice:
    """Adapter: reuse exactly the main calculator's date transport, without a surrogate ML model."""
    manifest={}
    def __init__(self,price):self.price=float(price)
    def predict(self,*args,**kwargs):return {"price":self.price}


def _adjust_price(transport,price,fields,side,when,allow_analogue):
    from marketbridge.time_engine import today
    if side=="ask":
        return transport.estimate(None,"",fields,when,side="ask",
                                  anchor_price=float(price),as_of=today())
    return transport.estimate(_FixedHistoricalPrice(price),"",fields,when,
                              side="bid",allow_newer_bid=allow_analogue,
                              as_of=today())


def _historical_bid_features(fields,auction,grade,allow_analogue):
    """Explicit five-year analogue, NOT the same model-year auction observation."""
    historical={**fields,"auction_date":str(auction),"auction_grade":grade}
    selected_year=int(fields["year"])
    if selected_year<=auction.year+1:
        return historical,None
    if not allow_analogue:
        raise ValueError(
            "This vehicle is newer than the 2014–2015 auction study. "
            "Enable 'Allow auction generation analogues for 2016–2021 cars' "
            "under Valuation date & auction coverage.")
    # TimeTransport's original wholesale reference is a five-year-old car in
    # 2015-indexed dollars, so use a five-year-old 2010 reference vehicle.
    # This historical identity is NEVER labeled as an actual auction of the
    # newer model year. Original trained weights are unchanged.
    if selected_year>2021:
        raise ValueError("Newer-than-2021 vehicles are not covered by this release.")
    reference_year=auction.year-5
    historical["year"]=reference_year
    return historical, {
        "kind":"age_matched_hypothetical_auction",
        "requested_model_year":selected_year,
        "proxy_model_year":reference_year,
        "reference_auction_date":str(auction),
        "warning":"Counterfactual five-year-old historical reference, not an observed "
                  "auction or verified valuation of this vehicle generation. "
                  "Make/trim and features may be out of the historical training distribution."}

def _usd(value):return "$"+f"{float(value):,.0f}"


def render(root):
    root=Path(root)
    st.markdown(CSS+"""
<section class="exp-hero" aria-label="Experimental vehicle pricing models">
<small>CAR PRICE STUDIO · THE RESEARCH GARAGE</small>
<h1>Two markets. One car.<br>Fewer pricing nightmares.</h1>
<p>Seller asking and wholesale auction estimates on the same screen, with the complete original experimental engines. CVaR95 asks how badly the model misses when luck takes a coffee break.</p>
<span class="exp-chip">✍ BGE + CVaR95 ResMLP / DeepCross</span>
<span class="exp-chip">🚘 CVaR95 DeepFM + Lemon XGBoost</span>
<span class="exp-chip">🛡️ CVaR95 · worst 5%</span>
<span class="exp-chip">📆 This month's scenario</span>
</section>""",unsafe_allow_html=True)
    from pricing.studio_design import market_guide
    from pricing.description_examples import load_examples
    from pricing.current_value import current_transport,clock_controls,render_market_clock,render_adjustment,render_time_evidence
    from pricing.cvar95_experiment_science import model_kpis,all_validation
    market_guide()
    examples=load_examples()
    st.markdown("### 🚗 01 / Pick a car, or bring your own")
    if examples:
        st.selectbox(
            f"Search {len(examples):,} real, price-redacted car adverts",
            range(len(examples)+1),key="experimental_example",
            format_func=lambda i:"Write my own listing" if i==0 else examples[i-1]["label"],
            on_change=_apply_example)
        st.caption("Real historical seller descriptions; prices and contact details redacted. Source: Tom’s Used Cars Dataset (CC BY 4.0). These are examples, not 400 fresh appraisals.")
    valuation_date,allow_analogue=clock_controls(prefix="experimental")
    # These historical models and their original data are never retrained by changing the date.
    try:
        transport,error=current_transport(root)
        render_market_clock(transport,valuation_date,error)
    except (OSError,ValueError,KeyError) as exc:
        transport=None
        st.warning("This month's market transport cannot be loaded: "+str(exc))
    ask_champion=champion_ask_available(root)
    bid_champion=champion_bid_available(root)
    ask_ready=ask_champion or ask_available(root)
    bid_ready=bid_champion or bid_available(root)
    a,b=st.columns(2)
    with a:
        st.markdown("**✍ Seller asking · full neural network**")
        if ask_champion:
            st.caption("NEW research winner: pinned BGE → trainable projection → ResMLP · trial 15. No Ridge or MiniLM price replacement.")
        else:
            st.caption("Original BGE text embeddings → complete DeepCross/AFT. No Ridge or MiniLM price replacement.")
            if ask_ready:st.info("CVaR95 ResMLP challenger is not release-verified; original model remains active.")
        if not ask_ready:st.warning("Asking model has not passed deployment parity; estimate disabled.")
    with b:
        st.markdown("**🔨 Wholesale auction · full ensemble**")
        if bid_champion:
            meta=json.loads((root/"models/neural-experimental-bid-cvar95/manifest.json").read_text())
            st.caption(f"NEW research winner: {meta['neural_weight']:.0%} tuned DeepFM16 + {meta['lemon_weight']:.0%} original Lemon XGBoost.")
        else:
            st.caption("Original 75% DeepFM16 + 25% Lemon XGBoost, with historical 2014–2015 auction assumptions.")
            if bid_ready:st.info("CVaR95 Bid challenger is not release-verified; original model remains active.")
        if not bid_ready:st.warning("Wholesale model has not passed deployment parity; estimate disabled.")
    st.caption("Original holdout metrics below are measured in historical dollars. They are NOT measured November 2026 accuracy.")
    for side,title,new_ready,new_folder in (
        ("ask","Seller asking",ask_champion,"neural-experimental-ask-cvar95"),
        ("bid","Wholesale auction",bid_champion,"neural-experimental-bid-cvar95"),
    ):
        if new_ready:
            d=json.loads((root/"models"/new_folder/"manifest.json").read_text())
            k=d["historical_test"]
            with st.expander(title+" · selected model original TEST accuracy",expanded=True):
                x,y,z=st.columns(3)
                x.metric("CVaR95 · worst 5%",_usd(k["CVaR95_absolute_USD"]))
                y.metric("RMSE",_usd(k["RMSE_USD"]))
                z.metric("MAE",_usd(k["MAE_USD"]))
                st.caption("Original untouched historical TEST after validation selection; "
                           "not a November 2026 forecast accuracy audit.")
                v=d["historical_validation"]
                old=d["historical_incumbent_test"]
                st.caption("Original validation CVaR95: "+_usd(v["CVaR95_absolute_USD"])
                           +" · previous model TEST CVaR95: "+_usd(old["CVaR95_absolute_USD"])
                           +" (the splits are not interchangeable).")
        elif all_validation(root,side):
            k=model_kpis(root,side)
            if k:
                with st.expander(title+" · previous historical test accuracy"):
                    x,y,z=st.columns(3)
                    x.metric("CVaR95 · worst 5%",_usd(k["CVaR95_absolute_USD"]))
                    y.metric("RMSE",_usd(k["RMSE_USD"]))
                    z.metric("MAE",_usd(k["MAE_USD"]))
    with st.expander("🧠 How the new models work · tail-risk research",expanded=False):
        explain_ask,explain_bid,explain_risk=st.tabs(
            ["✍ Asking · BGE ResMLP","🔨 Wholesale · DeepFM","🛡️ Risk & limitations"])
        with explain_ask:
            st.markdown("**A residual neural network that reads the advert.** "
                        "The pinned BGE-small encoder maps description text to "
                        "384 semantic dimensions. A learned projector compresses "
                        "this vector and residual MLP blocks mix it with the "
                        "vehicle's mileage, model year and category features. "
                        "The model predicts a historical *advertised* asking price. "
                        "No Ridge or MiniLM price head substitutes for the full network.")
            st.caption("Research winner: Ask25 trial 15 (when its original-test "
                       "checkpoint passes the separate serving release gate).")
        with explain_bid:
            st.markdown("**FM interactions plus an independent XGBoost voice.** "
                        "DeepFM learns individual effects, feature-pair interactions "
                        "and a compact nonlinear correction over auction attributes. "
                        "The complete neural output is blended with the original "
                        "Lemon-cleaned XGBoost wholesale estimate. The blend weight "
                        "was selected using VIN-disjoint historical validation.")
            st.caption("Research winner: Bid hyperparameter trial 18 when verified. "
                       "Auction label: completed 2014–2015 sale in 2015-indexed USD.")
        with explain_risk:
            st.markdown("**CVaR95:** the mean of absolute pricing errors among "
                        "the largest 5% of misses. It helps identify the models "
                        "that tend to make fewer costly mistakes; RMSE and MAE "
                        "remain secondary diagnostics.")
            st.warning("Both studies use historical training populations. "
                       "Ask estimates asking prices, Bid predicts auction sales, "
                       "and date transport to today's dollars is a scenario. "
                       "Models are NOT validated on November 2026 vehicle sales. "
                       "Recent model years may be outside the training support.")
    with st.expander("🔨 Auction reference assumptions",expanded=False):
        auction=st.date_input("Original auction date (2014–2015)",
            pd.Timestamp("2015-01-15").date(),min_value=pd.Timestamp("2014-01-01").date(),
            max_value=pd.Timestamp("2015-12-31").date(),key="experiment_joint_auction")
        grade=st.selectbox("Historical pre-sale grade",["unknown","1","2","3","4","5"],index=3,key="experiment_joint_grade")
        st.caption("2016–2021 cars can use an optional *hypothetical five-year-old auction analogue*. It is not evidence of what that vehicle generation sold for in 2015.")
    fields,notes,go=_vehicle("bge_ask",ready=ask_ready or bid_ready)
    if go:
        results={"fields":fields,"notes":notes,"date":str(valuation_date)}
        if ask_ready:
            try:
                from pricing.neural_ask_embedding_live import load as ask_load
                with st.spinner("Running the complete BGE + DeepCross/AFT network…"):
                    results["ask"]=(load_champion_ask(root) if ask_champion
                                    else ask_load(root)).predict(notes,fields)
            except Exception as exc:
                st.error("Asking engine unavailable: "+str(exc))
        if bid_ready:
            try:
                from pricing.neural_bid_blend_live import load as bid_load
                bid_fields,analogue=_historical_bid_features(fields,auction,grade,allow_analogue)
                with st.spinner("Running verified wholesale DeepFM + Lemon XGBoost…"):
                    results["bid"]=(load_champion_bid(root) if bid_champion
                                    else bid_load(root)).predict(bid_fields)
                if analogue:results["bid_analogue"]=analogue
            except (OSError,RuntimeError,ValueError,TypeError,KeyError) as exc:
                results["bid_error"]=str(exc)
                st.warning("Wholesale unavailable for this vehicle: "+str(exc))
        elif not bid_ready:
            results["bid_error"]="Verified wholesale weights are not installed or have failed the parity gate."
        st.session_state["experimental_combined_receipt"]=results
    saved=st.session_state.get("experimental_combined_receipt")
    if saved:
        st.markdown("### 💵 02 / Both prices, same car")
        prices={};adjusted={}
        columns=st.columns(2)
        for col,side,label in zip(columns,("ask","bid"),("Seller asking","Wholesale auction")):
            with col:
                result=saved.get(side)
                if result is None:
                    st.error(label+" could not be calculated: "+saved.get("bid_error" if side=="bid" else "ask_error","No verified prediction was returned."))
                    continue
                historical=float(result["price"])
                if side=="bid" and saved.get("bid_analogue"):
                    st.warning("⚠️ Historical age-matched analogue, not a real auction of this model year. Its 2015 baseline assumes an age-five proxy; accuracy for the newer generation is unknown.")
                    st.caption("Proxy model year: "+str(saved["bid_analogue"]["proxy_model_year"])+"; requested model year: "+str(saved["bid_analogue"]["requested_model_year"]))
                st.metric(label+" · historical anchor",_usd(historical))
                if transport is None:
                    st.warning("Current-month conversion unavailable; historical price only.")
                    continue
                try:
                    receipt=_adjust_price(transport,historical,saved["fields"],
                                          side,valuation_date,allow_analogue)
                    adjusted[side]=receipt;prices[side]=receipt["price"]
                    _price(label+" · "+str(valuation_date)+" scenario",receipt["price"],
                        "Date-adjusted estimate, NOT an observed 2026 transaction or a guaranteed bid.")
                    render_adjustment(receipt)
                except (OSError,ValueError,KeyError,TypeError) as exc:
                    st.warning("Cannot adjust "+label.lower()+" to this month: "+str(exc))
        if len(prices)==2:
            st.metric("Asking minus wholesale · indicative market gap",
                      _usd(prices["ask"]-prices["bid"]))
            st.caption("These are independent targets, not a guaranteed dealer profit or a matched-car trading spread.")
        if adjusted:
            render_time_evidence(transport,adjusted)
        if "ask" in saved and ask_ready:
            with st.expander("✍ What-if? Rewrite the description · rerun the whole neural model"):
                alt=st.text_area("Alternative advertisement",value=saved["notes"],key="experimental_reword",max_chars=8000)
                if st.button("Recalculate complete neural asking model",key="experimental_reword_go"):
                    try:
                        from pricing.neural_ask_embedding_live import load
                        revised=(load_champion_ask(root) if ask_champion
                                 else load(root)).predict(alt,saved["fields"])["price"]
                        new=_adjust_price(transport,revised,saved["fields"],"ask",valuation_date,allow_analogue) if transport else None
                        st.metric("Rewritten historical asking",_usd(revised),delta=_usd(revised-saved["ask"]["price"]))
                        if new:st.metric("Rewritten date-adjusted asking",_usd(new["price"]))
                        st.caption("A change in model prediction is not evidence rewriting an advert changes what a buyer pays.")
                    except (OSError,RuntimeError,ValueError,TypeError,KeyError) as exc:
                        st.error("What-if calculation unavailable: "+str(exc))
            with st.expander("📉 Mileage sensitivity · full-network reruns"):
                try:
                    from pricing.neural_ask_embedding_live import load
                    model=load_champion_ask(root) if ask_champion else load(root)
                    prepared=model.prepare_text(saved["notes"])
                    rows=[]
                    for change in (-20000,-10000,0,10000,20000):
                        f={**saved["fields"],"odometer":max(0,saved["fields"]["odometer"]+change)}
                        p=model.predict_prepared(f,prepared)["price"]
                        if transport:p=_adjust_price(transport,p,f,"ask",valuation_date,allow_analogue)["price"]
                        rows.append({"Mileage":int(f["odometer"]),"Asking USD":p})
                    st.line_chart(pd.DataFrame(rows).set_index("Mileage"))
                except (OSError,RuntimeError,ValueError,TypeError,KeyError) as exc:
                    st.info("Mileage sensitivity unavailable: "+str(exc))
        report={"model_family":"Parity-gated CVaR95-selected neural challengers (original fallback if unavailable)",
                "valuation_date":str(valuation_date),"vehicle":saved["fields"],
                "historical_predictions":{k:saved[k] for k in ("ask","bid") if k in saved},
                "date_adjusted":adjusted,"bid_analogue":saved.get("bid_analogue"),"bid_error":saved.get("bid_error"),"description":"omitted for privacy",
                "warning":"Date transport is an unvalidated individual 2026 scenario."}
        st.download_button("📥 Download both-model research receipt (JSON)",
            json.dumps(report,indent=2,default=str),file_name="experimental-bid-ask.json",
            mime="application/json",key="experimental_both_download")
    st.markdown("### 🧰 03 / Other tools from the main garage")
    st.caption("The original car calculator, listing inspector, timelines, scientific evidence, merchant comparisons and diagnostics remain accessible using the Studio panels above. No parallel, potentially inconsistent copy of their models is made.")
    with st.expander("🛡️ Why CVaR95? The business case and the original scientific evidence"):
        _explanation(root)
