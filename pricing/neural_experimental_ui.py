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


def _vehicle(prefix,bid=False,ready=False):
    with st.form(prefix+"_vehicle",clear_on_submit=False):
        st.markdown("#### 01 / Describe the vehicle")
        a,b,c=st.columns(3)
        year=a.number_input("Model year",1900,2021,2015,key=prefix+"_year")
        make=b.text_input("Make","Toyota",key=prefix+"_make")
        model=c.text_input("Model / trim","Corolla",key=prefix+"_model")
        a,b,c=st.columns(3)
        miles=a.number_input("Mileage (miles)",0,2000000,95000,step=1000,key=prefix+"_mileage")
        state=b.text_input("US state code","ca",max_chars=2,key=prefix+"_state")
        body=c.selectbox("Body style",["sedan","suv","truck","pickup",
              "hatchback","coupe","wagon","van","convertible","other","unknown"],
              key=prefix+"_body")
        with st.expander("⚙️ Mechanical, condition and title details"):
            a,b,c=st.columns(3)
            condition=a.selectbox("Condition",["good","excellent","fair",
                 "like new","new","salvage","unknown"],key=prefix+"_condition")
            fuel=b.selectbox("Fuel",["gas","diesel","hybrid","electric","other","unknown"],
                    key=prefix+"_fuel")
            transmission=c.selectbox("Transmission",["automatic","manual","other","unknown"],
                    key=prefix+"_transmission")
            a,b,c=st.columns(3)
            drive=a.selectbox("Drivetrain",["fwd","rwd","4wd","unknown"],key=prefix+"_drive")
            title=b.selectbox("Title",["clean","rebuilt","salvage","lien",
                      "missing","parts only","unknown"],key=prefix+"_title")
            cylinders=c.selectbox("Cylinders",["4 cylinders","6 cylinders","8 cylinders",
                      "3 cylinders","5 cylinders","unknown"],key=prefix+"_cylinders")
            paint=a.selectbox("Paint",["silver","white","black","grey",
                     "blue","red","green","brown","other","unknown"],key=prefix+"_paint")
        fields={"year":int(year),"manufacturer":make,"model":model,
                "odometer":float(miles),"state":state,"type":body,
                "condition":condition,"fuel":fuel,
                "transmission":transmission,"drive":drive,
                "title_status":title,"cylinders":cylinders,"paint_color":paint}
        if bid:
            st.markdown("#### 02 / Historical wholesale auction")
            a,b=st.columns(2)
            auction=a.date_input("2014–2015 auction date",
                  pd.Timestamp("2015-01-15").date(),
                  min_value=pd.Timestamp("2014-01-01").date(),
                  max_value=pd.Timestamp("2015-12-31").date(),
                  key=prefix+"_auction_date")
            grade=b.selectbox("Pre-sale auction grade",["unknown","1","2","3","4","5"],
                              index=3,key=prefix+"_grade")
            fields["auction_date"]=str(auction);fields["auction_grade"]=grade
            notes=""
        else:
            st.markdown("#### 02 / The car's entire story")
            notes=st.text_area("Seller's description",
                 "Well maintained. Clean title. New tires.",max_chars=8000,height=140,
                 key=prefix+"_description",
                 help="BGE turns wording into an embedding; complete DeepCross/AFT predicts the price. No Ridge head.")
        sent=st.form_submit_button(
                "🚀 Run complete "+("wholesale" if bid else "asking-price neural")+" model",
                type="primary",disabled=not ready,use_container_width=True)
    return fields,notes,sent


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

def render(root):
    root=Path(root)
    st.markdown(CSS+"""
<section class="exp-hero" aria-label="Experimental vehicle pricing models">
<small>CAR PRICE STUDIO · RESEARCH GARAGE · EXPERIMENTAL ONLY</small>
<h1>Different engines.<br>The whole story.</h1>
<p>Explore the full neural response to a seller's description, and an
ensemble for historical wholesale auctions. Real model weights, guarded
by strict original-holdout serving tests. Ordinary predictions meet the
unusual cases. Not another pricing oracle in a lab coat.</p>
<span class="exp-chip">✍ BGE + DeepCross/AFT</span>
<span class="exp-chip">🚘 DeepFM + XGBoost</span>
<span class="exp-chip">🛡️ CVaR95 · WORST 5%</span>
<span class="exp-chip">🇩🇪 Made in Germany</span>
</section>""",unsafe_allow_html=True)
    ask,bid,science=st.tabs([
        "✍ Full neural asking","🚘 Wholesale ensemble","🛡️ CVaR95 · Business & science"])
    with ask:_ask(root)
    with bid:_bid(root)
    with science:_explanation(root)
