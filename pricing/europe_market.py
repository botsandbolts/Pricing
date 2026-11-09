"""European ASK studio. English-only, data-honest, model-gated UX.

The American studio and its Ask/Bid pipelines are not imported or modified.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from pricing.europe_model import (ROOT,EuropeanModelUnavailable,load_bundle,
    available,estimate,batch_estimate,explain)

DATASET="https://huggingface.co/datasets/KTHESAT/european-used-car-market-2m-listings"
NAV=[
    "🚗 Price my car","🔎 Listing Inspector","📉 Value over time",
    "📊 Evaluation","🧪 Model lab","🔬 The science",
    "🕵️ Deal Detective","🧹 Outlier lab","🌍 Market intelligence",
    "🏪 Merchant lens","🚦 Pick your garage",
]
COUNTRIES={"DE":"Germany","FR":"France","NL":"Netherlands","BE":"Belgium",
           "AT":"Austria","ES":"Spain","IT":"Italy","PL":"Poland","CH":"Switzerland"}
METRIC_LABELS={"cvar95_absolute_eur":"CVaR95 · worst 5%",
               "mae_eur":"Mean absolute error","rmse_eur":"Root mean squared error",
               "median_ape_pct":"Median absolute percentage error"}
EVAL_FIELDS=("cvar95_absolute_eur","mae_eur","rmse_eur","median_ape_pct")

def money(value):
    return "€"+format(float(value),",.0f")

def _css():
    st.markdown("""<style>
    .eu-hero{padding:clamp(20px,4vw,44px);margin:10px 0 24px;border-radius:24px;
      background:radial-gradient(circle at 88% 18%,#295b75 0%,transparent 40%),
      linear-gradient(125deg,#14253b,#233b53 55%,#186255);
      color:#fff;border:2px solid #edd994;box-shadow:0 15px 38px #10203029}
    .eu-hero h1{color:#fff!important;font-size:clamp(1.8rem,4vw,3rem);line-height:1.13;
      letter-spacing:-.045em;margin:8px 0 12px}
    .eu-hero p{color:#e6eff4;max-width:770px;line-height:1.6}
    .eu-kicker{color:#ffe49d;font-weight:900;letter-spacing:.15em;font-size:.78rem}
    .eu-tags{display:flex;flex-wrap:wrap;gap:9px;margin:12px 0 0}
    .eu-tags span{border:1px solid #ffffff70;border-radius:99px;padding:5px 10px;
      background:#ffffff17;font-size:.85rem}
    .eu-panel{padding:16px 20px;background:rgba(32,109,116,.06);
      border:1px solid #9dbab6;border-radius:15px;margin:9px 0 15px}
    .eu-result{padding:22px 26px;border-radius:19px;margin:16px 0;
      background:linear-gradient(130deg,#11273c,#224765);
      border:2px solid #eed78a;color:white}
    .eu-result small{letter-spacing:.1em;font-weight:800;color:#d7efc1}
    .eu-result strong{display:block;color:white!important;
      font-size:clamp(2.3rem,5vw,4rem);letter-spacing:-.05em}
    .eu-result p{color:#e6edf0;margin:2px 0}
    </style>""",unsafe_allow_html=True)

def _header():
    st.markdown("""<section class="eu-hero" aria-label="European automotive price studio">
      <div class="eu-kicker">🇪🇺 CAR PRICE STUDIO · EUROPEAN EDITION</div>
      <h1>From Hamburg to Rome. Same love of cars.<br>Less guessing. More evidence.</h1>
      <p>Research-backed used-car <b>asking-price estimates</b> for supported European
         markets. No fictional auction bids. No registration. No advertising.
         Just a carefully tested XGBoost engine with the science left in.</p>
      <div class="eu-tags"><span>🇩🇪 Built independently in Germany</span>
        <span>€ Advertised Ask, not sale price</span><span>🔍 Explainable AI</span>
        <span>🛡️ Worst-error risk measured</span></div></section>""",
        unsafe_allow_html=True)

def _bundle():
    try:
        return load_bundle(),None
    except (EuropeanModelUnavailable,ValueError,OSError,RuntimeError) as exc:
        return None,str(exc)

@st.cache_data(show_spinner=False,ttl=7200)
def _market_table():
    file=ROOT/"market_stats.csv.gz"
    if not file.is_file():
        return pd.DataFrame()
    return pd.read_csv(file,compression="gzip",
                       dtype={"seller_country":str,"make":str,"model":str})

def _snapshot(bundle):
    return bundle[1].get("snapshot_date","2026-06-01") if bundle else "2026-06-01"

def _choices(schema):
    vocab=schema["category_vocabulary"] if schema else {}
    def values(key,default):
        names=sorted(v for v in vocab.get(key,{}) if v!="__missing__")
        return names or default
    return values

def _edit_vehicle(schema):
    """Persistent European facts; no US mile/state fields or model assumptions."""
    values=_choices(schema)
    st.markdown("### 🚘 01 / Tell us about your European car")
    st.caption("European conventions: km, horsepower, registration year and euro asking prices. "
               "Unknown specifications stay unknown.")
    with st.form("eu_ask_form",clear_on_submit=False):
        a,b,c=st.columns(3)
        countries=values("seller_country",list(COUNTRIES))
        countries=[v for v in countries if v!="__missing__"] or list(COUNTRIES)
        country=a.selectbox("Country",countries,
            format_func=lambda v:COUNTRIES.get(v.upper(),v.title()),key="eu_country")
        makes=values("make",["volkswagen","bmw","mercedes benz","toyota","renault"])
        preferred="volkswagen" if "volkswagen" in makes else makes[0]
        make=b.selectbox("Make",makes,index=makes.index(preferred),
                         key="eu_make")
        model=c.text_input("Model",value="golf" if make=="volkswagen" else "",
                           key="eu_model",max_chars=80,
                           help="Enter the exact model as shown in a European advert.")
        a,b,c=st.columns(3)
        registration=a.number_input("First-registration year",
                        min_value=1950,max_value=2026,value=2020,step=1,
                        key="eu_reg_year")
        month=b.selectbox("First-registration month",list(range(1,13)),
                          index=5,key="eu_reg_month")
        km=c.number_input("Mileage · km",min_value=0,max_value=2_000_000,
                          value=85000,step=1000,key="eu_km")
        a,b,c=st.columns(3)
        power=a.number_input("Power · hp",min_value=20,max_value=1200,
                             value=110,key="eu_hp")
        fuel=b.selectbox("Fuel",values("fuel",["petrol","diesel","electric","hybrid"]),
                         key="eu_fuel")
        transmission=c.selectbox("Transmission",values("transmission",["manual","automatic"]),
                                 key="eu_transmission")
        with st.expander("🔧 Optional vehicle and advert details"):
            a,b,c=st.columns(3)
            body=a.selectbox("Body style",["__missing__"]+values("body_type",[]),
                             key="eu_body")
            drive=b.selectbox("Drivetrain",["__missing__"]+values("drive_type",[]),
                              key="eu_drive")
            color=c.selectbox("Colour",["__missing__"]+values("color",[]),
                              key="eu_color")
            a,b,c=st.columns(3)
            trim=a.text_input("Version / trim (optional)",key="eu_trim")
            city=b.text_input("Seller city (optional)",key="eu_city")
            source=c.selectbox("Listing platform (optional)",
                 ["__missing__"]+values("source_site",[]),key="eu_source")
            a,b,c=st.columns(3)
            doors=a.number_input("Doors",0,8,5,key="eu_doors")
            seats=b.number_input("Seats",0,12,5,key="eu_seats")
            accident=c.selectbox("Accident-free status",["Not stated","Yes","No"],
                                 key="eu_accident")
        submitted=st.form_submit_button("🇪🇺 Calculate European asking price",
                   type="primary",use_container_width=True,disabled=schema is None)
    vehicle={
       "seller_country":country.lower(),"make":make,"model":model.strip(),
       "reg_year":int(registration),"reg_month":int(month),"mileage_km":float(km),
       "power_hp":float(power),"fuel":fuel,"transmission":transmission,
       "body_type":body,"drive_type":drive,"color":color,
       "version":trim,"seller_city":city,"source_site":source,
       "doors":float(doors) if doors else None,"seats":float(seats) if seats else None,
       "accident_free":None if accident=="Not stated" else int(accident=="Yes"),
       # The locked training-data snapshot date, NOT an unsupported live-price adjustment.
       "first_seen_at":schema.get("snapshot_date") if schema else None,
    }
    if submitted:
        if not model:
            st.error("Enter the exact model name, such as Golf or 320d.")
        else:
            try:
                price=estimate(vehicle)
                st.session_state["eu_receipt"]={
                    "vehicle":vehicle,"price_eur":price,"snapshot":_snapshot((None,schema)),
                    "market":"Europe","engine":"XGBoost Gaussian AFT · rank-1 sealed-test winner"
                }
                st.success("Valuation complete. The estimate is for a dataset-era advert, not a guaranteed sale.")
            except (ValueError,EuropeanModelUnavailable,RuntimeError) as exc:
                st.error("Valuation unavailable: "+str(exc))
    return vehicle

def _receipt(bundle):
    quote=st.session_state.get("eu_receipt")
    if not quote:
        st.info("Your European price receipt will appear here after valuation.")
        return None
    st.markdown('<div class="eu-result"><small>EUROPEAN ADVERTISED ASK · EUR</small>'
                '<strong>'+escape(money(quote["price_eur"]))+'</strong>'
                '<p>Rank-1 Gaussian AFT · research estimate · no auction/Bid component</p>'
                '</div>',unsafe_allow_html=True)
    st.caption("Historical listing-price research, not evidence of an actual sale. "
               "Do not interpret the price as an insurance, customs or tax ruling.")
    test=bundle[2]["winning_model"]["test_once"] if bundle else {}
    if test:
        a,b,c=st.columns(3)
        a.metric("Held-out test MAE",money(test["mae_eur"]))
        b.metric("Held-out test CVaR95",money(test["cvar95_absolute_eur"]))
        c.metric("Test median absolute % error",f'{test["median_ape_pct"]:.2f}%')
        st.caption("These are population-level test errors, not a guaranteed interval for this vehicle.")
    if st.button("🔍 Why THAT price? · show model contributions",
                 key="eu_explain_click",use_container_width=True):
        try:
            factors=explain(quote["vehicle"],bundle)
            if factors:
                d=pd.DataFrame(factors[:12],columns=["Feature","log-price contribution"])
                st.bar_chart(d.set_index("Feature"))
                st.caption("Signed additive XGBoost contributions on the log-price scale. "
                    "They show model associations, not causal effects.")
            else:
                st.info("Per-vehicle contributions were not available in this model format.")
        except (ValueError,RuntimeError) as exc:
            st.warning("Could not calculate contributions: "+str(exc))
    return quote

def _comparables(table,vehicle):
    if table.empty:
        st.info("Verified market summaries are not packaged yet; comparables are unavailable.")
        return
    country=str(vehicle["seller_country"]).lower()
    make=str(vehicle["make"]).lower()
    model=str(vehicle["model"]).lower()
    year=int(vehicle["reg_year"])
    matching=table[(table.seller_country.str.lower()==country)&
                   (table.make.str.lower()==make)&
                   (table.model.str.lower()==model)]
    if matching.empty:
        st.info("No sufficiently large matched group in the published research summary. "
                "That does not mean the vehicle is rare.")
        return
    matching=matching.assign(distance=(matching.reg_year-year).abs())
    close=matching[matching.distance<=3].sort_values(["distance","count"],ascending=[True,False])
    if close.empty:
        st.info("No nearby registration-year cohort with at least 8 observations.")
        return
    show=close.head(12)[["reg_year","count","median_ask_eur","median_km"]].rename(columns={
        "reg_year":"Registration year","count":"Listings",
        "median_ask_eur":"Median listed Ask · €","median_km":"Median mileage · km"
    })
    st.dataframe(show,use_container_width=True,hide_index=True)
    st.caption("Actual market-cohort aggregates. Similar vehicles are NOT confirmed duplicate cars. "
               "Asking prices are not completed-sale prices; mix of condition and equipment varies.")

def _valuation_page(bundle,error):
    st.markdown("### 🧮 European Asking Price Lab")
    if bundle is None:
        st.warning("The rank-1 model is undergoing reproducibility and package verification. "
                   "The calculator is gated; no fabricated or substitute prices will be shown.")
        st.caption(error or "Trained weights and vocabulary must be published together.")
    vehicle=_edit_vehicle(bundle[1] if bundle else None)
    quote=_receipt(bundle)
    with st.expander("📚 Comparable European adverts · inspect the evidence",expanded=True):
        _comparables(_market_table(),quote["vehicle"] if quote else vehicle)
    _feedback(quote)

def _feedback(quote):
    if quote is None:
        return
    st.markdown("### ⭐ Was this asking-price estimate useful?")
    with st.form("eu_feedback_form"):
        rating=st.radio("Rate the estimate",[
             "1 · Catastrophic","2 · Poor","3 · Good","4 · Excellent"
        ],index=None,horizontal=True)
        consent=st.checkbox("I agree to submit this rating and broad car facts, "
                            "without my description, VIN, address or contact details.")
        submit=st.form_submit_button("Send European feedback")
    if submit:
        if not consent or rating is None:
            st.warning("Please choose a rating and explicitly agree before submitting.")
        else:
            # No silent fallback into the US USD-only feedback table.
            try:
                from pricing.europe_feedback import store_feedback
                ok=store_feedback(quote,int(rating[0]),explicit_consent=consent)
                if ok:st.success("Thank you. Your European feedback has been saved.")
                else:st.warning("Feedback storage is unavailable; your rating was not stored.")
            except (ValueError,OSError,RuntimeError) as exc:
                st.warning("Feedback could not be saved: "+str(exc))

def _timeline(bundle):
    st.markdown("### 📉 Value over time · European scenario lab")
    st.caption("What-if estimates for the same car as registration year or mileage changes. "
               "These are scenarios, not observed historical sales or guaranteed depreciation.")
    quote=st.session_state.get("eu_receipt")
    if not quote or bundle is None:
        st.info("First calculate a European price to enable real model-based scenarios.")
        return
    vehicle=quote["vehicle"]
    km=float(vehicle["mileage_km"])
    mileage_levels=sorted({int(max(0,min(2_000_000,km*factor)))
                           for factor in (.25,.5,.75,1,1.25,1.5,2)})
    variants=[{**vehicle,"mileage_km":v} for v in mileage_levels]
    result=batch_estimate(variants,bundle)
    chart=pd.DataFrame({"Mileage · km":mileage_levels,"Estimated Ask · €":result})
    st.line_chart(chart.set_index("Mileage · km"))
    registration=int(vehicle["reg_year"])
    years=[y for y in range(max(1950,registration-4),
             min(2026,registration+4)+1) if y<=int(_snapshot(bundle)[:4])]
    if years:
        samples=[{**vehicle,"reg_year":y} for y in years]
        estimates=batch_estimate(samples,bundle)
        st.line_chart(pd.DataFrame({"Registration year":years,
              "Estimated Ask · €":estimates}).set_index("Registration year"))
    st.caption("No model output is a prospective time-price forecast. "
               "Mileage, age and unobserved condition can be correlated.")

def _inspector(bundle):
    st.markdown("### 🔎 Listing Inspector · compare, don't delete")
    st.caption("Popular vehicles with similar year and mileage are NOT duplicates. "
               "This tool compares aggregates and checks a listing against the model.")
    quote=st.session_state.get("eu_receipt")
    if quote:
        _comparables(_market_table(),quote["vehicle"])
        _forensic(quote)
    else:
        st.info("Create a valuation first to inspect comparable market cohorts.")

def _forensic(quote):
    st.markdown("### 🕵️ Asking-price anomaly detective")
    st.caption("An unusual asking price is a research signal—not evidence of fraud, "
               "smuggling or misrepresentation.")
    observed=st.number_input("Price in the advert · €",min_value=100.0,
                             max_value=1_000_000.0,value=float(round(quote["price_eur"])),
                             step=100.0,key="eu_observed_ask")
    model=float(quote["price_eur"])
    diff=observed-model
    a,b=st.columns(2)
    a.metric("Compared with model estimate",money(abs(diff)),
             delta=("Higher" if diff>0 else "Lower" if diff<0 else "Equal"))
    b.metric("Relative difference",f"{abs(diff)/max(model,100):.1%}")
    if abs(diff)/max(model,100)>.35:
        st.warning("This price is substantially different from the research estimate. "
                   "Check condition, specification, currency and market context. "
                   "No inference of illegality is justified.")
    else:
        st.info("Model and listing are reasonably close by this simple relative-error check. "
                "This is not a confidence interval or fraud classifier.")
    report={"market":"European advertised ASK","valuation_eur":model,
            "listing_ask_eur":observed,"difference_eur":diff,
            "model":"Rank-1 Gaussian AFT",
            "vehicle":{k:v for k,v in quote["vehicle"].items()
                       if k not in ("seller_city","version")},
            "limitations":"Not a sale price, validated personal interval, customs ruling, or fraud finding"}
    st.download_button("Download transparent research report · JSON",
        json.dumps(report,indent=2),file_name="european-ask-evidence.json",
        mime="application/json",key="eu_forensic_json")

def _evaluation(bundle):
    st.markdown("### 📊 European evaluation · full sealed holdouts")
    if bundle is None:
        st.info("A validated benchmark is bundled only with the released model.")
        return
    comp=bundle[2]
    train=comp["counts"]["train"]
    valid=comp["counts"]["validation"]
    test=comp["counts"]["test"]
    a,b,c=st.columns(3)
    a.metric("Train",f"{train:,}")
    b.metric("Validation",f"{valid:,}")
    c.metric("Sealed test",f"{test:,}")
    winner=comp["winning_model"]
    rows=[]
    for name,source in [("Validation",winner["validation"]),("Sealed test",winner["test_once"])]:
        rows.append({"Cohort":name,
          **{METRIC_LABELS[k]:round(source[k],2) for k in EVAL_FIELDS}})
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    st.caption("Model selected using validation CVaR95 only. "
        "The sealed test was evaluated once for the winner. "
        "These results are not individually calibrated uncertainty bands.")
    st.warning("A reliable per-country or per-model holdout scorecard has not yet been "
        "reproduced. Do not imply equal accuracy in every country.")
    st.download_button("Download research comparison · JSON",
        json.dumps(comp,indent=2),file_name="european-xgboost-comparison.json",
        mime="application/json",key="eu_eval_json")

def _model_lab(bundle):
    st.markdown("### 🧪 European Model Lab")
    st.caption("Twelve XGBoost candidates: ten Ray-tuned workers and two fixed baselines. "
               "Neural challengers remain independent until they pass the same checks.")
    if bundle is None:
        st.info("The model laboratory will unlock with the verified release bundle.")
        return
    comp=bundle[2]
    rows=[]
    for candidate in comp["leaderboard"]:
        val=candidate["validation"]
        rows.append({"Candidate":candidate["origin"],
          "Objective":candidate["family"],
          "CVaR95 · €":round(val["cvar95_absolute_eur"]),
          "MAE · €":round(val["mae_eur"]),
          "RMSE · €":round(val["rmse_eur"]),
          "Boosting rounds":candidate["iterations"]})
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    top=comp["winning_model"]
    with st.expander("Inspect the winning hyperparameters"):
        st.json(top["config"])
    st.caption("CVaR95 is the average absolute euro error in the worst 5% of cases; "
               "lower is better. Hyperparameters chosen using validation, never test.")

def _science(bundle):
    st.markdown("### 🔬 Under the bonnet · serious models, appropriately modest claims")
    st.markdown("""**What is the target?** The advertised European asking price, in euros.
    It is not a completed sale, wholesale auction bid, taxation estimate or customs value.

    **Data.** The independent research dataset contains 2,003,645 listing rows.
    After valid positive-price filtering, 2,003,371 rows were eligible. No unverified
    duplicate removal or similarity-based thinning was applied.

    **How does it work?** The winning algorithm is a histogram-based XGBoost
    regression tree ensemble with an exact-observation Gaussian accelerated
    failure-time loss. AFT is used as a distribution-aware positive-price
    objective: the data do **not** contain observed censoring intervals.

    **Why CVaR95?** A small average miss can conceal very costly mistakes.
    CVaR95 measures the mean absolute error among the worst 5% of validation
    observations. It is optimized through validation-based model selection.

    **What can't the model know?** An unreported fault, accident history, seller
    incentives, export taxes, language in a description or the negotiated price.
    No unsupported feature is silently invented.""")
    st.markdown("**Source and usage:** [European Used Car Market dataset]("+DATASET+
         ") · data supplied by KTHESAT under CC BY-NC-SA 4.0. "
         "Car Price Studio's software is separately MIT-licensed.")
    if bundle:
        winning=bundle[2]["winning_model"]
        st.info(f"Winner: Gaussian AFT with {winning['iterations']} boosting rounds; "
                f"sealed-test MAE {money(winning['test_once']['mae_eur'])}; "
                f"sealed-test CVaR95 {money(winning['test_once']['cvar95_absolute_eur'])}.")

def _market_intelligence():
    st.markdown("### 🌍 Market intelligence · European marketplace facts")
    table=_market_table()
    if table.empty:
        st.info("Aggregated market evidence is still being packaged.")
        return
    for col in ("seller_country","make","model"):
        if col not in table:st.warning("Market summary missing "+col);return
    countries=sorted(table.seller_country.dropna().unique().tolist())
    country=st.selectbox("Country",countries,key="eu_intel_country",
        format_func=lambda c:COUNTRIES.get(c.upper(),c))
    chosen=table[table.seller_country==country]
    biggest=chosen.groupby("make",as_index=False)["count"].sum().nlargest(20,"count")
    st.bar_chart(biggest.set_index("make"))
    st.caption("Counts describe the dataset, not the whole European car fleet. "
               "No inference about legal sales volumes or dealer profits is supported.")
    st.dataframe(biggest.rename(columns={"make":"Make","count":"Listing observations"}),
                 use_container_width=True,hide_index=True)

def _merchant_lens():
    st.markdown("### 🏪 Merchant lens · defensible scope")
    st.info("The European dataset does not establish reliable seller identities, "
            "dealer contracts or cross-market sales. We can compare country/marketplace "
            "aggregates, but we will not fabricate dealer rankings.")
    _market_intelligence()

def _garage_choices():
    st.markdown("### 🚦 Pick your garage · European research explorer")
    st.caption("Find makes with broad documented listing coverage before choosing a car.")
    table=_market_table()
    if table.empty:
        st.info("Market sample summaries are not yet packaged.")
        return
    country=st.selectbox("Market country",sorted(table.seller_country.unique().tolist()),
                         key="eu_garage_country",format_func=lambda c:COUNTRIES.get(c.upper(),c))
    d=table[table.seller_country==country].groupby(["make","model"],as_index=False).agg(
        sample_count=("count","sum"))
    st.dataframe(d.nlargest(25,"sample_count"),hide_index=True,use_container_width=True)
    st.caption("This is a listing-coverage explorer, not a garage recommendation or user-preference model.")

def render():
    _css()
    _header()
    from pricing.analytics import render_privacy
    render_privacy()
    bundle,error=_bundle()
    tab=st.pills("European Studio panels",NAV,selection_mode="single",
                 key="eu_workspace",label_visibility="collapsed") or NAV[0]
    if tab=="🚗 Price my car":
        _valuation_page(bundle,error)
    elif tab=="🔎 Listing Inspector":
        _inspector(bundle)
    elif tab=="📉 Value over time":
        _timeline(bundle)
    elif tab=="📊 Evaluation":
        _evaluation(bundle)
    elif tab=="🧪 Model lab":
        _model_lab(bundle)
    elif tab=="🔬 The science":
        _science(bundle)
    elif tab=="🕵️ Deal Detective":
        st.markdown("### 🕵️ Deal Detective · evidence before allegations")
        q=st.session_state.get("eu_receipt")
        if q:_forensic(q)
        else:st.info("First value a European listing to unlock the anomaly check.")
    elif tab=="🧹 Outlier lab":
        st.markdown("### 🧹 Outlier Lab")
        st.info("Suspected unusual observations are not confirmed errors or duplicates. "
                "This edition uses the untouched input population and does not remove rows.")
        _market_intelligence()
    elif tab=="🌍 Market intelligence":
        _market_intelligence()
    elif tab=="🏪 Merchant lens":
        _merchant_lens()
    elif tab=="🚦 Pick your garage":
        _garage_choices()
    st.divider()
    st.caption("🇩🇪 Independently built in Hamburg · No registration · No ads · "
        "Asking prices only · [Open-source code](https://github.com/botsandbolts/Pricing) · "
        "[Dataset and attribution]("+DATASET+") · "
        "[Privacy policy](?legal=privacy) · [Imprint](?legal=imprint)")
