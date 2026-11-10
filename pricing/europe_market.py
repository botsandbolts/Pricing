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
from pricing.i18n import tr, editorial_voice

from pricing.europe_model import (ROOT,EuropeanModelUnavailable,load_bundle,
    available,estimate,batch_estimate,explain)
from pricing.europe_price_clock import adjusted_quote, read_index, level_at, month_number, month_label
from pricing.europe_eurostat import (eurostat_level_at, read_eurostat)
from pricing.europe_hofm3 import SUPPORTED_COUNTRIES, ROOT as HOFM3_ROOT
from pricing.europe_model_pulse import read_pulse,lookup_pulse
from pricing.europe_bayesian_state_space import model_scenario_series

DATASET="https://huggingface.co/datasets/KTHESAT/european-used-car-market-2m-listings"
NAV=[
    "🚗 Price my car","🔎 Listing Inspector","📉 Value over time",
    "📈 Price Indices","📊 Evaluation","🧪 Model lab","🔬 The science",
    "🕵️ Deal Detective","🧹 Outlier lab","🌍 Market intelligence",
    "🏪 Merchant lens","🚦 Pick your garage",
]
COUNTRIES={"DE":"Germany","FR":"France","NL":"Netherlands","BE":"Belgium",
           "AT":"Austria","ES":"Spain","IT":"Italy","PL":"Poland"}
COUNTRY_OPTIONS=tuple(x for x in ("de","fr","nl","be","at","es","it","pl") if x in SUPPORTED_COUNTRIES)
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
    .st-key-eu_parody_banner{background:#0d1117!important;border:4px solid #ff991f!important;
      border-radius:21px!important;box-shadow:0 16px 40px #00000040!important;
      padding:clamp(8px,1.5vw,15px)!important;margin:0 0 16px!important}
    .st-key-eu_parody_banner [data-testid="stImage"]{border-radius:12px;overflow:hidden}
    .eu-parody-tagline{color:#ff991f!important;text-align:left;font-size:clamp(.85rem,1.7vw,1.2rem);
      font-weight:950;letter-spacing:.055em;margin:8px 3px 2px}
    .eu-steps{display:flex;flex-wrap:wrap;gap:10px;align-items:stretch;margin:12px 0 17px}
    .eu-step{flex:1 1 175px;min-width:0;border:1px solid #83b9ac;border-radius:13px;padding:14px;background:rgba(42,106,106,.08)}
    .eu-step b{display:block;margin:0 0 6px;font-size:.95rem;color:inherit}
    .eu-step span{font-size:.86rem;line-height:1.5;display:block}
    .eu-arrow{display:flex;align-items:center;font-size:2rem;color:#298e81;font-weight:bold}
    @media(max-width:650px){.eu-step{flex:1 1 100%}.eu-arrow{display:none}}
    .eu-impact-graph,.eu-benchmark{border:1px solid #8fb6b2;border-radius:15px;
       padding:13px 17px;margin:8px 0 16px;background:rgba(31,91,91,.055)}
    .eu-impact-row,.eu-benchmark-row{display:grid;
       grid-template-columns:minmax(130px,1.4fr) minmax(80px,2fr) minmax(65px,90px);
       gap:12px;align-items:center;padding:8px 2px}
    .eu-impact-label,.eu-benchmark-row span{font-size:.88rem;line-height:1.4}
    .eu-impact-track{height:17px;border-radius:9px;background:#d5e2df;overflow:hidden}
    .eu-impact-bar{height:100%;border-radius:9px;min-height:17px}
    .eu-impact-up{background:#16806d}.eu-impact-down{background:#c86d3a}
    .eu-impact-reference{background:#7896b2}
    .eu-impact-number,.eu-benchmark-row strong{font-variant-numeric:tabular-nums;
       text-align:right;font-size:.91rem}
    .eu-benchmark-group{margin:10px 0 15px}
    .eu-benchmark-group>b{font-size:1rem}
    @media(max-width:600px){.eu-impact-row,.eu-benchmark-row{
       grid-template-columns:minmax(0,1fr) 74px;gap:6px}
       .eu-impact-label,.eu-benchmark-row span{grid-column:1 / -1}
       .eu-impact-track{grid-column:1}.eu-impact-number,.eu-benchmark-row strong{grid-column:2}}
    </style>""",unsafe_allow_html=True)

def _parody_banner():
    """European-only independent satire, safely outside US pricing routes."""
    artwork=Path(__file__).resolve().parent.parent/"assets"/"europe_parody_split_blue_orange.svg"
    if not artwork.is_file():
        return  # Missing decorative art must never block the valuation calculator.
    with st.container(key="eu_parody_banner",border=True):
        st.image(artwork,width="stretch",alt=(
            "Car Price Studio Europa: auffällige eigenständige blau-orange Werbeparodie "
            "mit großer kontrastierender NICHT-Typografie und dem Spruch "
            "'wir kaufen dein Auto NICHT. Wir schätzen es.' "
            "Wortspiel zu Autokauf-Werbung (wirksufendeinaito, "
            "wirkaufendeinauto/WKDA, Autohero, AUTO1), ohne geschäftliche Verbindung."
        ))
        st.markdown('<div class="eu-parody-tagline">🏁 NULL KAUFINTERESSE. '
                    '100 % WERTSCHÄTZUNG. 🇪🇺</div>',unsafe_allow_html=True)
        st.caption("Satirische Eigenwerbung. Car Price Studio ist unabhängig "
                   "und nicht mit WKDA, Autohero oder AUTO1 verbunden.")

def _header():
    _parody_banner()
    local_headline, local_invitation = editorial_voice()
    st.markdown("### "+escape(local_headline))
    st.caption(local_invitation)
    st.caption("🌐 "+tr("market_intro"))
    st.markdown("""<section class="eu-hero" aria-label="European automotive price studio">
      <div class="eu-kicker">🇪🇺 CAR PRICE STUDIO · EUROPEAN EDITION</div>
      <h1>From Hamburg to Rome. Same love of cars.<br>Less guessing. More evidence.</h1>
      <p>Research-backed used-car <b>asking-price estimates</b> for supported European
         markets. No fictional auction bids. No registration. No advertising.
         Our European model is evaluated against held-out asking prices; market-time adjustment is a separate, transparent research layer.</p>
      <div class="eu-tags"><span>🇩🇪 Built independently in Germany</span>
        <span>€ Advertised Ask, not sale price</span><span>📊 Real what-if estimates</span>
        <span>🛡️ Worst-error risk measured</span></div></section>""",
        unsafe_allow_html=True)

def _bundle():
    try:
        return load_bundle(),None
    except (EuropeanModelUnavailable,ValueError,OSError,RuntimeError) as exc:
        return None,str(exc)

@st.cache_data(show_spinner=False,ttl=7200)
def _market_table():
    file=HOFM3_ROOT/"market_stats.csv.gz"
    if not file.is_file():file=ROOT/"market_stats.csv.gz"
    if not file.is_file():
        return pd.DataFrame()
    return pd.read_csv(file,compression="gzip",
                       dtype={"seller_country":str,"make":str,"model":str})

def _snapshot(bundle):
    return bundle[1].get("snapshot_date","2026-11-01") if bundle else "2026-11-01"

def _choices(schema):
    vocab=schema["category_vocabulary"] if schema else {}
    def values(key,default):
        names=sorted(v for v in vocab.get(key,{}) if v!="__missing__")
        return names or default
    return values

def _edit_vehicle(schema):
    """Persistent European facts; no US mile/state fields or model assumptions."""
    values=_choices(schema)
    st.markdown("### 🚘 01 / "+tr("form"))
    st.caption("European conventions: km, horsepower, registration year and euro asking prices. "
               "Unknown specifications stay unknown.")
    st.info("📅 **October 2026 observed market reference** · The selected "
            "HOFM3 was trained on asking prices verified 4–15 October 2026 "
            "(median 13 October). The separately generated November-indexed "
            "column was **not** its training target. No additional Eurostat "
            "adjustment is applied. Monthly market trends remain in the "
            "research panel.")
    with st.form("eu_ask_form",clear_on_submit=False):
        a,b,c=st.columns(3)
        country=a.selectbox(tr("country")+" · 8",COUNTRY_OPTIONS,
            format_func=lambda v:COUNTRIES[v.upper()],key="eu_country_v2")
        makes=values("make",["volkswagen","bmw","mercedes benz","toyota","renault"])
        preferred="volkswagen" if "volkswagen" in makes else makes[0]
        make=b.selectbox(tr("make"),makes,index=makes.index(preferred),
                         key="eu_make")
        model=c.text_input(tr("model"),value="golf" if make=="volkswagen" else "",
                           key="eu_model",max_chars=80,
                           help="Enter the exact model as shown in a European advert.")
        a,b,c=st.columns(3)
        registration=a.number_input(tr("year"),
                        min_value=1950,max_value=2026,value=2020,step=1,
                        key="eu_reg_year")
        month=b.selectbox(tr("month"),list(range(1,13)),
                          index=5,key="eu_reg_month")
        km=c.number_input(tr("mileage"),min_value=0,max_value=2_000_000,
                          value=85000,step=1000,key="eu_km")
        a,b,c=st.columns(3)
        power=a.number_input(tr("power"),min_value=20,max_value=1200,
                             value=110,key="eu_hp")
        fuel=b.selectbox(tr("fuel"),values("fuel",["petrol","diesel","electric","hybrid"]),
                         key="eu_fuel")
        transmission=c.selectbox(tr("transmission"),values("transmission",["manual","automatic"]),
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
        submitted=st.form_submit_button("🇪🇺 "+tr("calculate"),
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
                snapshot=_snapshot((None,schema))
                st.session_state["eu_receipt"]={
                    "vehicle":vehicle,"price_eur":price,"base_price_eur":price,
                    "snapshot":snapshot,"selected_date":snapshot,
                    "clock":{"status":"built_in_recent_market","factor":1.0},
                    "market":"Europe","engine":"HOFM3 Tail-MAE · October 2026 evidence",
                }
                st.success("HOFM3 valuation calculated. No second date adjustment; "
                           "advertised ASK, not a completed sale.")
            except (ValueError,EuropeanModelUnavailable,RuntimeError) as exc:
                st.error("Valuation unavailable: "+str(exc))
    return vehicle

def _receipt(bundle):
    quote=st.session_state.get("eu_receipt")
    if quote and quote.get("engine")!="HOFM3 Tail-MAE · October 2026 evidence":
        st.session_state.pop("eu_receipt",None)
        quote=None
        st.info("European neural model upgraded. Recalculate to refresh your receipt.")
    if not quote:
        st.info("Your European asking-price receipt will appear here after valuation.")
        return None
    st.markdown('<div class="eu-result"><small>EUROPEAN ADVERTISED ASK · EUR</small>'
                '<strong>'+escape(money(quote["price_eur"]))+'</strong>'
                '<p>HOFM3 · Tail-MAE · October 2026 observed ASK</p>'
                '</div>',unsafe_allow_html=True)
    st.caption("Trained on observed October 4–15, 2026 asking prices "
               "(reference October 13). **No additional Eurostat factor** "
               "is applied. This is not a verified November 2026 "
               "market price or completed sale; mileage is your input.")
    pilot=lookup_pulse(read_pulse(),quote["vehicle"].get("seller_country"),
                       quote["vehicle"].get("make"),quote["vehicle"].get("model"))
    if pilot:
        with st.expander("🧪 "+quote["vehicle"]["make"].title()+" "+
                         quote["vehicle"]["model"].title()+
                         " · real listing pulse (research only)"):
            st.metric("Relative move vs German matched-listing basket · "+
                      pilot["early_month"]+" → "+pilot["late_month"],
                      f'{pilot["relative_pct"]:+.2f}%',
                      help="Robust quality-adjusted Huber residual contrast, shrunk "
                           "towards zero. Only a one-month research observation.")
            st.caption("Sample: "+str(pilot["n_early"])+" earlier and "
                       +str(pilot["n_late"])+" later adverts; approximate "
                       "interval "+f'{pilot["interval_low_pct"]:+.2f}%'+" to "+
                       f'{pilot["interval_high_pct"]:+.2f}%'+
                       ". **Not applied** to your price. A two-month comparison "
                       "is not a validated make/model price index, and cannot be "
                       "extrapolated into future months.")
    st.caption("An advertised asking-price estimate, not evidence of a completed sale "
               "or a customs, insurance, taxation or financing decision.")
    test=bundle[2]["overall_champion"]["test_once"] if bundle else {}
    if test:
        a,b,c=st.columns(3)
        a.metric("Sealed-test MAE",money(test["mae_eur"]))
        b.metric("Worst 5% · CVaR95",money(test["cvar95_absolute_eur"]))
        c.metric("Median absolute % error",f'{test["median_ape_pct"]:.2f}%')
        st.caption("These are dataset-level errors, not a personal confidence interval.")
    if st.button("🔍 What moves this price? · show real what-if graphics",
                 key="eu_explain_click",use_container_width=True):
        try:
            factors=explain(quote["vehicle"],bundle)
            if factors:
                magnitude=max(1.,max(abs(change) for _,change in factors))
                rows=[]
                for label,change in factors:
                    width=0 if abs(change)<1 else max(2.,100.*abs(change)/magnitude)
                    sign="+" if change>0 else "−" if change<0 else "±"
                    tone="eu-impact-up" if change>=0 else "eu-impact-down"
                    rows.append(
                        '<div class="eu-impact-row"><span class="eu-impact-label">'+escape(label)+'</span>'
                        '<div class="eu-impact-track"><div class="eu-impact-bar '+tone+
                        '" style="width:'+f"{width:.1f}"+'%"></div></div>'
                        '<strong class="eu-impact-number">'+sign+escape(money(abs(change)))+'</strong></div>'
                    )
                st.markdown('<div class="eu-impact-graph" role="img" '
                            'aria-label="Hypothetical one-variable price changes, shown in euros">'
                            +''.join(rows)+'</div>',unsafe_allow_html=True)
                st.caption("Each bar is a real model counterfactual with one detail changed; "
                           "no extra market-index factor is applied. "
                           "Positive means a higher predicted asking price. "
                           "These are hypothetical model sensitivities, not causal effects or SHAP attribution.")
            else:
                st.info("No supported variations are available for this vehicle.")
        except (ValueError,RuntimeError) as exc:
            st.warning("Could not calculate what-if estimates: "+str(exc))
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

def _how_it_works():
    """A compact, responsive graphic designed for ordinary car owners."""
    st.markdown("""<div class="eu-steps" aria-label="How the European ASK valuation works">
       <div class="eu-step"><b>① Tell us about the car</b><span>Model, age,
         kilometres, power, country and optional equipment</span></div>
       <div class="eu-arrow" aria-hidden="true">›</div>
       <div class="eu-step"><b>② Machine learning, not guesswork</b><span>Vehicle features
         are compared with documented asking-price patterns</span></div>
       <div class="eu-arrow" aria-hidden="true">›</div>
       <div class="eu-step"><b>③ Your independent price view</b><span>Euro ASK-price
         estimate. We never buy your car, so we have no purchase offer to push.</span></div>
    </div>""",unsafe_allow_html=True)
    st.caption("This model uses structured car facts, not advert prose, photos or VINs. "
               "Its target is the advertised price, never a verified completed sale.")


def _valuation_page(bundle,error):
    st.markdown("### 🧮 "+tr("calculate"))
    _how_it_works()
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
    st.caption("A change in vehicle specifications, not a calendar-time forecast. "
               "No additional Eurostat multiplier. Changing registration year is "
               "NOT equivalent to projecting the actual car's depreciation.")

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
            "model":"HOFM3 Tail-MAE · fresh European ASK CVaR95 winner",
            "vehicle":{k:v for k,v in quote["vehicle"].items()
                       if k not in ("seller_city","version")},
            "limitations":"Not a sale price, validated personal interval, customs ruling, or fraud finding"}
    st.download_button("Download transparent research report · JSON",
        json.dumps(report,indent=2),file_name="european-ask-evidence.json",
        mime="application/json",key="eu_forensic_json")

def _evaluation(bundle):
    st.markdown("### 📉 European evaluation · what improved?")
    if bundle is None:
        st.info("Verified research metrics become available with the released weights.")
        return
    comp=bundle[2]
    a,b,c=st.columns(3)
    a.metric("Training adverts",f'{comp["training_rows"]:,}')
    b.metric("Validation adverts",f'{comp["validation_rows"]:,}')
    c.metric("Sealed test adverts",f'{comp["test_rows"]:,}')
    champion=comp["overall_champion"]
    data=[{"Cohort":name,**{METRIC_LABELS[k]:round(metric[k],2) for k in EVAL_FIELDS}}
        for name,metric in [("Validation",champion["validation"]),
                            ("Untouched test",champion["test_once"])]]
    st.dataframe(pd.DataFrame(data),hide_index=True,use_container_width=True)
    old_path=ROOT/"comparison.json"
    if old_path.is_file():
        old=json.loads(old_path.read_text(encoding="utf-8"))
        previous=old["winning_model"]
        # Both held-out groups were computed on exactly the same frozen dataset.
        if old["counts"]["test"]==comp["test_rows"] and old["counts"]["validation"]==comp["validation_rows"]:
            chart=pd.DataFrame({
                "New HOFM3":[champion["validation"]["cvar95_absolute_eur"],
                              champion["test_once"]["cvar95_absolute_eur"]],
                "Previous XGBoost":[previous["validation"]["cvar95_absolute_eur"],
                                    previous["test_once"]["cvar95_absolute_eur"]],
            },index=["Validation","Sealed test"])
            st.markdown("#### Head-to-head: costliest 5% of valuation errors")
            max_error=max(float(chart.max().max()),1.)
            bars=[]
            for name in chart.index:
                bars.append('<div class="eu-benchmark-group"><b>'+escape(str(name))+'</b>')
                for label in ("New HOFM3","Previous XGBoost"):
                    value=float(chart.loc[name,label])
                    tone="eu-impact-up" if label=="New HOFM3" else "eu-impact-reference"
                    width=100.*value/max_error
                    bars.append('<div class="eu-benchmark-row"><span>'+escape(label)+'</span>'
                        '<div class="eu-impact-track"><div class="eu-impact-bar '+tone+
                        '" style="width:'+f"{width:.1f}"+'%"></div></div>'
                        '<strong>'+escape(money(value))+'</strong></div>')
                bars.append('</div>')
            st.markdown('<div class="eu-benchmark" role="img" '
                        'aria-label="Validation and sealed-test worst-five-percent error comparison">'
                        +''.join(bars)+'</div>',unsafe_allow_html=True)
            new=champion["test_once"]["cvar95_absolute_eur"]
            old_error=previous["test_once"]["cvar95_absolute_eur"]
            st.success(f"The new neural engine reduced worst-5% average test errors by "
                       f"{(old_error-new)/old_error*100:.1f}% versus the previous XGBoost model.")
    st.caption("CVaR95 is the mean absolute error in the worst 5% of examples. "
               "The candidate was selected on validation; the test was scored once.")
    st.warning("Scores differ by model, market and car type. They do not promise "
               "this error for any single car or establish a transaction value.")
    st.download_button("Download complete neural research comparison · JSON",
        json.dumps(comp,indent=2),file_name="european-neural-comparison.json",
        mime="application/json",key="eu_eval_json")


def _model_lab(bundle):
    st.markdown("### 🧪 Ray Tune model lab · five neural architectures")
    st.caption("One November-2026 refresh: 15 Ray workers, five architectures and "
               "30 loss functions. Rankings use CVaR95 on genuine recent asking prices.")
    if bundle is None:
        st.info("Model research details unlock when verified weights are available.")
        return
    comp=bundle[2]
    rows=[]
    for candidate in comp["neural_leaderboard"]:
        score=candidate["validation"]
        rows.append({"Neural architecture":candidate["architecture"],
                     "Loss":candidate["loss"],
                     "CVaR95 · €":round(score["cvar95_absolute_eur"]),
                     "MAE · €":round(score["mae_eur"]),
                     "Worker":candidate["origin"]})
    st.dataframe(pd.DataFrame(rows).sort_values("CVaR95 · €"),
                 hide_index=True,use_container_width=True)
    with st.expander("Exact winning hyperparameters"):
        st.json(comp["overall_champion"]["config"])
    st.caption("192,149 validation listings; 192,046 sealed test listings. "
               "Recent-country ISO-2 observations only.")


def _science(bundle):
    st.markdown("### 🔬 Under the bonnet · a neural engine, without the jargon")
    _how_it_works()
    st.markdown("""**Why a neural network?** A used car's features interact. An extra
    20,000 km may mean something quite different for a young electric hatchback
    and an older high-performance saloon. HOFM3 learns these combinations
    from real European listing patterns, rather than assuming a fixed deduction.

    **What is HOFM3?** Higher-Order Factorization Machines combine embeddings,
    pairwise and third-order interactions with a residual network. This
    winner has two residual blocks and 32-dimensional embeddings.

    **Why focus on the worst cases?** Big mistakes can be expensive even when
    the average estimate looks good. The Tail-MAE objective and validation
    CVaR95 selection favour models with smaller extreme euro mistakes.

    **What did we train on?** Approximately two million European car ads.
    1,536,801 October-verified adverts were used for training; 192,149 for
    validation and 192,046 for sealed testing. Older adverts were excluded,
    seller countries canonicalized, and forecast-generated November prices
    were not model labels.

    **What remains unknown?** The model does not read seller descriptions,
    VINs or photographs. It does not inspect the vehicle or know the final
    negotiated amount. Asking prices are not sales prices, customs decisions
    or tax valuations.""")
    st.markdown("**Dataset and usage:** [European Used Car Market]("+DATASET+
        ") · CC BY-NC-SA 4.0. Car Price Studio software is separately MIT-licensed.")
    if bundle:
        winner=bundle[2]["overall_champion"]
        st.info(f"Selected: HOFM3 with Tail-MAE; sealed-test MAE "
                f"{money(winner['test_once']['mae_eur'])}; worst-5% average "
                f"error {money(winner['test_once']['cvar95_absolute_eur'])}.")


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



def _eurostat_index_panel():
    st.markdown("#### 🇪🇺 Eurostat · second-hand car price clock")
    st.caption("Official monthly HICP for second-hand motor cars (ECOICOP **CP07112**), "
               "base 2025=100. This is a consumer-price **market proxy**, not "
               "our own hedonic Fisher index of asking-price advertisements.")
    source=read_eurostat()
    if source.get("status")!="validated" or not source.get("countries"):
        st.warning(source.get("message") or
                   "No verified Eurostat used-car monthly index is packaged.")
        return
    countries=source["countries"]
    country=st.selectbox("National second-hand-car index",sorted(countries),
                         format_func=lambda x:COUNTRIES.get(x,x),
                         key="eu_hicp_country")
    record=countries[country]
    rows=pd.DataFrame(record.get("months",[]))
    if rows.empty or not {"month","index"}.issubset(rows.columns):
        st.warning("The published index has no usable monthly observations.")
        return
    rows=rows.sort_values("month").copy()
    last=rows.iloc[-1]
    earlier=rows.iloc[-13] if len(rows)>=13 else rows.iloc[0]
    a,b,c=st.columns(3)
    a.metric("Latest published month",str(last["month"]))
    b.metric("Official used-car HICP",f'{float(last["index"]):.2f}')
    c.metric("Change vs 12 months ago" if len(rows)>=13 else "Change over series",
             f'{100*(float(last["index"])/float(earlier["index"])-1):+.2f}%')
    st.caption("Official observations end at **"+str(last["month"])+"**. "
               "Later dates are explicitly forecasts, never official data.")
    horizon=st.slider("Damped Holt horizon · months",1,12,6,key="eu_hicp_horizon")
    import plotly.graph_objects as go
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=rows["month"],y=rows["index"],mode="lines",
                             name="Eurostat · published",line=dict(color="#086c9c",width=3)))
    last_index=month_number(str(last["month"]))
    projected=[(month_label(last_index+i),
                eurostat_level_at(source,country,month_label(last_index+i)))
               for i in range(1,horizon+1)]
    scenarios=[(m,p) for m,p in projected if p.get("method")=="holt_forecast"]
    if scenarios:
        fig.add_trace(go.Scatter(
            x=[str(last["month"])]+[m for m,_ in scenarios],
            y=[float(last["index"])]+[p["level"] for _,p in scenarios],
            mode="lines+markers",name="Damped Holt · forecast",
            line=dict(color="#d6882e",dash="dash",width=2)))
        fig.add_trace(go.Scatter(
            x=[m for m,_ in scenarios]+[m for m,_ in reversed(scenarios)],
            y=[p["upper"] for _,p in scenarios]
              +[p["lower"] for _,p in reversed(scenarios)],
            fill="toself",fillcolor="rgba(214,136,46,0.13)",
            line=dict(width=0),hoverinfo="skip",name="Trend-model band"))
    fig.update_layout(title=f"Second-hand car prices · {COUNTRIES.get(country,country)}",
                      xaxis_title="Month",yaxis_title="HICP · 2025=100",
                      height=410,legend=dict(orientation="h"),
                      margin=dict(l=10,r=10,t=54,b=10))
    st.plotly_chart(fig,use_container_width=True)
    st.caption("Blue: published Eurostat monthly consumer price index. "
               "Orange: modelled, unobserved damped-Holt scenario. "
               "Shaded area: trend-model diagnostics, NOT individual "
               "vehicle-price confidence intervals.")
    st.download_button("⬇️ Download Eurostat published monthly series (CSV)",
                       rows.to_csv(index=False).encode("utf-8"),
                       file_name="eurostat_second_hand_car_"+country+".csv",
                       mime="text/csv",key="eu_hicp_download")
    with st.expander("📚 Official source, coverage and economic meaning"):
        st.markdown("Source: [Eurostat HICP · ECOICOP version 2]"
                    "(https://ec.europa.eu/eurostat/databrowser/view/"
                    "prc_hicp_minr/default/table?lang=en). "
                    "Dataset \`prc_hicp_minr\`, code \`CP07112\`, unit \`I25\` "
                    "(2025=100). This measures national consumer second-hand car "
                    "price changes. It is an interim **proxy** for changes "
                    "in classified advertisement ASK prices.")
        st.dataframe(rows,use_container_width=True,hide_index=True)


def _model_pulse_panel():
    """Distinct make/model observation, not an official price index."""
    st.markdown("#### 🚘 Make & Model Pulse · experimental listing evidence")
    st.caption("Two distinct sources, two levels of evidence: **years of "
               "observed monthly Eurostat country indices**, plus a short "
               "make/model listing comparison. Bayesian residual state-space "
               "scenarios can connect them, but historical model deviations "
               "outside the listing window are **inferred, not observed**. "
               "They never alter production customer valuations.")
    research=read_pulse()
    pairs=research.get("research_pairs",[])
    if not pairs:
        st.info(research.get("message") or
                "No well-supported make/model months passed the sample and "
                "robustness gates. National Eurostat remains the price clock.")
        return
    frame=pd.DataFrame(pairs[0].get("models",[]))
    if frame.empty:
        st.info("No make/model cohort met the minimum sample-size and fit checks.")
        return
    pair=pairs[0]
    st.caption("Country: **"+COUNTRIES.get(pair["country"],pair["country"])+"** · "
               "Observation: **"+pair["early_month"]+" → "+pair["late_month"]+
               "** · "+str(pair["n_training"])+" training listings · "
               +str(pair["model_count"])+" supported model groups.")
    best=frame.sort_values("n_early",ascending=False)
    makes=sorted(frame.make.unique())
    make=st.selectbox("Manufacturer",makes,
                      key="eu_model_pulse_make",
                      format_func=lambda x:x.title())
    eligible=frame[frame.make==make].sort_values("n_early",ascending=False)
    model=st.selectbox("Model",eligible.model.tolist(),
                       key="eu_model_pulse_model",
                       format_func=lambda x:x.title())
    row=eligible[eligible.model==model].iloc[0]
    a,b,c=st.columns(3)
    a.metric("Relative movement vs German matched-listing basket",
             f'{float(row["relative_pct"]):+.2f}%',
             help="Positive: model's quality-adjusted residual rose faster than "
                  "the matched listing basket. Negative: slower.")
    b.metric("Earlier-month adverts",int(row["n_early"]))
    c.metric("Later-month adverts",int(row["n_late"]))
    st.caption("Approximate 95% Bayesian conditional credible interval: "
               +f'{float(row["interval_low_pct"]):+.2f}%'+" to "
               +f'{float(row["interval_high_pct"]):+.2f}%'+
               ". This excludes selection and time-series uncertainty.")
    if "posterior_probability_positive" in row:
        st.metric("Posterior probability this model outpaced the reference",
                  f'{100*float(row["posterior_probability_positive"]):.1f}%',
                  help="Conditional on the fixed hierarchical Gaussian priors "
                       "and approximate robust standard error. Not an observed "
                       "market probability or a price forecasting accuracy score.")
        st.caption("Same-make reference models: "
                   +str(int(row.get("brand_peer_count",0)))+
                   ". More evidence means less shrinkage. This single-month "
                   "comparison is never fed into live valuations.")
    subset=frame.sort_values("relative_pct").copy()
    subset["Model"]=subset["make"].str.title()+" · "+subset["model"].str.title()
    subset=subset.nlargest(15,"n_early").sort_values("relative_pct")
    st.bar_chart(subset.set_index("Model")["relative_pct"],horizontal=True)
    _model_state_space_visual(research,pair,make,model)
    with st.expander("📚 Why these figures are not applied to valuations"):
        st.markdown("A pooled **SGDRegressor with Huber loss on log(EUR ASK)** "
                    "controls for model, age, mileage, power, fuel, gearbox, "
                    "body type and advertising portal. Robust residuals are "
                    "compared between **actual observed listing months**. "
                    "The model-level difference is relative to the wider "
                    "matched German listing basket and shrunk using **two-level "
                    "Gaussian Bayesian partial pooling**: the same manufacturer "
                    "and a country-centred prior. The approximate conditional "
                    "credible interval and posterior probability omit some "
                    "uncertainties and are not calibrated time-series predictions.")
        st.markdown("It is **not** a model-specific Fisher index, **not** a "
                    "completed-sales series, **not** proof of causality, and "
                    "not an independent forecast. We need at least six "
                    "consecutive observed monthly cohorts and out-of-time "
                    "validation before using make/model market-time factors "
                    "for prices. Eurostat remains the separate national "
                    "market-time adjustment.")
        st.dataframe(frame.sort_values(["make","model"]),
                     hide_index=True,use_container_width=True)
    st.download_button("⬇️ Download experimental model-level research (CSV)",
                       frame.to_csv(index=False).encode("utf-8"),
                       file_name="europe_make_model_pulse_research.csv",
                       mime="text/csv",key="eu_model_pulse_download")


def _model_state_space_visual(research,pair,make,model):
    """National observed HICP history plus Bayesian model-residual *scenarios*."""
    st.markdown("##### 🔭 Bayesian Time Machine · country index + model residual")
    st.caption("The **blue national path** comes from published Eurostat observations "
               "(with a dashed segment for unpublished future months). "
               "The **purple make/model path** is a conditional Bayesian AR(1) "
               "scenario. Its historical model-specific trajectory is **not "
               "observed**. Both paths are normalized to 100 in the first "
               "month of the model listing comparison.")
    a,b=st.columns(2)
    backcast=a.select_slider("National history and model backcast · months",
                             options=[12,24,36,48,60,72],value=24,
                             key="eu_state_space_backcast")
    horizon=b.slider("Future Bayesian scenario · months",0,12,6,
                     key="eu_state_space_horizon")
    pilot=lookup_pulse(research,pair["country"],make,model)
    outcome=model_scenario_series(read_eurostat(),pilot,
                                  lookback_months=backcast,
                                  forward_months=horizon)
    if outcome.get("status")!="research_only":
        st.warning("Bayesian model scenario unavailable: "
                   +outcome.get("reason","No qualifying evidence."))
        return
    frame=pd.DataFrame(outcome["points"])
    observed=frame[frame["national_data_method"]=="observed"]
    forecast=frame[frame["national_data_method"]!="observed"]
    col1,col2,col3=st.columns(3)
    source=read_eurostat()
    col1.metric("Published national months",
                len(source["countries"][pair["country"]]["months"]))
    col2.metric("Observed model residual transitions",1)
    col3.metric("Model-relative prior volatility / month",
                f'{100*outcome["prior_stationary_sd_monthly_log"]:.1f}%',
                help="Assumed monthly stationary standard deviation of "
                     "the latent make/model log-growth differential. "
                     "It is NOT estimated from decades of make/model observations.")
    import plotly.graph_objects as go
    fig=go.Figure()
    # Band is conditional on the chosen state-space prior and sparse evidence;
    # it DOES NOT include the national Holt forecast uncertainty.
    fig.add_trace(go.Scatter(
        x=frame["month"].tolist()+frame["month"].iloc[::-1].tolist(),
        y=frame["model_scenario_upper"].tolist()+
          frame["model_scenario_lower"].iloc[::-1].tolist(),
        fill="toself",fillcolor="rgba(123,79,176,.12)",
        mode="lines",line=dict(width=0),hoverinfo="skip",
        name="Conditional model-residual band · not calibrated"))
    fig.add_trace(go.Scatter(
        x=frame["month"],y=frame["model_scenario_median"],
        mode="lines",name="Bayesian model scenario · inferred",
        line=dict(color="#8051ae",width=3,dash="dot")))
    if not observed.empty:
        fig.add_trace(go.Scatter(
            x=observed["month"],y=observed["national_index"],
            mode="lines",name="Eurostat · published country index",
            line=dict(color="#12699a",width=3)))
    if not forecast.empty:
        joined=pd.concat([observed.tail(1),forecast],ignore_index=True)
        fig.add_trace(go.Scatter(
            x=joined["month"],y=joined["national_index"],
            mode="lines",name="Country Holt scenario · unpublished",
            line=dict(color="#12699a",width=2,dash="dash")))
    transition=frame[frame["month"]==outcome["one_model_observation_month"]]
    if not transition.empty:
        fig.add_trace(go.Scatter(
            x=transition["month"],y=transition["model_scenario_median"],
            mode="markers",name="One month with direct model-listing evidence",
            marker=dict(size=10,color="#8c5db6",symbol="diamond")))
    fig.update_layout(
        title=make.title()+" "+model.title()+
              " · inferred residual path versus "+COUNTRIES.get(pair["country"],
              pair["country"])+" second-hand-car HICP",
        xaxis_title="Month",
        yaxis_title="Market index · 100 in "+outcome["anchor_month"],
        height=445,margin=dict(l=10,r=15,t=64,b=12),
        legend=dict(orientation="h",y=-.28))
    st.plotly_chart(fig,use_container_width=True)
    st.caption("**Actual evidence:** "+outcome["national_first_published_month"]+
               "–"+outcome["national_last_published_month"]+
               " Eurostat monthly country values; one relative model-listing "
               "contrast from "+outcome["anchor_month"]+" to "+
               outcome["one_model_observation_month"]+". The other purple "
               "points are **posterior backcasts/forecasts**, not observations. "
               "The purple range is **not a calibrated car-price interval**.")
    st.info("**Why the two paths differ:** the state-space model assumes that "
            "the model's quality-adjusted growth can differ temporarily from "
            "the national Eurostat trend. It also assumes that the German "
            "matched-listing benchmark is comparable to the Eurostat market. "
            "That transfer assumption has **not been validated**. "
            "Calendar-time market movement is separate from vehicle ageing "
            "or extra mileage, and the scenario is **never used in pricing**.")
    with st.expander("🧬 The Bayesian equations and limitations"):
        st.latex(r"\Delta\log I_{m,t}=\Delta\log HICP_{c,t}+\delta_{m,t}")
        st.latex(r"\delta_{m,t}=\phi\delta_{m,t-1}+\epsilon_{m,t}")
        st.markdown("A **stationary, mean-zero Gaussian AR(1) prior** (assumed "
                    "monthly relative volatility of 3.5%, persistence 0.65) "
                    "is conditioned on just one noisy, partially pooled "
                    "SGD–Huber model residual change. The historical national "
                    "HICP values are real; historical make/model residuals "
                    "are regularized, **prior-driven reconstructions**. "
                    "This is a fast **two-stage empirical-Bayes approximation**, "
                    "not a fully learned multiyear Bayesian residual process.")
        st.caption(outcome["uncertainty_warning"])
        st.caption(outcome["depreciation_warning"])
    st.download_button(
        "⬇️ Export observed national index + inferred model scenario · CSV",
        frame.to_csv(index=False).encode("utf-8"),
        file_name="europe_bayesian_model_scenario_"+pair["country"]+"_"+
                  make.replace(" ","_")+"_"+model.replace(" ","_")+".csv",
        mime="text/csv",key="eu_state_space_csv")

def _price_indices():
    """European price clock: empirical coverage first, no fabricated time series."""
    st.markdown("### 📈 European Price Index Observatory")
    _eurostat_index_panel()
    st.divider()
    _model_pulse_panel()
    st.divider()
    st.markdown("#### 🧪 Hedonic Fisher · independent listing research")
    st.caption("The market has a pulse. We measure it without confusing more "
               "expensive cars on offer with an actual increase in like-for-like prices.")
    one,two,three,four=st.columns(4)
    one.metric("Target", "Log € ASK")
    two.metric("Robust fit", "Huber")
    three.metric("Price formula", "Fisher")
    four.metric("Forward scenarios", "Damped Holt")
    st.markdown("**How the engine works:** monthly ads → robust hedonic regressions "
                "on log prices → reprice the same vehicle baskets at both dates → "
                "Laspeyres × Paasche (geometric mean) → monthly Fisher chain → "
                "log interpolation or damped Holt beyond sample.")
    st.info("A date on an advert is not a historical transaction. The index uses "
            "advertised ASK prices, adjusts for vehicle-mix differences and is "
            "independent of the vehicle's depreciation.")
    release=read_index()
    coverage=pd.DataFrame(release.get("coverage",[]))
    if not coverage.empty and {"country","month","n"}.issubset(coverage.columns):
        with st.expander("📦 Verified source-date coverage",expanded=False):
            country_totals=coverage.groupby("country",as_index=False)["n"].sum()
            st.bar_chart(country_totals.set_index("country"))
            st.dataframe(coverage.sort_values(["country","month"]),
                         use_container_width=True,hide_index=True)
            st.caption("These bars count dated listings, not the number of "
                       "independently verified market transactions.")
    countries=release.get("countries",{})
    if release.get("status")!="validated" or not countries:
        st.warning("No independently validated multi-month index is installed. "
                   +release.get("message","The build requires repeated genuine "
                                          "observation months."))
        pilots=release.get("pilot_countries",{})
        if pilots:
            st.markdown("#### 🧪 The pilot garage · actual observations, not a live index")
            selected_pilot=st.selectbox(
                "Research country", sorted(pilots),key="eu_index_pilot_country",
                format_func=lambda x:COUNTRIES.get(x,x))
            p=pd.DataFrame(pilots[selected_pilot].get("months",[]))
            if not p.empty and {"month","index","n"}.issubset(p):
                p=p.sort_values("month")
                st.line_chart(p.set_index("month")["index"])
                change=100*(float(p["index"].iloc[-1])/float(p["index"].iloc[0])-1)
                st.metric("Preliminary common-basket Fisher movement",f"{change:+.2f}%")
                st.dataframe(p,use_container_width=True,hide_index=True)
                st.caption("EXPERIMENTAL pilot only. This result comes from too few "
                           "independent months for a validated price index or "
                           "Holt forecast. It never adjusts a customer's quote.")
        else:
            st.caption("The October 2026 source snapshot alone does not establish "
                       "a European historical monthly price index. We need more "
                       "genuine independent months to train and validate Holt.")
        return
    selected=st.selectbox("🌍 Country index",sorted(countries),
                          format_func=lambda x:COUNTRIES.get(x,x),
                          key="eu_index_country")
    rows=countries[selected].get("months",[])
    frame=pd.DataFrame(rows)
    if frame.empty or not {"month","index","n"}.issubset(frame):
        st.warning("This country's published index has no valid chart data.")
        return
    frame=frame.sort_values("month").copy()
    a,b,c=st.columns(3)
    a.metric("Observed months",str(len(frame)))
    b.metric("Latest quality-adjusted index",f'{frame["index"].iloc[-1]:.1f}',
             help="First supported month = 100. Does not mean 100 euros.")
    movement=100*(float(frame["index"].iloc[-1])/float(frame["index"].iloc[0])-1)
    c.metric("Period movement",f"{movement:+.2f}%")
    import plotly.graph_objects as go
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=frame["month"],y=frame["index"],name="Observed Fisher",
                             mode="lines+markers",line=dict(color="#136f74",width=3)))
    horizon=st.slider("Holt scenario horizon · months",1,12,6,key="eu_holt_horizon")
    last=month_number(frame["month"].iloc[-1])
    projected=[level_at(release,selected,month_label(last+n))
               for n in range(1,horizon+1)]
    forecast_points=[(month_label(last+n),point)
                     for n,point in enumerate(projected,1)
                     if point.get("method")=="holt_forecast"]
    if forecast_points:
        fx=[frame["month"].iloc[-1]]+[m for m,_ in forecast_points]
        fy=[float(frame["index"].iloc[-1])]+[p["level"] for _,p in forecast_points]
        fig.add_trace(go.Scatter(x=fx,y=fy,name="Holt scenario (NOT observed)",
                                 mode="lines+markers",
                                 line=dict(color="#d38d32",dash="dash",width=2)))
        fig.add_trace(go.Scatter(
            x=[m for m,_ in forecast_points]+[m for m,_ in forecast_points[::-1]],
            y=[p["upper"] for _,p in forecast_points]
              +[p["lower"] for _,p in forecast_points[::-1]],
            fill="toself",fillcolor="rgba(211,141,50,0.12)",
            line=dict(width=0),hoverinfo="skip",name="Trend-model uncertainty"))
    fig.update_layout(title="Quality-adjusted ASK index · 100 at first observed month",
                      xaxis_title="Listing month",yaxis_title="Hedonic Fisher index",
                      legend=dict(orientation="h"),margin=dict(l=10,r=10,t=54,b=10),
                      height=410)
    st.plotly_chart(fig,use_container_width=True)
    if forecast_points:
        st.caption("Orange: damped Holt forecast in log-index space, up to 12 months "
                   "from a minimum of 6 consecutive observed months. Shading is "
                   "a trend-model diagnostic, NOT a vehicle-valuation interval.")
    st.dataframe(frame.rename(columns={"month":"Observed month",
                                        "index":"Quality-adjusted index",
                                        "n":"Listing count",
                                        "overlap":"Matched model groups"}),
                 use_container_width=True,hide_index=True)
    st.download_button("⬇️ Download published monthly index (CSV)",
                       frame.to_csv(index=False).encode("utf-8"),
                       file_name="europe_hedonic_fisher_"+selected+".csv",
                       mime="text/csv",key="eu_index_download")
    with st.expander("🔬 Methodology · what the Fisher index does and does not say"):
        st.markdown("Huber regression minimizes a robust loss on **log advertised "
                    "prices**. Both monthly models price the **same overlapping "
                    "vehicle baskets**. Laspeyres uses the earlier-month basket; "
                    "Paasche uses the later-month basket. Their geometric mean "
                    "forms the chained Fisher index. We do not use raw medians as "
                    "the main market index, and we do not claim these are sales.")
        st.markdown("**Gaps:** short gaps are interpolated on the log index. "
                    "**Beyond observations:** a conservatively damped Holt trend "
                    "may project at most 12 months; no unsupported extrapolation "
                    "is called observed data. Long gaps and sparse countries are "
                    "explicitly unavailable.")

def render():
    _css()
    _header()
    from pricing.analytics import render_privacy
    render_privacy()
    bundle,error=_bundle()
    # A labelled, keyboard-operable navigation control is more reliable than
    # 12 wrapping pill buttons, especially on a narrow phone screen.
    tab=st.selectbox(tr("panels"),NAV,
                 format_func=lambda name: ("🚗 "+tr("form")) if name=="🚗 Price my car" else name,
                 key="eu_workspace_selector_v2",
                 help="Choose any European research panel. Your previous ASK receipt persists.")
    if tab=="🚗 Price my car":
        _valuation_page(bundle,error)
    elif tab=="🔎 Listing Inspector":
        _inspector(bundle)
    elif tab=="📉 Value over time":
        _timeline(bundle)
    elif tab=="📈 Price Indices":
        _price_indices()
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
