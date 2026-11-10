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
from pricing.europe_price_clock import adjusted_quote, read_index, level_at, month_number, month_label

DATASET="https://huggingface.co/datasets/KTHESAT/european-used-car-market-2m-listings"
NAV=[
    "🚗 Price my car","🔎 Listing Inspector","📉 Value over time",
    "📈 Price Indices","📊 Evaluation","🧪 Model lab","🔬 The science",
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
        chosen_date=st.date_input(
            "📅 Price estimate date · European market clock",
            value=date.today(),min_value=date(2015,1,1),
            max_value=date(date.today().year+1,12,31),key="eu_valuation_date",
            help="Monthly hedonic Fisher market movement, with log interpolation or "
                 "bounded damped-Holt scenarios where enough genuine monthly data exist. "
                 "A selected date does not guarantee a supported index.")
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
                base_price=estimate(vehicle)
                snapshot=_snapshot((None,schema))
                indexed,clock=adjusted_quote(
                    base_price,snapshot,chosen_date,country)
                price=indexed if indexed is not None else base_price
                st.session_state["eu_receipt"]={
                    "vehicle":vehicle,"price_eur":price,"base_price_eur":base_price,
                    "snapshot":snapshot,"selected_date":chosen_date.isoformat(),
                    "clock":clock,"market":"Europe",
                    "engine":"European asking-price model",
                }
                st.success("Valuation calculated. Advertised ASK, not a completed sale.")
                if indexed is None:
                    st.warning("A valid date-adjusted index is unavailable for this "
                               "country/date. The displayed price remains the "
                               "dataset-era estimate; it is NOT a price for your "
                               "selected date.")
            except (ValueError,EuropeanModelUnavailable,RuntimeError) as exc:
                st.error("Valuation unavailable: "+str(exc))
    return vehicle

def _receipt(bundle):
    quote=st.session_state.get("eu_receipt")
    if quote and quote.get("engine")!="European asking-price model":
        st.session_state.pop("eu_receipt",None)
        quote=None
        st.info("European pricing was upgraded. Recalculate to include the valuation date.")
    if not quote:
        st.info("Your European asking-price receipt will appear here after valuation.")
        return None
    clock=quote.get("clock",{})
    status=clock.get("status","unavailable")
    st.markdown('<div class="eu-result"><small>EUROPEAN ADVERTISED ASK · EUR</small>'
                '<strong>'+escape(money(quote["price_eur"]))+'</strong>'
                '<p>Independent European asking-price model · no auction/Bid component</p>'
                '</div>',unsafe_allow_html=True)
    if status=="unavailable":
        st.warning("⚠️ Date adjustment NOT applied. You selected "
                   +str(quote.get("selected_date","unknown"))+". "
                   +str(clock.get("reason","A validated index is unavailable."))
                   +" The amount shown is the "+str(quote.get("snapshot",""))
                   +" dataset-era estimate.")
    elif status=="snapshot":
        st.info("📅 Dataset-snapshot month · "+str(quote.get("snapshot",""))
                +". No daily market movement is assumed.")
    else:
        change=100*(float(clock["factor"])-1)
        st.metric("Hedonic Fisher market movement · "
                  +clock["base_month"]+" → "+clock["target_month"],
                  f"{change:+.2f}%",
                  help="Quality-adjusted monthly market index movement. It does not "
                       "project additional mileage or vehicle ageing.")
        st.caption("Index status: **"+status+"**. "
                   "Forecasts are damped log-Holt scenarios, not observed prices. "
                   "The index isolates market movement and does not separately "
                   "forecast vehicle depreciation.")
        if "trend_only_range" in clock:
            low,high=clock["trend_only_range"]
            st.caption("Illustrative trend-only forecast band: "
                       +money(low)+"–"+money(high)
                       +". This excludes car-model error and is not a calibrated "
                        "confidence interval for your vehicle.")
    if st.session_state.get("eu_valuation_date") and (
        str(st.session_state["eu_valuation_date"]) != quote.get("selected_date")
    ):
        st.warning("The selected date changed. Press Calculate to refresh this receipt.")
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
                clock_factor=float(quote.get("clock",{}).get("factor",1))
                for label,change in factors:
                    change*=clock_factor
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
                st.caption("Each bar uses the underlying European model after changing one detail; "
                           "a common validated market index factor is then applied. "
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
    st.markdown("### 🧮 European Asking Price Lab")
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
    if quote.get("clock",{}).get("status") not in ("unavailable",None):
        result=result*float(quote["clock"].get("factor",1))
    chart=pd.DataFrame({"Mileage · km":mileage_levels,"Estimated Ask · €":result})
    st.line_chart(chart.set_index("Mileage · km"))
    registration=int(vehicle["reg_year"])
    years=[y for y in range(max(1950,registration-4),
             min(2026,registration+4)+1) if y<=int(_snapshot(bundle)[:4])]
    if years:
        samples=[{**vehicle,"reg_year":y} for y in years]
        estimates=batch_estimate(samples,bundle)
        if quote.get("clock",{}).get("status") not in ("unavailable",None):
            estimates=estimates*float(quote["clock"].get("factor",1))
        st.line_chart(pd.DataFrame({"Registration year":years,
              "Estimated Ask · €":estimates}).set_index("Registration year"))
    st.caption("This is a change in car specifications, not a calendar-time forecast. "
               "When available, the chosen date's quality-adjusted monthly market "
               "factor is applied consistently to each scenario. Changing registration "
               "year is NOT the same as projecting the same car's depreciation.")

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
            "model":"ResMLP Tail-MAE · Ray CVaR95 winner",
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
                "New ResMLP":[champion["validation"]["cvar95_absolute_eur"],
                              champion["test_once"]["cvar95_absolute_eur"]],
                "Previous XGBoost":[previous["validation"]["cvar95_absolute_eur"],
                                    previous["test_once"]["cvar95_absolute_eur"]],
            },index=["Validation","Sealed test"])
            st.markdown("#### Head-to-head: costliest 5% of valuation errors")
            max_error=max(float(chart.max().max()),1.)
            bars=[]
            for name in chart.index:
                bars.append('<div class="eu-benchmark-group"><b>'+escape(str(name))+'</b>')
                for label in ("New ResMLP","Previous XGBoost"):
                    value=float(chart.loc[name,label])
                    tone="eu-impact-up" if label=="New ResMLP" else "eu-impact-reference"
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
    st.caption("Two research waves, 20 neural finalists, 30 objective families and "
               "two XGBoost baselines. Ranking uses validation CVaR95.")
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
    st.caption("The neural shortlist was assessed on a frozen 200,039-advert validation "
               "sample. No unsupported claim that all 30 losses won was made.")


def _science(bundle):
    st.markdown("### 🔬 Under the bonnet · a neural engine, without the jargon")
    _how_it_works()
    st.markdown("""**Why a neural network?** A used car's features interact. An extra
    20,000 km may mean something quite different for a young electric hatchback
    and an older high-performance saloon. ResMLP learns these combinations
    from real European listing patterns, rather than assuming a fixed deduction.

    **What are residual connections?** Small neural layers refine their
    previous answer while keeping useful information from the earlier layer.
    Our winner has two such blocks, and the learned answer becomes an asking
    price in euros.

    **Why focus on the worst cases?** Big mistakes can be expensive even when
    the average estimate looks good. The Tail-MAE objective and validation
    CVaR95 selection favour models with smaller extreme euro mistakes.

    **What did we train on?** Approximately two million European car ads.
    1,602,247 eligible adverts were used for training; 200,039 for validation
    and 201,085 were held aside for final testing. The input was not
    deduplicated using unverified similarity assumptions.

    **What remains unknown?** The model does not read seller descriptions,
    VINs or photographs. It does not inspect the vehicle or know the final
    negotiated amount. Asking prices are not sales prices, customs decisions
    or tax valuations.""")
    st.markdown("**Dataset and usage:** [European Used Car Market]("+DATASET+
        ") · CC BY-NC-SA 4.0. Car Price Studio software is separately MIT-licensed.")
    if bundle:
        winner=bundle[2]["overall_champion"]
        st.info(f"Selected: ResMLP with Tail-MAE; sealed-test MAE "
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


def _price_indices():
    """European price clock: empirical coverage first, no fabricated time series."""
    st.markdown("### 📈 European Price Index Observatory")
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
    tab=st.pills("European Studio panels",NAV,selection_mode="single",
                 key="eu_workspace",label_visibility="collapsed") or NAV[0]
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
