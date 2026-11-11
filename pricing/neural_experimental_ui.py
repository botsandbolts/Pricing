"""Experimental model workshop: two independent parity-gated neural engines.

Ask: sole IBM Granite 97M QInt8 ONNX + DCNv2 AFT.
Bid: verified full historical auction cohort, ResMLP and Huber USD.
Research models are never substituted by the primary calculator's predictions.
"""
import json
import uuid
import logging
import sys
import time
from pathlib import Path
from html import escape
import numpy as np
import pandas as pd
import streamlit as st
from functools import partial
from pricing.i18n import tr, locale
from pricing.i18n_story import story
from pricing.i18n_fields import field
from pricing.i18n_enums import option_label, option_label_for
from pricing.granite_aft_ask_live import available as granite_aft_available, load as load_granite_aft
from pricing.granite_aft_i18n import copy as granite_copy

from pricing.ray_bid_winner_live import available as ray_bid_available, load as load_ray_bid
from pricing.us_bid_resmlp_live import available as resmlp_bid_available, load as load_resmlp_bid


def _valuation_probe(stage, started=None):
    """Server-only evaluation checkpoint; never record vehicle text or VINs.

    The latest production log ended after Granite initialized without an exception.
    These phase markers distinguish model-load, embedding and head failures,
    including an abrupt process termination where Python can't print a traceback.
    """
    elapsed=round((time.monotonic()-started)*1000) if started is not None else None
    try:
        import resource
        peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    except (OSError,ImportError,ValueError):
        peak_mb=None
    logging.getLogger("car_price_studio.valuation").warning(
        "valuation stage=%s elapsed_ms=%s peak_rss_mb=%s python=%s",
        stage,elapsed,peak_mb,sys.version.split()[0])
    from pricing.memory_watch import record as log_memory
    log_memory(stage,elapsed_ms=elapsed,
               detailed=stage in ("form_submitted","receipt_saved",
                                  "ask_checkpoint_complete",
                                  "bid_checkpoint_complete"))


def _bound_native_cpu_threads():
    """Keep Torch from spawning an excessive CPU worker pool on Cloud."""
    import torch
    if torch.get_num_threads()>2:
        torch.set_num_threads(2)


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


def _vehicle(prefix,bid=False,ready=False,compact=False,focus=False,*,
             two_models=False,english_ready=False,international_ready=False,
             english_cvar=8959.62):
    """One unified, keyboard-accessible form. Main pricing has no scan/BART step."""
    # Set initial values BEFORE creating the widgets. The example picker also
    # writes these keys, so passing positional defaults as well makes
    # Streamlit 1.66 emit a widget-state warning and risks stale text.
    for name, default in (
        ("year", 2015), ("make", "Toyota"), ("model", "Corolla"),
        ("mileage", 95000), ("state", "ca"),
        ("description", "Well maintained. Clean title. New tires."
         if locale() in ("en-GB","en-US") else ""),
    ):
        st.session_state.setdefault(prefix+"_"+name,default)
    with st.form(prefix+"_vehicle",clear_on_submit=False):
        st.markdown("### 🚗 01 / "+tr("form") if focus else
                    "### 🚗 02 / "+tr("form") if compact else
                    "#### 01 / "+tr("form"))
        if compact:
            st.caption("No VIN upload, no special instructions and no degree in automotive engineering required. The description is optional for Bid but useful for Ask."
                       if locale() in ("en-GB","en-US") else story("form_intro"))
        a,b,c=st.columns(3)
        year=a.number_input(field("model_year"),min_value=1900,max_value=2021,
                            key=prefix+"_year",persist_state="session")
        make=b.text_input(tr("make"),key=prefix+"_make",persist_state="session")
        model=c.text_input(tr("model")+" / trim",key=prefix+"_model",persist_state="session")
        a,b,c=st.columns(3)
        miles=a.number_input(field("miles"),min_value=0,max_value=2000000,
                             step=1000,key=prefix+"_mileage",persist_state="session")
        state=b.text_input(field("state"),max_chars=2,
                           key=prefix+"_state",persist_state="session")
        body=c.selectbox(field("body"),[
              "sedan","suv","truck","pickup","hatchback","coupe",
              "wagon","van","convertible","other","unknown"],
              key=prefix+"_body",persist_state="session",format_func=partial(option_label_for,locale()))
        if compact:
            st.caption("Three useful facts; everything else is recorded as 'unknown', not guessed."
                       if locale() in ("en-GB","en-US") else field("form_help"))
            a,b,d=st.columns(3)
            condition=a.selectbox(field("condition"),[
                "unknown","good","excellent","fair","like new","new","salvage"],
                key=prefix+"_condition",persist_state="session",format_func=option_label)
            title=b.selectbox(field("title"),[
                "unknown","clean","rebuilt","salvage","lien","missing","parts only"],
                key=prefix+"_title",persist_state="session")
            transmission=d.selectbox(tr("transmission"),[
                "unknown","automatic","manual","other"],
                key=prefix+"_transmission",persist_state="session")
            fuel=st.session_state.get(prefix+"_fuel","unknown")
            drive=st.session_state.get(prefix+"_drive","unknown")
            cylinders=st.session_state.get(prefix+"_cylinders","unknown")
            paint=st.session_state.get(prefix+"_paint","unknown")
        else:
            with st.expander(field("optional")):
                a,b,d=st.columns(3)
                condition=a.selectbox("Condition",[
                    "good","excellent","fair","like new","new","salvage","unknown"],
                    key=prefix+"_condition",persist_state="session",format_func=option_label)
                fuel=b.selectbox(tr("fuel"),[
                    "gas","diesel","hybrid","electric","other","unknown"],
                    key=prefix+"_fuel",persist_state="session",format_func=option_label)
                transmission=d.selectbox("Transmission",[
                    "automatic","manual","other","unknown"],
                    key=prefix+"_transmission",persist_state="session",format_func=option_label)
                a,b,d=st.columns(3)
                drive=a.selectbox(field("drive"),[
                    "fwd","rwd","4wd","unknown"],key=prefix+"_drive",persist_state="session")
                title=b.selectbox(field("title"),[
                    "clean","rebuilt","salvage","lien","missing","parts only","unknown"],
                    key=prefix+"_title",persist_state="session")
                cylinders=d.selectbox("Cylinders",[
                    "4 cylinders","6 cylinders","8 cylinders",
                    "3 cylinders","5 cylinders","unknown"],
                    key=prefix+"_cylinders",persist_state="session")
                paint=a.selectbox(field("color"),[
                    "silver","white","black","grey","blue","red","green",
                    "brown","other","unknown"],key=prefix+"_paint",persist_state="session")
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
                  key=prefix+"_auction_date",persist_state="session")
            grade=b.selectbox("Pre-sale auction grade",[
                  "unknown","1","2","3","4","5"],
                  index=3,key=prefix+"_grade",persist_state="session")
            fields["auction_date"]=str(auction)
            fields["auction_grade"]=grade
            notes=""
        else:
            st.markdown("### ✍ 03 / "+("What does the advert say?" if locale() in ("en-GB","en-US") else field("description")) if compact
                        else "#### 02 / "+("The car's entire story" if locale() in ("en-GB","en-US") else field("description")))
            notes=st.text_area(field("description"),
                max_chars=8000,height=125 if compact else 140,
                key=prefix+"_description",persist_state="session",
                help=granite_copy(locale())["intro"])
        if two_models and not bid:
            st.markdown("### 🌍 04 / IBM Granite R2 97M · DCNv2 AFT")
            st.caption(granite_copy(locale())["intro"])
            clicked=st.form_submit_button("🌍 "+granite_copy(locale())["cta"],
                key=prefix+"_calculate_granite",type="primary",
                disabled=not ready,width="stretch")
            submitted="granite" if clicked else False
        else:
            clicked=st.form_submit_button(
                "🚗 "+tr("calculate") if compact else
                "🌍 "+granite_copy(locale())["cta"],
                key=prefix+"_calculate_single_tap",
                type="primary",disabled=not ready,width="stretch")
            submitted="granite" if clicked else False

    return fields,notes,submitted

def _price(label,price,subtitle):
    if not np.isfinite(float(price)) or float(price)<=0:
        raise ValueError("Invalid price; cannot display")
    st.markdown('<section class="exp-result" aria-label="'+escape(label)+'">'
                '<small>'+escape(label.upper())+'</small><strong>$'+f"{price:,.0f}"+
                '</strong><p>'+escape(subtitle)+'</p></section>',
                unsafe_allow_html=True)




def _explanation(root):
    """CVaR95 documentation, with no outcome-filtered historical comparisons."""
    st.markdown("### 🧪 CVaR95 · when a valuation misses badly")
    st.write("CVaR95 is the mean absolute dollar error among the worst 5% "
             "of an original validation set. MAE and RMSE are reported too. "
             "Historic Ask and Bid errors are measured on different markets.")
    from pricing.cvar95_experiment_science import render as show_cvar_science
    show_cvar_science(root)

def _apply_example():
    from pricing.description_examples import load_examples, selected_example
    e=selected_example(load_examples(),st.session_state.get("experimental_example",0))
    if e is None:return
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
        if key=="year":v=max(1900,min(2021,int(v)))
        if key=="mileage":v=max(0,min(2000000,int(float(v))))
        st.session_state["granite_ask_"+key]=v
    st.session_state["granite_ask_description"]=e.get("text","")[:8000]
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


def render(root,*,main=False):
    """Public homepage and research fallback share exactly one Ask/Bid inference path."""
    root=Path(root)
    # The homepage campaign now lives in studio_design.render_hero().
    # Keep the model lab's smaller introduction but never stack duplicate
    # full-width banners above the public price calculator.
    if main:
        st.markdown(CSS,unsafe_allow_html=True)
    else:
        st.markdown(CSS+"""
<section class="exp-hero" aria-label="Experimental vehicle pricing models">
<small>CAR PRICE STUDIO · MODEL RESEARCH</small>
<h1>Two markets. One car.<br>Fewer pricing nightmares.</h1>
<p>Seller asking and wholesale auction estimates from the same verified
models used on our main price calculator.</p>
<span class="exp-chip">🌍 Granite QInt8 + DCNv2 AFT</span>
<span class="exp-chip">🚘 ResMLP + robust Huber</span>
<span class="exp-chip">🛡 CVaR95 · worst 5%</span>
</section>""",unsafe_allow_html=True)
    from pricing.studio_design import market_guide
    from pricing.description_examples import load_examples
    from pricing.current_value import current_transport,clock_controls,render_market_clock,render_adjustment,render_time_evidence
    from pricing.cvar95_experiment_science import model_kpis,all_validation
    # One canonical pricing page; the homepage CTA changes only the hash.
    market_guide()
    if locale() not in ("en-GB","en-US"):
        st.caption(story("us_pitch"))
    else:
        st.markdown("### 🧰 Full-cohort car valuation laboratory")
        st.caption("The only seller Ask engine is IBM Granite 97M INT8 ONNX "
                   "+ DCNv2 AFT, verified on the original historical validation. "
                   "Wholesale Bid uses ResMLP Huber on original auction sales.")

    if main and locale() in ("en-GB","en-US"):
        # Visibility is intentional: Captum itself still runs ONLY on a user
        # click after pricing. A pre-price guide prevents the key feature
        # disappearing beyond the fold for mobile and first-time visitors.
        st.markdown("""
<section aria-label="Where to find the price explanation"
style="border:1px solid #6daea8;border-left:5px solid #168775;
border-radius:14px;padding:12px 15px;margin:10px 0 14px;
background:rgba(34,150,132,.07)">
<div style="font-size:1.03rem;font-weight:800;margin-bottom:5px">
🔬 Why THAT price? · Your AI explanation is included
</div>
<div style="font-size:.91rem;line-height:1.5">
<strong>1.</strong> Enter your car.
<strong>2.</strong> Tap <strong>Calculate seller Ask + wholesale Bid</strong>.
<strong>3.</strong> Find the open <strong>Why THAT price?</strong>
panel directly under your price results. The AI explains itself
only when you ask. No model circus required. 🎪🚗
</div></section>
""",unsafe_allow_html=True)
    if main and locale() not in ("en-GB","en-US"):
        st.caption("🔬 "+story("submit_tip"))
    # Real anchor target for the homepage's gold call-to-action.
    # Placed directly above the editable pricing form, not on a dead route.
    if main:
        st.markdown('<div id="studio-calculator" tabindex="-1"></div>',
                    unsafe_allow_html=True)
    examples=load_examples()
    st.markdown("### 🚗 01 / "+("Start with an example—or your own car" if locale() in ("en-GB","en-US") else story("form_intro")))
    if examples:
        st.selectbox(
            (f"Search {len(examples):,} real, price-redacted car adverts" if locale() in ("en-GB","en-US") else field("compare")),
            range(len(examples)+1),key="experimental_example",persist_state="session",
            format_func=lambda i:("Write my own listing" if locale() in ("en-GB","en-US") else story("form_intro")) if i==0 else examples[i-1]["label"],
            on_change=_apply_example)
        st.caption("Historical, price-redacted car adverts (CC BY 4.0). Choose an example to populate the form, or enter your own car. Picking a car does not automatically submit a valuation." if locale() in ("en-GB","en-US") else story("form_intro"))
    # Cheap, read-only availability checks before the focused form. The
    # default page preserves the same research/model introduction ordering.
    resmlp_bid_ready=resmlp_bid_available(root)
    ray_bid_ready=ray_bid_available(root)
    ask_ready=granite_aft_available(root)
    bid_ready=resmlp_bid_ready or ray_bid_ready
    ask_champion=False
    bid_champion=False

    valuation_date,allow_analogue=clock_controls(
        prefix="main" if main else "experimental",prominent=main)
    # These historical models and their original data are never retrained by changing the date.
    try:
        transport,error=current_transport(root)
        render_market_clock(transport,valuation_date,error)
    except (OSError,ValueError,KeyError) as exc:
        transport=None
        st.warning("This month's market transport cannot be loaded: "+str(exc))
    with st.expander("🔨 Historical wholesale auction reference · advanced" if locale() in ("en-GB","en-US") else field("optional"),expanded=False):
        auction=st.date_input("Original auction date (2014–2015)",
            pd.Timestamp("2015-01-15").date(),min_value=pd.Timestamp("2014-01-01").date(),
            max_value=pd.Timestamp("2015-12-31").date(),key="experiment_joint_auction")
        grade=st.selectbox("Historical pre-sale grade",["unknown","1","2","3","4","5"],index=3,key="experiment_joint_grade")
        st.caption("2016–2021 cars can use an optional *hypothetical five-year-old auction analogue*. It is not evidence of what that vehicle generation sold for in 2015.")
    fields,notes,go=_vehicle(
        "granite_ask",ready=ask_ready,compact=main,
        two_models=main,english_ready=False,
        international_ready=ask_ready)
    choice="granite"
    chosen_ready=ask_ready
    request_signature=json.dumps({
        "fields":fields,"notes":notes,"date":str(valuation_date),
        "auction":str(auction),"grade":grade,
        "allow_analogue":bool(allow_analogue),"choice":choice
    },sort_keys=True,separators=(",",":"),default=str)
    pending=st.session_state.get("experimental_combined_receipt")
    # If a widget/browser rerun interrupted the first click, finish the same
    # request automatically, without asking the visitor to tap Granite again.
    if (not go and pending and
        pending.get("_request_signature")==request_signature and
        (("ask" not in pending and "ask_error" not in pending) or
         (bid_ready and "bid" not in pending and "bid_error" not in pending))):
        go="granite"

    # The public date picker can predate a particular model year. Reject before
    # expensive Granite inference, and do not retain a stale previous receipt.
    if go:
        from marketbridge.time_engine import age_on
        age=age_on(fields["year"],valuation_date)
        if age<0:
            st.error("Time-machine paradox: that car model didn't exist on "
                     "the selected historical date. Choose an earlier model year "
                     "or a later valuation date.")
            st.session_state.pop("experimental_combined_receipt",None)
            go=False
        elif age>30:
            st.error("This research supports cars aged 0–30 years on the "
                     "selected date. Collector cars need a specialist appraisal.")
            st.session_state.pop("experimental_combined_receipt",None)
            go=False
    if go:
        # Acknowledge the first tap immediately, before Granite initialization.
        progress_notice=st.empty()
        progress_notice.info("⏳ One click received. Calculating your Ask and Bid…")
        _valuation_probe("form_submitted")
        # A browser may trigger another rerun while the first cold-started
        # Granite evaluation is still running. Identify *one* exact submitted
        # request, reuse a successful Ask checkpoint, and avoid double billing,
        # duplicate database rows and redundant ONNX inference.
        previous=st.session_state.get("experimental_combined_receipt")
        same_request=bool(previous and
            previous.get("_request_signature")==request_signature)
        if same_request:
            results=dict(previous)
        else:
            results={"fields":fields,"notes":notes,"date":str(valuation_date),
                     "ask_choice":choice,
                     "ask_backend":"granite_aft_only",
                     "request_id":str(uuid.uuid4()),
                     "_request_signature":request_signature}
        if "ask" in results and "bid" in results:
            progress_notice.empty()
            st.info("✅ These exact inputs have already been evaluated. "
                    "Your results are below; no second inference was needed.")
        # Preserve partial successes when a later Streamlit rerun interrupts
        # the script, including after the 10-second initial Granite load.
        st.session_state["experimental_combined_receipt"]=dict(results)
        try:
            _bound_native_cpu_threads()
            _valuation_probe("torch_cpu_threads_bounded")
        except (ImportError,RuntimeError,OSError) as exc:
            logging.getLogger("car_price_studio.valuation").exception(
                "Cannot initialize CPU pricing runtime: %s",type(exc).__name__)
            st.error("Pricing runtime is unavailable on this server. "
                     "The form remains available; try again after the app recovers.")
            results["ask_error"]="Pricing runtime initialization failed."
            results["bid_error"]="Pricing runtime initialization failed."
            ask_ready=bid_ready=chosen_ready=False
        if chosen_ready and "ask" not in results:
            results.pop("ask_error",None)
            phase="load_checkpoint"
            started=time.monotonic()
            _valuation_probe("ask_checkpoint_start")
            try:
                with st.spinner("🌍 Loading verified Granite INT8 ONNX · DCNv2 AFT…"):
                    model=load_granite_aft(root)

                    _valuation_probe("ask_checkpoint_complete",started)
                    phase="encode_description"
                    started=time.monotonic()
                    _valuation_probe("ask_granite_encode_start")
                    # Exactly one immutable Granite encoder, no other Ask backend.
                    prepared=model.prepare_text(notes)
                    _valuation_probe("ask_granite_encode_complete",started)
                    phase="price_head"
                    started=time.monotonic()
                    _valuation_probe("ask_native_forward_start")
                    results["ask"]=model.predict_prepared(fields,prepared)
                    results["ask_embedding"]=prepared["embedding"]
                    st.session_state["experimental_combined_receipt"]=dict(results)
                    _valuation_probe("ask_native_forward_complete",started)
            except Exception as exc:
                logging.getLogger("car_price_studio.valuation").exception(
                    "Ask valuation failed at phase=%s error_type=%s",
                    phase,type(exc).__name__)
                results["ask_error"]=(
                    "Ask engine failed during "+phase+" ("+type(exc).__name__+
                    "). Check server logs for valuation stage.")
                st.error("Seller Ask couldn't finish "+phase.replace("_"," ")+
                         ". Your inputs are preserved and Bid can still run. "
                         "Details are in the app's private server log.")
        if bid_ready and "bid" not in results:
            results.pop("bid_error",None)
            phase="build_features"
            started=time.monotonic()
            _valuation_probe("bid_features_start")
            try:
                from pricing.neural_bid_blend_live import load as bid_load
                bid_fields,analogue=_historical_bid_features(fields,auction,grade,allow_analogue)
                _valuation_probe("bid_features_complete",started)
                phase="load_checkpoint"
                started=time.monotonic()
                _valuation_probe("bid_checkpoint_start")
                with st.spinner("🔨 Running the verified "+("ResMLP · Huber USD" if resmlp_bid_ready else
                        "full-original-cohort Ray DeepFM" if ray_bid_ready else "original auction model")+" Bid model…"):
                    model=(load_resmlp_bid(root) if resmlp_bid_ready else
                           load_ray_bid(root))
                    _valuation_probe("bid_checkpoint_complete",started)
                    phase="price_head"
                    started=time.monotonic()
                    _valuation_probe("bid_native_forward_start")
                    results["bid"]=model.predict(bid_fields)
                    results["bid_fields"]=bid_fields
                    _valuation_probe("bid_native_forward_complete",started)
                if analogue:results["bid_analogue"]=analogue
                st.session_state["experimental_combined_receipt"]=dict(results)
            except Exception as exc:
                logging.getLogger("car_price_studio.valuation").exception(
                    "Bid valuation failed at phase=%s error_type=%s",
                    phase,type(exc).__name__)
                results["bid_error"]="Bid engine failed during "+phase+" ("+type(exc).__name__+")."
                st.warning("Wholesale Bid couldn't finish "+phase.replace("_"," ")+
                           ". Ask remains available if it completed. "
                           "Check the private server log for details.")
        elif not bid_ready and "bid_error" not in results and "bid" not in results:
            results["bid_error"]="Verified wholesale weights are not installed or have failed the parity gate."
        progress_notice.empty()
        st.session_state["experimental_combined_receipt"]=results
        _valuation_probe("receipt_saved")
        if "ask" in results and "bid" in results:
            st.success("✅ Both prices are ready below." if locale() in ("en-GB","en-US") else field("quote"))
        elif "ask" in results or "bid" in results:
            st.warning("One market is unavailable; the successful valuation is preserved. "
                       "The other estimate will never be replaced by invented values.")
    saved=st.session_state.get("experimental_combined_receipt")
    if saved:
        st.markdown("### 💵 "+("Your car · estimated prices" if locale() in ("en-GB","en-US") else field("quote")))
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
                    st.warning("⚠️ The wholesale estimate uses an age-matched proxy, not an observed auction for this model year. Accuracy for newer generations is unknown.")
                    st.caption("Proxy model year: "+str(saved["bid_analogue"]["proxy_model_year"])+"; requested model year: "+str(saved["bid_analogue"]["requested_model_year"]))
                if transport is None:
                    st.warning("The selected-date estimate is unavailable. No historical price is shown as a substitute.")
                    continue
                try:
                    receipt=_adjust_price(transport,historical,saved["fields"],
                                          side,valuation_date,allow_analogue)
                    adjusted[side]=receipt;prices[side]=receipt["price"]
                    flavour=("🌍 Granite multilingual" if side=="ask" and saved.get("ask_choice")=="granite"
                             else "🇬🇧 Granite English" if side=="ask" else "auction")
                    _price(label+" · "+flavour+" · "+str(valuation_date)+" scenario",
                        receipt["price"],
                        "Date-adjusted scenario, not an observed transaction or a guaranteed bid.")
                    render_adjustment(receipt)
                    from pricing.analytics import record_price
                    record_price(side,receipt["price"],saved["fields"],
                        model=result.get("model","unknown"),
                        source="granite-aft-"+saved.get("ask_choice","granite"),
                        valuation_date=valuation_date,event_key=saved.get("request_id"))
                except (OSError,ValueError,KeyError,TypeError) as exc:
                    st.warning("Cannot adjust "+label.lower()+" to this month: "+str(exc))
        if len(prices)==2:
            st.metric("Asking minus wholesale · indicative market gap",
                      _usd(prices["ask"]-prices["bid"]))
            st.caption("These are independent targets, not a guaranteed dealer profit or a matched-car trading spread.")
        if main:
            # High-visibility explanation directly after prices. Preserve
            # lazy Captum imports and unchanged native cached checkpoints.
            st.markdown("### 🔍 04 / Why THAT price? · Your AI price detective")
            st.caption("Right here under your quote. Choose Ask or Bid and "
                       "click Explain; the first quote has zero Captum overhead.")
            with st.expander("🧠 About the neural engine",expanded=False):
                st.caption(granite_copy(locale())["intro"])
            available_to_explain=[side for side in ("ask","bid") if side in saved]
            if available_to_explain:
                labels={"ask":"🪧 Seller Ask","bid":"🔨 Wholesale Bid"}
                chosen_side=st.radio(
                    "Which neural estimate should we explain?",
                    available_to_explain,format_func=lambda s:labels[s],
                    horizontal=True,key="experimental_explain_market")
                if st.button("🔍 Explain this price · show real AI feature contributions",
                             type="primary",width="stretch",
                             key="experimental_explain_go"):
                    try:
                        _bound_native_cpu_threads()
                        with st.spinner("Inspecting the neural network · 24 small, memory-bounded steps…"):
                            from pricing.neural_price_explanation import calculate
                            interpretation=calculate(root,chosen_side,saved)
                        st.session_state["experimental_neural_explanation"]={
                            "request_id":saved.get("request_id"),
                            "market":chosen_side,"result":interpretation}
                    except Exception as exc:
                        logging.getLogger("car_price_studio.explanation").exception(
                            "On-demand explanation failed: %s",type(exc).__name__)
                        st.error("The explanation could not run ("+
                                 type(exc).__name__+"). Your price is still valid; "
                                 "please try again without recalculating.")
                previous=st.session_state.get("experimental_neural_explanation")
                if (previous and previous.get("request_id")==saved.get("request_id")
                        and previous.get("market")==chosen_side):
                    from pricing.neural_price_explanation import render as render_neural_explanation
                    render_neural_explanation(previous["result"],
                        adjusted.get(chosen_side),saved.get("bid_analogue"))


        if main and prices.get("ask") and saved.get("request_id"):
            st.markdown("### ⭐ "+("Your verdict: was our AI valuation accurate?" if locale() in ("en-GB","en-US") else field("feedback_title")))
            st.caption("The price detective needs human witnesses too. "
                       "Please compare the estimate with what you know about your car.")
            key=str(saved["request_id"])
            if st.session_state.get("_feedback_saved_"+key):
                st.success("🙏 Your feedback is saved. Thank you for helping us improve!")
            else:
                with st.form("accuracy_feedback_"+key):
                    score=st.radio("Rate this quote", [1,2,3,4],
                        index=None,horizontal=True,
                        format_func=lambda n:{1:"1 · Catastrophic 💥",2:"2 · Poor 😬",
                            3:"3 · Good 👍",4:"4 · Excellent 🏆"}[n],
                        key="rating_"+key)
                    permission=st.checkbox(("I agree to save my rating, basic car details and both price estimates. No seller story, IP address, VIN or contact details." if locale() in ("en-GB","en-US") else field("feedback_consent")),
                        key="feedback_permission_"+key)
                    submitted=st.form_submit_button(
                        "📮 "+("Send accuracy feedback" if locale() in ("en-GB","en-US") else field("feedback_submit")),type="primary")
                if submitted:
                    from pricing.analytics import record_accuracy_feedback
                    if score is None:
                        st.warning("Please choose one rating from 1 to 4.")
                    elif not permission:
                        st.warning("Please agree to saving this feedback first.")
                    elif record_accuracy_feedback(
                        key,"granite",int(score),
                        saved["fields"],prices,saved["date"],explicit_consent=True):
                        st.session_state["_feedback_saved_"+key]=True
                        st.success("✅ Thank you! Feedback saved to our private database.")
                    else:
                        st.error("Feedback was not saved; your car valuation is unaffected.")
        if "ask" in saved and ask_ready:
            with st.expander("✍ What-if? Rewrite the description · rerun the whole neural model"):
                alt=st.text_area("Alternative advertisement",value=saved["notes"],key="experimental_reword",max_chars=8000)
                if st.button("Recalculate complete neural asking model",key="experimental_reword_go"):
                    try:
                        revised=load_granite_aft(root).predict(alt,saved["fields"])["price"]

                        new=_adjust_price(transport,revised,saved["fields"],"ask",valuation_date,allow_analogue) if transport else None
                        st.metric("Rewritten historical asking",_usd(revised),delta=_usd(revised-saved["ask"]["price"]))
                        if new:st.metric("Rewritten date-adjusted asking",_usd(new["price"]))
                        st.caption("A change in model prediction is not evidence rewriting an advert changes what a buyer pays.")
                    except (OSError,RuntimeError,ValueError,TypeError,KeyError) as exc:
                        st.error("What-if calculation unavailable: "+str(exc))
            with st.expander("📉 Mileage sensitivity · full-network reruns"):
                st.caption("Five real model calls only when you ask; your memory "
                           "budget is not an all-you-can-eat buffet.")
                if st.button("Draw mileage sensitivity",key="experimental_mileage_go"):
                    try:
                        model=load_granite_aft(root)

                        prepared=({"embedding":saved["ask_embedding"],"raw":saved["notes"]}
                                  if "ask_embedding" in saved else model.prepare_text(saved["notes"]))
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
                "warning":"Date transport is a model-based reconstruction or forecast, not an observed transaction."}
        st.download_button("📥 Download both-model research receipt (JSON)",
            json.dumps(report,indent=2,default=str),file_name="experimental-bid-ask.json",
            mime="application/json",key="experimental_both_download")
    if main:
        st.markdown("### 🚙 The car doesn't read spreadsheets. Luckily, we do.")
        st.caption("Two separate historical targets, one transparent car studio. "
                   "The asking price isn't a completed retail sale, and the auction "
                   "estimate isn't a dealer offer. Date scenarios aren't transactions.")
    else:
        st.markdown("### 🧰 Other tools in Car Price Studio")
        st.caption("More comparison tools are available from the Studio panels.")
    with st.expander("⚙️ Which engines priced this car? · verified model details",expanded=False):
        a,b=st.columns(2)
        with a:
            st.markdown("**🌍 Sole Ask: Granite INT8 + DCNv2 AFT**")
            if ask_ready:
                from pricing.granite_aft_ask_live import manifest as granite_manifest
                q=granite_manifest(root)["validation"]
                st.caption(granite_copy(locale())["intro"])
                st.caption("Original 2021 validation ({:,} listings): "
                           "CVaR95 USD {:,.0f} · MAE USD {:,.0f} · RMSE USD {:,.0f}."
                           .format(q["rows"],q["CVaR95_absolute_USD"],
                                   q["MAE_USD"],q["RMSE_USD"]))
            else:
                st.warning("Granite serving checkpoint unavailable; no Ask fallback.")

        with b:
            st.markdown("**🔨 Wholesale auction · complete original cohort**")
            if resmlp_bid_ready:
                meta=json.loads((root/"models/us-bid-resmlp-20261111/manifest.json").read_text())
                stats=meta["original_validation"]
                st.caption("ResMLP + Huber USD; TRAIN-only model-specific mileage. "
                           "Original validation: CVaR95 USD {:,.0f}, MAE USD {:,.0f}, "
                           "RMSE USD {:,.0f}, {:,} completed auctions."
                           .format(stats["CVaR95_absolute_USD"],stats["MAE_USD"],
                                   stats["RMSE_USD"],stats["rows"]))
            elif ray_bid_ready:
                st.caption("Original full-cohort DeepFM; model-native historical wholesale Bid.")
            else:
                st.warning("Verified wholesale auction model unavailable.")
        st.caption("Ask estimates advertised prices; Bid estimates completed "
                   "2014–15 wholesale sale amounts. Neither is a live purchase offer.")

    st.markdown("### 🧪 Research notes: matched historical populations")
    st.caption("Ask: 2021 Craigslist advertised prices, full original "
               "training and validation. Bid: 2014–15 completed wholesale "
               "auctions, full original training and validation. No "
               "observed-price-dependent row exclusions in active serving. "
               "Historical research cannot prove 2026 market accuracy.")
    if saved and "ask" in saved:
        st.info(granite_copy(locale())["historical"])

    with st.expander("🧠 How the new neural models learn",expanded=False):
        st.markdown("**Asking:** Frozen Granite multilingual QInt8 embeddings "
                    "enter our sole DCNv2 AFT model with TRAIN-only mileage features.")
        st.markdown("**Bid:** The preferred ResMLP uses Huber-dollar loss "
                    "and train-only make/model annual mileage.")
        st.caption("CVaR95 is the worst-five-percent average absolute dollar "
                   "error. It is not a legal valuation or a guarantee.")

    with st.expander("🛡️ Why CVaR95? The business case and the original scientific evidence"):
        _explanation(root)
