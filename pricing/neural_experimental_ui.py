"""NEURAL LAB: visually branded, traceable, asking-only experimental calculator.

The progress bar follows genuine inference steps rather than a fake timer.
No database dependency for model files. Turso analytics is consented only.
"""
import json
import time
import uuid
from html import escape
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from pricing.neural_experimental_live import available,load

STYLE="""
<style>
.nl-hero{background:radial-gradient(circle at 83% 18%,#6952b4,transparent 40%),
linear-gradient(125deg,#0d1936,#18375a 57%,#39245c);border-radius:26px;
padding:clamp(24px,4vw,52px);color:white;margin:10px 0 23px;
border:1px solid #8f85b1;box-shadow:0 14px 38px #13233b33}
.nl-hero h1{color:#fff;font-size:clamp(2.1rem,5vw,4rem);
letter-spacing:-.065em;line-height:1.06;margin:.5rem 0 1rem}
.nl-hero p{color:#e4eafa;font-size:clamp(.97rem,1.6vw,1.15rem);
line-height:1.55;max-width:740px}
.nl-label{color:#b2eed8;font-size:.76rem;font-weight:850;letter-spacing:.17em}
.nl-chip{display:inline-block;border-radius:99px;padding:.34rem .75rem;
border:1px solid #c4dded99;background:#ffffff13;color:#fff;
margin:8px 7px 0 0;font-size:.78rem;font-weight:750}
.nl-receipt{background:linear-gradient(127deg,#15234a,#343576);border:1px solid #a09cce;
border-radius:23px;padding:23px;margin:10px 0 16px;color:white}
.nl-receipt strong{display:block;color:white;letter-spacing:-.045em;
font-size:clamp(2.1rem,5vw,3.9rem)}
.nl-receipt small{color:#b5efd5;letter-spacing:.13em;font-weight:850}
.nl-receipt p{color:#e7eafa;margin:0}
@media(prefers-reduced-motion:reduce){.nl-hero *{animation:none!important;transition:none!important}}
</style>
"""
DEFAULTS={"year":2015,"manufacturer":"Toyota","model":"Corolla SE","odometer":95000,
          "state":"ca","type":"sedan","condition":"good","fuel":"gas",
          "transmission":"automatic","drive":"fwd","title_status":"clean",
          "cylinders":"4 cylinders","paint_color":"silver"}


def _header():
    st.markdown(STYLE+"""<section class="nl-hero" aria-label="Neural Lab experimental car valuation">
<div class="nl-label">CAR PRICE STUDIO · EXPERIMENTAL MODELS · EDITION 01</div>
<h1>NEURAL LAB<br>Cars have stories.<br>We read between the lines.</h1>
<p>Deep &amp; Cross meets MiniLM. A futuristic test kitchen for the price
a seller might <strong>ask</strong> for a used car. Made in Germany,
seasoned with actual research—and just enough science-fiction nonsense.</p>
<span class="nl-chip">🧠 DeepCross / AFT</span>
<span class="nl-chip">✍ MiniLM + 24 text factors</span>
<span class="nl-chip">🇩🇪 Made in Germany</span>
<span class="nl-chip">🧪 Alpha · asking only</span>
</section>""",unsafe_allow_html=True)


def _benchmark(root):
    try:
        d=json.loads((Path(root)/"reports/deepcross-aft-tuning/report.json").read_text())
        w=d["winner_by_validation"]
        if d["status"]!="complete" or w["variant"]!="dropout12" or w["features"]!="minilm":
            raise ValueError("Published winner mismatches release")
        columns=st.columns(3)
        columns[0].metric("Historical validation MAE","$"+f"{w['validation_mae_usd']:,.0f}")
        columns[1].metric("Historical test MAE","$"+f"{w['test_mae_usd']:,.0f}")
        reference=d["full_xgboost_published_unpaired"]["test"]["mae_usd"]
        columns[2].metric("Less test error vs full XGBoost","$"+f"{reference-w['test_mae_usd']:,.0f}")
        st.caption("Measured on 2021 asking listings, not today's sale prices. "
                   "Full-XGBoost reference is aggregate rather than row-paired.")
        return d
    except (OSError,ValueError,KeyError,TypeError):
        st.info("The original held-out benchmark report is not installed.")
        return {}


def _editor():
    for key,val in DEFAULTS.items():
        st.session_state.setdefault("neural_"+key,val)
    st.session_state.setdefault("neural_description","Well maintained. Clean title. New tires.")
    st.markdown("### 01 / Meet your car")
    st.caption("Every field is editable. Pick a real example, or write a listing from scratch.")
    from pricing.description_examples import render_examples
    render_examples(True,prefix="neural",description_key="neural_description",
                    example_key="neural_examples")
    from marketbridge.time_engine import today
    with st.expander("🕰️ The time-machine controls",expanded=False):
        choice=st.radio("Valuation date",["Today","Choose date","End of this month"],
                        horizontal=True,key="neural_date_mode")
        if choice=="Choose date":
            chosen=st.date_input("Price date",today(),key="neural_selected_date")
        elif choice=="End of this month":
            from pricing.current_value import month_end
            chosen=month_end(today())
        else:chosen=today()
    with st.form("neural_vehicle_form"):
        a,b,c=st.columns(3)
        year=a.number_input("Model year",min_value=1900,max_value=2021,key="neural_year")
        make=b.text_input("Make",key="neural_manufacturer")
        model=c.text_input("Model / trim",key="neural_model")
        a,b,c=st.columns(3)
        miles=a.number_input("Mileage (miles)",min_value=0,max_value=2000000,
                             step=1000,key="neural_odometer")
        state=b.text_input("US state code",max_chars=2,key="neural_state")
        body=c.selectbox("Body style",
           ["sedan","SUV","truck","pickup","hatchback","coupe","wagon",
            "van","convertible","other","unknown"],key="neural_type")
        with st.expander("⚙️ Fine details · mechanicals, title, paint"):
            a,b,c=st.columns(3)
            condition=a.selectbox("Condition",["unknown","excellent","good",
                                   "fair","like new","new","salvage"],key="neural_condition")
            fuel=b.selectbox("Fuel",["gas","diesel","hybrid","electric","other","unknown"],
                             key="neural_fuel")
            transmission=c.selectbox("Transmission",["automatic","manual","other","unknown"],
                                     key="neural_transmission")
            a,b,c=st.columns(3)
            drive=a.selectbox("Drive",["unknown","fwd","rwd","4wd"],key="neural_drive")
            title=b.selectbox("Title",["clean","rebuilt","salvage","lien",
                          "missing","parts only","unknown"],key="neural_title_status")
            cylinders=c.selectbox("Cylinders",["unknown","3 cylinders","4 cylinders",
                            "5 cylinders","6 cylinders","8 cylinders","10 cylinders",
                            "12 cylinders","other"],key="neural_cylinders")
            paint=a.selectbox("Paint colour",["unknown","white","black","silver",
                             "grey","blue","red","green","brown","yellow",
                             "orange","purple","custom"],key="neural_paint_color")
        st.markdown("**02 / Your car's autobiography**")
        notes=st.text_area("Seller description",max_chars=8000,height=155,
                           key="neural_description",
                           help="Only used for this prediction; text is not recorded in analytics.")
        submitted=st.form_submit_button("🚀 Launch the price experiment",
                           type="primary",use_container_width=True)
    fields=dict(year=year,manufacturer=make,model=model,odometer=miles,state=state,
                type=body,condition=condition,fuel=fuel,transmission=transmission,
                drive=drive,title_status=title,cylinders=cylinders,paint_color=paint)
    st.caption(f"Selected date: {chosen:%d %b %Y}. This panel has no wholesale bid estimate.")
    return fields,notes,chosen,submitted


def _track(result):
    try:
        from pricing.analytics import record_price
        from pricing.feature_telemetry import record_server_run
        record_price("ask",result["scenario"]["price"],result["fields"],
            "deepcross_aft_minilm_dropout12_v1","experimental_neural_lab",
            str(result["date"]),result["request"])
        record_server_run("neural_lab","deepcross_aft_minilm_dropout12","success",
            input_count=1,result_count=1,duration_ms=int(result["elapsed"]*1000),
            event_key=result["request"])
    except Exception:pass


def _track_error(request,elapsed):
    try:
        from pricing.feature_telemetry import record_server_run,record_server_error
        record_server_run("neural_lab","deepcross_aft_minilm_dropout12","error",
                          input_count=1,result_count=0,
                          duration_ms=int(elapsed*1000),event_key=request)
        record_server_error("neural_lab","inference_failure")
    except Exception:pass


def _run(root,fields,notes,date):
    """Progress ONLY advances when real processing stages are completed."""
    from pricing.current_value import current_transport
    from marketbridge.time_engine import today
    started=time.monotonic();request=str(uuid.uuid4())
    progress=st.progress(0,text="🛸 Preflight checks: has anyone seen the flux capacitor?")
    try:
        model=load(root)
        progress.progress(18,text="👩‍🍳 The neural oven is preheated. Fetching the secret spice blend…")
        prepared=model.prepare_text(notes)
        progress.progress(45,text="🪐 MiniLM has decoded your car's autobiography. Galactic paperwork complete.")
        historical=model.predict_prepared(fields,prepared)
        progress.progress(67,text="🧁 Folding the data into DeepCross. Do not open the oven door.")
        transport,market_warning=current_transport(root)
        scenario=transport.estimate(model,notes,fields,date,side="ask",
                                     as_of=today(),anchor_price=historical["price"])
        progress.progress(86,text="⏳ The used-car time machine is negotiating with the calendar…")
        elapsed=time.monotonic()-started
        result=dict(request=request,fields=fields,notes=notes,date=str(date),
                    historical=historical,scenario=scenario,
                    predicted_text_price=prepared["price_text_usd"],
                    elapsed=elapsed,clock_warning=market_warning)
        st.session_state["neural_lab_receipt"]=result
        _track(result)
        progress.progress(100,text="🎂 The price soufflé has risen. Science served warm.")
        return result
    except (OSError,RuntimeError,ValueError,KeyError,TypeError) as exc:
        st.session_state.pop("neural_lab_receipt",None)
        _track_error(request,time.monotonic()-started)
        st.error("The Neural Lab could not produce a verified valuation: "+str(exc))
        st.info("No substitute estimate was invented. The main calculator is unaffected.")
        return None
    finally:
        progress.empty()


def _receipt(result,root):
    fields=result["fields"];r=result["scenario"]
    label=f"{fields['year']} {fields['manufacturer']} {fields['model']}"
    price="$"+f"{r['price']:,.0f}"
    st.markdown("""<section class="nl-receipt" aria-label="Experimental asking-price receipt">
<small>SELLER ASKING · DATE-ADJUSTED SCENARIO</small><strong>"""+
        escape(price)+"""</strong><p>"""+escape(label)+""" · """+
        escape(str(result["date"]))+"""</p></section>""",unsafe_allow_html=True)
    st.caption("Advertised asking-price scenario—not a completed sale, dealer offer or wholesale bid.")
    a,b,c=st.columns(3)
    a.metric("May 2021 neural anchor","$"+f"{result['historical']['price']:,.0f}")
    b.metric("Ageing","×"+f"{r['age_factor']:.3f}")
    c.metric("Market movement","×"+f"{r['market_factor']:.3f}")
    st.caption("MiniLM description-only price input: $"+f"{result['predicted_text_price']:,.0f}"+
               ". It is a model feature, not a second transaction estimate.")
    with st.expander("🛠️ Price-clock receipts & evidence",expanded=True):
        from pricing.current_value import render_adjustment,current_transport,render_market_clock,render_time_evidence
        render_adjustment(r)
        try:
            transport,error=current_transport(root)
            render_market_clock(transport,result["date"],error)
            render_time_evidence(transport,{"ask":r})
        except (OSError,ValueError,KeyError):
            st.caption("Detailed time evidence unavailable; numerical factors above are unchanged.")
    st.caption(f"Actual inference time: {result['elapsed']:.1f}s. The alpha deliberately prioritizes correctness over speed.")


def _scenarios(root,result):
    st.markdown("### 03 / Question the oracle")
    st.caption("What-if experiments rerun the fitted network. These are not causal effects, "
               "SHAP explanations or confidence intervals.")
    fields=result["fields"];notes=result["notes"]
    original_miles=float(fields["odometer"])
    from pricing.current_value import current_transport
    from marketbridge.time_engine import today
    model=load(root);prepared=model.prepare_text(notes)
    transport,_=current_transport(root)
    def estimate(f,detail):
        historical=model.predict_prepared(f,prepared)
        return transport.estimate(model,detail,f,result["date"],side="ask",
                                  as_of=today(),anchor_price=historical["price"])["price"]
    with st.expander("📉 Mileage sensitivity",expanded=True):
        values=[]
        for change in [-20000,-10000,0,10000,20000]:
            scenario={**fields,"odometer":max(0,min(2000000,original_miles+change))}
            try:
                v=estimate(scenario,notes)
                values.append({"Mileage":int(scenario["odometer"]),"Ask USD":round(v)})
            except ValueError:continue
        if values:
            df=pd.DataFrame(values)
            st.line_chart(df.set_index("Mileage")["Ask USD"])
            st.dataframe(df,hide_index=True,use_container_width=True)
    with st.expander("🔧 Condition sensitivity"):
        if st.button("Compare conditions",key="neural_conditions"):
            rows=[]
            for c in ["excellent","good","fair","salvage"]:
                try:rows.append({"Condition":c,"Ask USD":round(estimate({**fields,"condition":c},notes))})
                except ValueError:continue
            if rows:st.bar_chart(pd.DataFrame(rows).set_index("Condition"))
    with st.expander("✍ Rewrite the listing, keep the vehicle"):
        alt=st.text_area("Alternative description",value=notes,max_chars=8000,
                         key="neural_whatif_text")
        if st.button("Compare the new wording",key="neural_compare_text"):
            try:
                altered=model.prepare_text(alt)
                historical=model.predict_prepared(fields,altered)
                new=transport.estimate(model,alt,fields,result["date"],side="ask",
                          as_of=today(),anchor_price=historical["price"])["price"]
                st.metric("Alternative asking scenario","$"+f"{new:,.0f}",
                          delta="$"+f"{new-result['scenario']['price']:+,.0f}")
                st.caption("A model response to wording, not a causal attribution.")
            except ValueError as exc:st.warning(str(exc))
    with st.expander("📆 Value over time · the future garage"):
        annual=st.slider("Expected miles per year",0,40000,12000,1000,key="neural_annual")
        if st.button("Draw asking-price history & scenarios",key="neural_timeline"):
            rows=[];origin=pd.Timestamp(result["date"])
            for months in [-12,-9,-6,-3,0,3,6]:
                month=origin+pd.DateOffset(months=months)
                f={**fields,"odometer":max(0,min(2000000,original_miles+months*annual/12))}
                try:
                    hist=model.predict_prepared(f,prepared)
                    val=transport.estimate(model,notes,f,month.date(),side="ask",
                                as_of=today(),anchor_price=hist["price"])["price"]
                    rows.append({"Date":month.date(),"Ask USD":val,"Mileage":int(f["odometer"])})
                except ValueError:continue
            if rows:
                st.line_chart(pd.DataFrame(rows).set_index("Date")["Ask USD"])
                st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
        st.caption("Future values depend on unverified depreciation and market-index assumptions.")


def _export(result):
    st.markdown("### 04 / Keep the receipts")
    public={k:v for k,v in result.items() if k!="notes"}
    public["description"]="omitted from exported receipt for privacy"
    a,b=st.columns(2)
    a.download_button("⬇ Download valuation report (JSON)",
        json.dumps(public,default=str,indent=2),
        file_name="neural-lab-valuation.json",mime="application/json",
        use_container_width=True)
    b.download_button("⬇ Download model-method reference",
        "NEURAL LAB / deepcross-aft-minilm-dropout12 / historical asking / May 2021\n"+
        "Validation MAE 3046.61 USD / test MAE 3261.54 USD\n"+
        "Today is an experimental time extrapolation, not a verified price interval.\n",
        file_name="neural-lab-readme.txt",use_container_width=True)


def render(root):
    root=Path(root)
    _header()
    report=_benchmark(root)
    st.caption("Experimental asking only. The wholesale-auction engine will be a separate later experiment.")
    if not available(root):
        st.warning("The tuned neural serving bundle has not passed installation checks "
                   "on this deployment. The tab is ready, but it will not fabricate a price.")
        st.markdown("[Inspect verified deployment workflow](https://github.com/botsandbolts/Pricing/actions/workflows/neural-lab-deploy.yml)")
        from pricing.neural_experimental_docs import render as documentation
        documentation(root,report)
        return
    left,right=st.columns([1.6,1],gap="large")
    with left:
        fields,notes,date,submitted=_editor()
        if submitted:
            _run(root,fields,notes,date)
    result=st.session_state.get("neural_lab_receipt")
    with right:
        if result:_receipt(result,root)
        else:
            st.info("🍰 Your price receipt will appear here when the neural oven has finished.")
            st.caption("No wholesale estimate is included.")
    if result:
        try:_scenarios(root,result)
        except (OSError,RuntimeError,ValueError,KeyError) as exc:
            st.info("Additional what-if graphs temporarily unavailable: "+str(exc))
        _export(result)
    from pricing.neural_experimental_docs import render as documentation
    documentation(root,report)
