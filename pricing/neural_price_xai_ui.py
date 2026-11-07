"""Accessible, bounded, on-demand XAI workbench for the main Ask/Bid receipt.

Every displayed contribution comes from the serving checkpoint and is
reconciled to the actual saved price. All expensive analyses need a click.
No fake LLM narratives, third-party inference or public data transmission.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from pricing import neural_price_xai as xai

STYLE="""<style>
.xai-head{border:1px solid #a0b6d9;background:linear-gradient(118deg,#162c49,#29485f);
color:#fff;border-radius:20px;padding:clamp(17px,3vw,28px);margin:20px 0 10px}
.xai-head h2{color:#fff!important;font-size:clamp(1.5rem,3vw,2.1rem);margin:0 0 6px}
.xai-head p{color:#e6eff8;font-size:.96rem;line-height:1.55;margin:0}
.xai-kicker{font-weight:800;color:#a1ead6;font-size:.76rem;letter-spacing:.1em}
@media(max-width:680px){.xai-head{padding:16px}.xai-head h2{font-size:1.5rem}}
</style>"""


def _money(n):
    return "$"+f"{float(n):,.0f}"


def _signed(n):
    return ("+" if n>0 else "−" if n<0 else "")+_money(abs(float(n)))


def _signature(saved,date,ask_winner,bid_winner):
    payload=[saved.get("fields"),saved.get("bid_fields"),
             str(saved.get("notes","")),str(saved.get("date")),
             str(date),bool(ask_winner),bool(bid_winner),
             [(side,saved.get(side,{}).get("model"),
               saved.get(side,{}).get("price")) for side in ("ask","bid")]]
    return hashlib.sha256(json.dumps(payload,sort_keys=True,default=str).encode()).hexdigest()


def _model(root,side,champions):
    if side=="ask":
        if champions["ask"]:
            from pricing.neural_champion_live import load_ask
            return load_ask(root)
        from pricing.neural_ask_embedding_live import load
        return load(root)
    if side=="bid":
        if champions["bid"]:
            from pricing.neural_champion_live import load_bid
            return load_bid(root)
        from pricing.neural_bid_blend_live import load
        return load(root)
    raise ValueError("Invalid explanation market")


def _plot_grouped(report,title):
    import plotly.graph_objects as go
    entries=sorted(report["groups"],key=lambda z:abs(z["dollars"]),reverse=True)
    labels=[v["name"] for v in entries][::-1]
    values=[v["dollars"] for v in entries][::-1]
    colors=["#129073" if value>=0 else "#ce624e" for value in values]
    fig=go.Figure(go.Bar(
        y=labels,x=values,orientation="h",marker_color=colors,
        text=[_signed(q) for q in values],textposition="outside",
        hovertemplate="%{y}<br>Allocated difference: $%{x:+,.0f}<extra></extra>"))
    fig.update_layout(title=title,height=345,margin=dict(l=20,r=90,t=45,b=35),
                      xaxis=dict(title="Historical model dollars · vs selected reference",
                                 zeroline=True,showgrid=True,automargin=True),
                      yaxis=dict(automargin=True),showlegend=False,
                      font=dict(size=13))
    st.plotly_chart(fig,width="stretch",
                    key="xai_group_"+report["market"]+"_"+report["reference"])


def _plain(report,market):
    groups=report["groups"]
    positive=sorted((g for g in groups if g["dollars"]>1),key=lambda x:-x["dollars"])
    negative=sorted((g for g in groups if g["dollars"]< -1),key=lambda x:x["dollars"])
    who="advertised seller asking" if market=="ask" else "wholesale auction"
    base=report["base_usd"];target=report["target_usd"]
    st.markdown("**What the real model is saying**")
    st.write("Compared with the selected **hypothetical reference configuration**, "
             "this car's "+who+" estimate is **"+_signed(target-base)+"** different "
             "(from "+_money(base)+" to "+_money(target)+").")
    if positive:
        st.success("What pushes the modeled price **up**: "+
                   "; ".join(x["name"]+" ("+_signed(x["dollars"])+")"
                             for x in positive[:3])+".")
    if negative:
        st.warning("What pushes the modeled price **down**: "+
                   "; ".join(x["name"]+" ("+_signed(x["dollars"])+")"
                             for x in negative[:3])+".")
    if not positive and not negative:
        st.info("This vehicle matches the chosen reference for all modeled groups.")
    st.caption("These are contributions to the model's estimate relative to the "
               "reference—not mechanical repair values, causal effects, discounts "
               "you can demand, or SHAP against the actual training population.")
    st.write("**The reference matters.** Change it above and the attributed "
             "dollars will change even though your car's actual modeled price won't.")


def _card(report,side):
    a,b,c=st.columns(3)
    a.metric("Reference configuration",_money(report["base_usd"]))
    b.metric("Your historical model price",_money(report["target_usd"]))
    c.metric("Explained change",_signed(report["target_usd"]-report["base_usd"]))
    st.caption("Exact enumeration: "+str(report["coalitions_evaluated"])+
               " model coalitions; reconciliation residual "+
               f"${report['reconciliation_error_usd']:+.6f}.")
    _plot_grouped(report,"Which groups move the "+("Ask" if side=="ask" else "Bid")+"?")
    st.dataframe(pd.DataFrame([
        {"Driver":g["name"],"Historical USD":round(g["dollars"],2),
         "Direction":g["direction"]} for g in report["groups"]
    ]),hide_index=True,width="stretch")


def _bridge(saved,adjusted):
    if not adjusted:
        st.info("A date-adjustment receipt is unavailable. The historical neural "
                "model can still be explained on its source-price basis.")
        return
    st.markdown("#### 📆 The price time machine, without the science fiction")
    rows=[]
    for side,r in adjusted.items():
        original=saved.get(side)
        if original is None:continue
        bridge=xai.historical_date_bridge(original["price"],r)
        rows.append({
            "Market":"Seller Ask" if side=="ask" else "Wholesale Bid",
            "Historical model price":bridge["historical_anchor_usd"],
            "Age adjustment":bridge["vehicle_age_effect_usd"],
            "Market-index adjustment":bridge["market_index_effect_usd"],
            "Selected-date price":bridge["date_adjusted_usd"],
        })
    if rows:
        st.dataframe(pd.DataFrame(rows).style.format({
            x:"${:,.0f}" for x in (
              "Historical model price","Age adjustment",
              "Market-index adjustment","Selected-date price")}),
            hide_index=True,width="stretch")
        st.caption("Historical anchor + age effect + market-index effect = "
                   "the EXACT displayed date-adjusted estimate. This is "
                   "arithmetic decomposition, not TreeSHAP or causal economics.")
        st.warning("The retail market-index proxy used for Bid does not validate "
                   "today's wholesale sale prices or turn the Ask–Bid gap "
                   "into dealer profit.")


def _what_if(root,saved,date,adjusted,transport,allow_analogue,champions,token):
    st.markdown("#### 🧰 Change something. Make the *real model* answer.")
    st.caption("No surrogate chatbot and no recalculated market history. "
               "We evaluate new vehicle inputs with the SAME model weights.")
    fields=saved["fields"]
    with st.form("xai_what_if"):
        a,b,c=st.columns(3)
        mileage=a.number_input("What-if mileage · miles",min_value=0,max_value=2000000,
                                value=int(fields["odometer"]),step=5000)
        conds=["unknown","good","excellent","fair","like new","new","salvage"]
        titles=["unknown","clean","rebuilt","salvage","lien","missing","parts only"]
        old_c=str(fields.get("condition") or "unknown").lower()
        old_t=str(fields.get("title_status") or "unknown").lower()
        condition=b.selectbox("What-if condition",conds,
                  index=conds.index(old_c) if old_c in conds else 0)
        title=c.selectbox("What-if title",titles,
                  index=titles.index(old_t) if old_t in titles else 0)
        grade_options=["unknown","1","2","3","4","5"]
        old_grade=str(saved.get("bid_fields",{}).get("auction_grade","unknown"))
        grade=st.selectbox("Wholesale auction pre-sale grade (Bid only)",
                 grade_options,index=grade_options.index(old_grade)
                 if old_grade in grade_options else 0,
                 help="The Bid model uses auction grade; the seller's condition "
                      "dropdown is not secretly substituted for this signal.")
        notes=st.text_area("Alternative seller wording (Ask only)",
                value=saved.get("notes",""),max_chars=8000,height=105,
                help="This changes Ask text embeddings. Bid uses auction facts, "
                     "not seller-description language.")
        run=st.form_submit_button("🔬 Recalculate BOTH models",type="primary",
                                  use_container_width=True)
    if run:
        changed={**fields,"odometer":float(mileage),"condition":condition,
                 "title_status":title}
        results={}
        for side in ("ask","bid"):
            if side not in saved:continue
            try:
                model=_model(root,side,champions)
                if side=="ask":new=model.predict(notes,changed)["price"]
                else:
                    bfields={**saved["bid_fields"],"odometer":float(mileage),
                             "auction_grade":grade,"title_status":title}
                    new=model.predict(bfields)["price"]
                if not np.isfinite(new) or new<=0:raise ValueError("Invalid native price")
                when=None
                if transport is not None:
                    from pricing.neural_experimental_ui import _adjust_price
                    when=_adjust_price(transport,new,changed,side,date,allow_analogue)
                original=float(saved[side]["price"])
                results[side]={
                    "historical_usd":float(new),
                    "historical_change_usd":float(new-original),
                    "date_adjusted_usd":float(when["price"]) if when else None,
                    "date_change_usd":float(when["price"]-adjusted[side]["price"])
                        if when and side in adjusted else None,
                }
            except (OSError,ValueError,RuntimeError,TypeError,KeyError,MemoryError) as error:
                results[side]={"error":str(error)}
        st.session_state["xai_whatif_result"]={"signature":token,"results":results}
    old=st.session_state.get("xai_whatif_result",{})
    if old.get("signature")==token:
        results=old["results"]
        for side,item in results.items():
            st.markdown("**"+("Seller Ask" if side=="ask" else "Wholesale Bid")+"**")
            if "error" in item:
                st.warning("Scenario unavailable: "+item["error"]);continue
            x,y=st.columns(2)
            x.metric("Historical model scenario",_money(item["historical_usd"]),
                     delta=_signed(item["historical_change_usd"]))
            if item["date_adjusted_usd"] is not None:
                y.metric("Selected-date scenario",_money(item["date_adjusted_usd"]),
                         delta=_signed(item["date_change_usd"])
                         if item["date_change_usd"] is not None else None)
        st.caption("A change in the model's predicted price after editing inputs "
                   "does NOT prove the same action changes what an actual buyer pays.")
        st.caption("Bid responds to auction grade and other pre-sale facts, NOT "
                   "the retail condition word or seller advert text.")


def _technical(root,saved,champions,token):
    tab_captum,tab_tree,tab_phrases=st.tabs([
        "🧠 Captum · neural","🌳 TreeSHAP · XGBoost",
        "✍ Full BGE phrase tests"])
    with tab_captum:
        st.write("**Integrated Gradients in dollars** (native PyTorch). "
                 "Continuous numerical features and frozen text embeddings "
                 "are interpolated while categorical codes are held fixed. "
                 "This is a *neural-branch* explanation, not the full Bid ensemble.")
        available=[s for s in ("ask","bid") if s in saved]
        if available:
            side=st.radio("Which network?",available,horizontal=True,key="xai_ig_side",
                          format_func=lambda x:"Ask neural network" if x=="ask" else "Bid DeepFM")
            steps=st.select_slider("Numerical integration steps",
                  options=[16,32,64],value=32,key="xai_ig_steps")
            if st.button("Calculate Captum Integrated Gradients",key="xai_ig_go"):
                try:
                    model=_model(root,side,champions)
                    fields=saved["fields"] if side=="ask" else saved["bid_fields"]
                    ref=xai.benchmark_reference({
                        **fields,"listing_text":saved.get("notes","")},"ask" if side=="ask" else "bid")
                    with st.spinner("Integrating native model gradients, not guessing…"):
                        report=xai.captum_integrated_gradients(
                            model,side,fields,ref,saved.get("notes",""),steps)
                    st.session_state["xai_ig_receipt"]={
                        "token":token,"side":side,"steps":steps,"result":report}
                except Exception as exc:
                    st.warning("Captum diagnostics unavailable: "+str(exc))
        old=st.session_state.get("xai_ig_receipt",{})
        if old.get("token")==token:
            r=old["result"]
            st.caption("Convergence delta: "+
                       f"${r['convergence_delta_usd']:.3f} · "+
                       str(r["steps"])+" integration steps.")
            st.dataframe(pd.DataFrame(r["attributions"]),
                         hide_index=True,width="stretch")
            st.info(r["warning"])
        st.markdown("[Captum methodology](https://captum.ai/api/integrated_gradients.html)")
    with tab_tree:
        if "bid" not in saved:
            st.info("A verified wholesale Bid is needed to explain its Lemon XGBoost branch.")
        else:
            st.write("**Native TreeSHAP:** exactly explains the Lemon XGBoost "
                     "log-price score. We then translate contributions to "
                     "dollars along the exponential path. These are not dollar "
                     "SHAP values and do not explain DeepFM.")
            if st.button("Compute native XGBoost TreeSHAP",key="xai_tree_go"):
                try:
                    model=_model(root,"bid",champions)
                    report=xai.bid_treeshap(model,saved["bid_fields"],
                               saved["bid"]["xgboost_lemon_price"])
                    st.session_state["xai_treeshap_receipt"]={"token":token,"result":report}
                except Exception as exc:
                    st.warning("Native tree breakdown unavailable: "+str(exc))
            old=st.session_state.get("xai_treeshap_receipt",{})
            if old.get("token")==token:
                r=old["result"]
                a,b=st.columns(2)
                a.metric("Lemon XGBoost baseline",_money(r["tree_baseline_usd"]))
                b.metric("Actual Lemon XGBoost",_money(r["tree_prediction_usd"]))
                frame=pd.DataFrame(r["features"])
                if not frame.empty:
                    frame=frame.reindex(frame.path_rescaled_usd.abs().sort_values(
                                         ascending=False).index).head(15)
                    st.dataframe(frame.rename(columns={
                        "name":"Feature","log_price_effect":"Log-price TreeSHAP",
                        "path_rescaled_usd":"Dollar path contribution"}),
                        hide_index=True,width="stretch")
                st.caption(r["note"])
            st.markdown("[XGBoost TreeSHAP documentation](https://xgboost.readthedocs.io/en/stable/python/python_api.html)")
    with tab_phrases:
        if "ask" not in saved:
            st.info("Ask and a seller description are required.")
        else:
            st.write("**Actual phrase-removal experiment.** We remove one "
                     "sentence or word group at a time, rerun the full BGE "
                     "encoder and the native Ask network, and show the prediction "
                     "change. These deltas overlap; they must not be added.")
            limit=st.select_slider("Description detail",options=[4,6,8,12],
                                    value=6,key="xai_phrase_limit")
            if st.button("Run full BGE phrase-removal tests",key="xai_phrase_go"):
                try:
                    model=_model(root,"ask",champions)
                    with st.spinner("Re-embedding each changed advert in native BGE…"):
                        r=xai.phrase_removal_effects(model,saved["fields"],
                             saved.get("notes",""),saved["ask"]["price"],limit)
                    st.session_state["xai_phrase_receipt"]={
                        "token":token,"result":r,"limit":limit}
                except Exception as exc:
                    st.warning("Phrase tests unavailable: "+str(exc))
            old=st.session_state.get("xai_phrase_receipt",{})
            if old.get("token")==token:
                r=old["result"]
                if r["items"]:
                    st.dataframe(pd.DataFrame([{
                        "Removed words":row["fragment"],
                        "New predicted Ask":round(row["without_fragment_usd"]),
                        "Change":round(row["removal_delta_usd"])
                    } for row in r["items"]]),hide_index=True,width="stretch")
                else:st.info("Add some seller description text to test its phrases.")
                st.warning(r.get("warning","The phrase effects are not causal."))


def _receipt_download(state,saved,date):
    def redact(report):
        result={k:v for k,v in report.items()
                if k not in ("fields","description","text","raw")}
        if "groups" in result:
            result["groups"]=[{
                "name":g["name"],"dollars":g["dollars"],
                "direction":g["direction"]} for g in result["groups"]]
        return result
    out={
        "product":"Car Price Studio · Made in Germany",
        "date":str(date),
        "models":{side:saved[side].get("model") for side in ("ask","bid")
                  if side in saved},
        "grouped_explanations":{k:redact(v) for k,v in state["sides"].items()
                                 if "error" not in v},
        "method":"Exact 64-coalition grouped interventional Shapley",
        "description":"omitted for privacy",
        "vehicle_facts":"omitted for privacy",
        "not_causal":True,
        "not_current_market_accuracy":True,
        "warning":"Benchmark comparison, not a statistical prediction interval, "
                  "dealer profit or observed sale price.",
    }
    return json.dumps(out,indent=2,allow_nan=False)


def render(root,saved,date,adjusted,transport,allow_analogue,
           ask_champion=False,bid_champion=False):
    """Called ONLY after the actual main-page estimate is saved and displayed."""
    if not any(s in saved for s in ("ask","bid")):return
    root=Path(root)
    champions={"ask":bool(ask_champion),"bid":bool(bid_champion)}
    token=_signature(saved,date,ask_champion,bid_champion)
    st.markdown(STYLE+"""
<section class="xai-head" aria-label="Explainable AI price diagnostics">
<div class="xai-kicker">CAR PRICE STUDIO · EVIDENCE GARAGE</div>
<h2>🔍 Why this price?</h2>
<p>Because “the algorithm said so” is not a financing strategy.
See the drivers, challenge them, and check the model’s arithmetic.
No guessing, no magic unicorns, just the actual weights.</p>
</section>""",unsafe_allow_html=True)
    st.caption("Explanations are optional. Your initial price is fast; "
               "deeper analyses run only when you ask.")
    if saved.get("bid_analogue"):
        proxy=saved["bid_analogue"]
        st.warning("⚠️ Wholesale explanations concern the age-matched historical "
                   "proxy model year "+str(proxy.get("proxy_model_year"))+
                   ", not actual auctions of your "+str(proxy.get("requested_model_year"))+
                   " generation. The missing generation cannot be explained "
                   "away by sophisticated mathematics.")
    mode=st.radio("Compared with which hypothetical vehicle?",
                  ["Same make, model, year and body; neutralized other details",
                   "A fixed 2010 Toyota Corolla reference"],
                  horizontal=False,key="xai_reference_choice",
                  help="The reference is NOT a dataset average or an independent appraisal.")
    mode_id="same" if mode.startswith("Same") else "benchmark"
    state=st.session_state.get("xai_group_receipt",{})
    if (state.get("signature")!=token or state.get("reference")!=mode_id):
        state={}
    if st.button("🔬 Explain the ACTUAL Ask & Bid prices",
                 type="primary",key="xai_group_go",use_container_width=True):
        reports={}
        for side in ("ask","bid"):
            if side not in saved:continue
            try:
                model=_model(root,side,champions)
                with st.spinner("Auditing the "+side.upper()+
                    " model across 64 counterfactual coalitions…"):
                    if side=="ask":
                        reports["ask"]=xai.explain_ask(
                            model,saved["fields"],saved.get("notes",""),
                            saved["ask"]["price"],mode_id)
                    elif "bid_fields" in saved:
                        reports["bid"]=xai.explain_bid(
                            model,saved["bid_fields"],saved["bid"]["price"],mode_id)
                    else:
                        raise ValueError("Saved historical auction feature receipt missing")
            except Exception as error:
                reports[side]={"error":str(error)}
        state={"signature":token,"reference":mode_id,"sides":reports}
        st.session_state["xai_group_receipt"]=state
        st.session_state.pop("xai_whatif_result",None)
        st.session_state.pop("xai_ig_receipt",None)
        st.session_state.pop("xai_treeshap_receipt",None)
        st.session_state.pop("xai_phrase_receipt",None)
    if not state:
        st.info("Hit **Explain** to calculate a checkable breakdown. "
                "The app will not spend CPU on 128 models until you request it.")
        return
    results=state["sides"]
    tab_plain,tab_drivers,tab_clock,tab_scenarios,tab_technical,tab_share=st.tabs([
        "💡 Simple","📊 Dollar drivers","📅 The price clock",
        "🧪 What if?","🧠 Advanced AI","📥 Research receipt"])
    with tab_plain:
        st.caption("Seller Ask and wholesale Bid answer different economic questions.")
        for side in ("ask","bid"):
            if side not in results:continue
            st.markdown("### "+("🏷 Seller asking" if side=="ask" else "🔨 Wholesale auction"))
            if "error" in results[side]:
                st.warning("Native explanation unavailable: "+results[side]["error"])
            else:_plain(results[side],side)
        st.caption("A large Ask–Bid gap is NOT a risk-free dealer margin. "
                   "Condition, transport, reconditioning, taxes and fees matter.")
    with tab_drivers:
        for side in ("ask","bid"):
            if side not in results:continue
            st.markdown("#### "+("🏷 Seller Ask" if side=="ask" else "🔨 Wholesale Bid"))
            if "error" in results[side]:
                st.warning("Cannot attribute "+side+": "+results[side]["error"])
            else:_card(results[side],side)
        st.warning("Exhaustive grouped SHAP is additive for this chosen "
                   "reference and grouping. It is not causal and can "
                   "evaluate implausible vehicle combinations.")
    with tab_clock:_bridge(saved,adjusted)
    with tab_scenarios:
        _what_if(root,saved,date,adjusted,transport,allow_analogue,champions,token)
    with tab_technical:_technical(root,saved,champions,token)
    with tab_share:
        st.markdown("#### 🔬 Science you can inspect, not a marketing prophecy")
        st.write("The grouped breakdown uses **all 64 coalitions** for six "
                 "predeclared feature groups. The stated hypothetical comparator "
                 "determines the attribution; the total must exactly reconcile "
                 "to the native saved model estimate.")
        st.write("For the Bid ensemble, DeepFM and XGBoost are separate branches; "
                 "native TreeSHAP explains the tree's log-price score, while "
                 "Captum explains continuous numerical paths through the "
                 "neural network. These are not substitute price models.")
        st.write("For Ask, BGE text-embedding attributions and phrase tests are "
                 "different mathematical questions. Phrase removals rerun the "
                 "actual encoder and do not add to dollar SHAP.")
        st.write("**Interpretation:** model behavior, not causality, a calibrated "
                 "confidence interval, a guaranteed retail or wholesale offer, "
                 "or an estimate of realized dealership profit.")
        st.download_button("📥 Download explanation receipt (no advert text)",
            _receipt_download(state,saved,date),
            file_name="car-price-studio-explanation.json",
            mime="application/json",key="xai_receipt_download")
        st.caption("Only your explicit download contains the redacted report. "
                   "No new third-party analytics or external explanation service "
                   "is used to generate these diagnostics.")
