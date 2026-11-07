"""Low-RAM Captum on the existing verified neural Ask/Bid heads only.

One-car, one-step minibatches; no BGE backward pass; no surrogate,
model copies, cross-model imports or background work.
"""
import hashlib
import json
import math
import threading

import numpy as np

_LANE=threading.BoundedSemaphore(1)
ASK_NAMES=("Model year","Age in 2021","Log mileage","Mileage per year",
           "Log description characters","Log description words","Punctuation")
BID_NAMES=("Model year","Auction vehicle age","Log mileage",
           "Log mileage per year","Auction grade")
STEPS=(8,16,24)


def receipt_id(saved):
    details={
       "model_ask":saved.get("ask",{}).get("model"),
       "model_bid":saved.get("bid",{}).get("model"),
       "ask":saved.get("ask",{}).get("price"),
       "bid":saved.get("bid",{}).get("price"),
       "fields":saved.get("fields"),"bid_fields":saved.get("bid_fields"),
       "notes":saved.get("notes"),"date":saved.get("date")}
    return hashlib.sha256(json.dumps(details,sort_keys=True,default=str).encode()).hexdigest()


def _model(root,side,ask_champion,bid_champion):
    if side=="ask":
        if ask_champion:
            from pricing.neural_champion_live import load_ask
            return load_ask(root)
        from pricing.neural_ask_embedding_live import load
        return load(root)
    if side=="bid":
        if bid_champion:
            from pricing.neural_champion_live import load_bid
            return load_bid(root)
        from pricing.neural_bid_blend_live import load
        return load(root)
    raise ValueError("Choose the Ask or Bid model")


def _native_inputs(model,side,saved):
    import torch
    original=getattr(model,"original",model)
    if side=="ask":
        if "ask_embedding" not in saved:
            raise ValueError("Recalculate Ask once to reuse its original BGE embedding.")
        prepared={"embedding":np.asarray(saved["ask_embedding"],dtype=np.float32),
                  "raw":saved["notes"]}
        num,cat,txt=original.encode_inputs(saved["fields"],prepared)
        network=getattr(model,"model",None)
        if network is None:network=original.price_network
        fixed=(torch.as_tensor(cat,dtype=torch.int64),
               torch.as_tensor(txt,dtype=torch.float32))
        weight=1.
        saved_price=float(saved["ask"]["price"])
        scope="Full Ask pricing head; the frozen BGE text vector and categories stay fixed."
        names=ASK_NAMES
    else:
        if "bid_fields" not in saved:
            raise ValueError("Recalculate Bid to save the actual auction features.")
        num,cat,_raw,_unknown=original.features(saved["bid_fields"])
        network=getattr(model,"model",None)
        if network is None:raise ValueError("Native DeepFM head is missing")
        fixed=(torch.as_tensor(cat,dtype=torch.int64),)
        weight=float(saved["bid"].get("neural_weight",getattr(model,"alpha",.75)))
        if not .5<=weight<=1.:raise ValueError("Unverified Bid blend weight.")
        saved_price=float(saved["bid"]["deepfm_price"])*weight
        scope="Only the weighted DeepFM part of Bid; XGBoost and categories stay fixed."
        names=BID_NAMES
    x=torch.as_tensor(num,dtype=torch.float32)
    if tuple(x.shape)!=(1,len(names)) or not np.isfinite(x.numpy()).all():
        raise ValueError("Invalid original numerical feature dimensions")
    if not math.isfinite(saved_price) or saved_price<=0:
        raise ValueError("Invalid recorded neural prediction")
    return network,x,fixed,weight,saved_price,names,scope


def explain(root,saved,side,steps=8,*,ask_champion=False,bid_champion=False):
    if steps not in STEPS:raise ValueError("Choose 8, 16 or 24 steps")
    if side not in ("ask","bid") or side not in saved:
        raise ValueError("Calculate a saved price first")
    if not _LANE.acquire(blocking=False):
        raise RuntimeError("The explanation lane is busy. Retry shortly.")
    try:
        import torch
        from captum.attr import IntegratedGradients
        model=_model(root,side,ask_champion,bid_champion)
        network,x,fixed,weight,saved_price,names,scope=_native_inputs(model,side,saved)
        if network.training:raise ValueError("Training-mode neural network is unsafe")
        baseline=torch.zeros_like(x)
        low,high=math.log(100.),math.log(1_000_000.)
        def dollars(numbers,*constant):
            raw=(network(numbers,constant[0],constant[1])[:,0]
                 if side=="ask" else network(numbers,constant[0]).reshape(-1))
            return torch.exp(torch.clamp(raw,min=low,max=high))*weight
        with torch.no_grad():
            actual=float(dollars(x,*fixed).item())
            anchor=float(dollars(baseline,*fixed).item())
        if not all(map(math.isfinite,(actual,anchor))):
            raise ValueError("Nonfinite model prediction")
        if abs(actual-saved_price)>max(.75,.0001*saved_price):
            raise ValueError("Captum's model does not reproduce the saved price")
        with torch.enable_grad():
            attrs,convergence=IntegratedGradients(
                dollars,multiply_by_inputs=True).attribute(
                    x,baselines=baseline,additional_forward_args=fixed,
                    n_steps=steps,method="gausslegendre",internal_batch_size=1,
                    return_convergence_delta=True)
        values=np.asarray(attrs.detach().cpu().numpy().reshape(-1),dtype=float)
        delta=float(convergence.detach().cpu().item())
        if len(values)!=len(names) or not np.isfinite(values).all() or not math.isfinite(delta):
            raise ValueError("Numerical attribution is invalid")
        residual=abs(anchor+float(values.sum())-actual)
        return {
            "market":side,"scope":scope,"steps":steps,"batch_size":1,
            "anchor_usd":anchor,"neural_usd":actual,
            "delta_usd":delta,"reconciliation_error_usd":residual,
            "adequate":residual<=max(25.,.10*abs(actual-anchor)),
            "items":[{"feature":name,"dollars":float(value)}
                     for name,value in zip(names,values)],
            "reference":"Standardized numeric medians (zero); same categories and BGE vector. The synthetic baseline need not be a real car.",
            "warning":"Captum explains local neural sensitivity, NOT causal dollar value, dealer profit, text words or the XGBoost contribution.",
        }
    finally:
        _LANE.release()


def render(root,saved,*,ask_champion=False,bid_champion=False):
    import streamlit as st
    sides=[s for s in ("ask","bid") if s in saved and
           (s!="bid" or "bid_fields" in saved)]
    if not sides:return
    st.markdown("""
<div style="background:linear-gradient(110deg,#152f49,#166975);
border-radius:17px;padding:18px 21px;margin:18px 0 8px;color:white">
<div style="font-size:.73rem;letter-spacing:.12em;font-weight:750;color:#a5f3e2">
CAR PRICE STUDIO · EXPLAINABLE AI · MADE IN GERMANY</div>
<div style="font-size:1.45rem;font-weight:800;margin:6px 0;color:white">
🔬 Why THAT price?</div>
<div style="color:#e8fafa">The model has opinions. Now it has to show its work.
No second elephant-sized neural network in the garage.</div>
</div>""",unsafe_allow_html=True)
    # The explanation form is ALWAYS visible after a successful quote.
    # A collapsed expander made this flagship feature effectively invisible
    # on mobile even though all Captum integration tests passed.
    with st.container(border=True):
        st.markdown("**🧠 Captum · explain one real model prediction**")
        st.caption("Tap the button below to investigate this price. "
                   "Your first appraisal remains fast: Captum runs only when "
                   "you request it, with one interpolation point in memory.")
        with st.form("captum_light_form"):
            side=st.radio("Explain which market?",sides,horizontal=True,
                format_func=lambda s:"🏷 Seller Ask · neural price" if s=="ask"
                  else "🔨 Wholesale Bid · DeepFM only")
            steps=st.select_slider("Precision versus CPU time",options=[8,16,24],
                value=8,format_func=lambda v:{
                    8:"⚡ Quick · 8 steps",
                    16:"⚖️ Balanced · 16 steps",
                    24:"🔍 More detail · 24 steps"}[v])
            st.caption("RAM limiter: **internal_batch_size=1**, one vehicle, "
                       "Gauss–Legendre, no BGE backpropagation.")
            go=st.form_submit_button("🔬 Explain THIS price · Captum",
                type="primary",use_container_width=True)
        # A stored explanation belongs to ONE chosen market and step budget.
        # Changing Ask to Bid (or 8 to 24 steps) must never show stale bars.
        rid=receipt_id(saved)+f":{side}:{steps}"
        if go:
            try:
                with st.spinner("Doing tiny numerical integrations, not inviting another model…"):
                    result=explain(root,saved,side,steps,
                        ask_champion=ask_champion,bid_champion=bid_champion)
                st.session_state["captum_lite_result"]={"id":rid,"data":result}
            except (ValueError,RuntimeError,TypeError,KeyError,ImportError,
                    OSError,MemoryError):
                st.session_state.pop("captum_lite_result",None)
                st.warning("Explanation unavailable. Your original model price "
                           "is unchanged; retry with fewer steps.")
        current=st.session_state.get("captum_lite_result")
        if current and current.get("id")==rid:
            r=current["data"]
            if not r["adequate"]:
                st.warning("This low-step approximation didn't converge "
                           "sufficiently. Try 16 or 24 steps; individual "
                           "driver figures are hidden until they pass.")
            else:
                st.markdown("#### 💵 How numerical features move the neural estimate")
                st.caption("Against normalized training-median numbers; "
                           "categories and the BGE text embedding stay fixed.")
                for item in sorted(r["items"],key=lambda x:-abs(x["dollars"])):
                    left,right=st.columns([3,1])
                    left.write(("🟢 " if item["dollars"]>=0 else "🟠 ")+item["feature"])
                    right.metric("Model effect","$"+format(item["dollars"],"+,.0f"))
            a,b=st.columns(2)
            a.metric("Neural baseline","$"+format(r["anchor_usd"],",.0f"))
            b.metric("Actual neural output","$"+format(r["neural_usd"],",.0f"))
            st.caption("Integration reconciliation error: $"+
                format(r["reconciliation_error_usd"],",.2f")+
                " · "+str(r["steps"])+" sequential steps · batch **1**")
            st.info(r["scope"]+" "+r["warning"])
            st.caption(r["reference"])
