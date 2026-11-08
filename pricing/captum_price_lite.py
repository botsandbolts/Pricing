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
INTEGRATION_STEPS=24  # Single fixed budget: 24 sequential Gauss–Legendre evaluations


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
    # Captum MUST explain exactly the same native checkpoint that priced
    # the receipt; never a prior champion or a surrogate neuron.
    from pricing.overnight_bohb_live import (
        available as bohb_available, load_ask as bohb_load_ask,
        load_bid as bohb_load_bid)
    if side=="ask":
        from pricing.ray_lemon75_ask_live import (
            available as lemon_available, load as lemon_load)
        if lemon_available(root):
            return lemon_load(root)
        if bohb_available(root,"ask"):
            return bohb_load_ask(root)
        if ask_champion:
            from pricing.neural_champion_live import load_ask
            return load_ask(root)
        from pricing.neural_ask_embedding_live import load
        return load(root)
    if side=="bid":
        if bohb_available(root,"bid"):
            return bohb_load_bid(root)
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


def explain(root,saved,side,steps=INTEGRATION_STEPS,*,ask_champion=False,bid_champion=False):
    if steps != INTEGRATION_STEPS:
        raise ValueError("Captum uses exactly 24 integration steps; custom budgets are disabled")
    if side not in ("ask","bid") or side not in saved:
        raise ValueError("Calculate a saved price first")
    if not _LANE.acquire(blocking=False):
        raise RuntimeError("The explanation lane is busy. Retry shortly.")
    import time
    from pricing.memory_watch import record as log_memory
    started=time.monotonic()
    log_memory("captum_start",detailed=True)
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
        log_memory("captum_24step_complete",
                   elapsed_ms=1000*(time.monotonic()-started),detailed=True)
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
    except Exception:
        log_memory("captum_failed",
                   elapsed_ms=1000*(time.monotonic()-started),detailed=True)
        raise
    finally:
        _LANE.release()



def selected_date_breakdown(saved, side, attribution, receipt):
    """Translate actual 24-step Captum dollar terms using the public price clock.

    The original head is still evaluated in training-period dollars. The
    selected-day quote is historical full-model price * depreciation *
    aggregate market-index ratio. Those multipliers are constant for this
    vehicle and attribution path. Bid also retains its original XGBoost
    component, which Captum does not attempt to explain.
    """
    if side not in ("ask", "bid") or attribution.get("market") != side:
        raise ValueError("Explanation market does not match the quote")
    if not attribution.get("adequate"):
        raise ValueError("Attribution has not passed convergence checks")
    if receipt.get("side") != side or not receipt.get("date"):
        raise ValueError("Selected-date receipt is unavailable")
    source=float(saved[side]["price"])
    anchor=float(receipt["anchor_price_reference_usd"])
    age=float(receipt["age_factor"])
    market=float(receipt["market_factor"])
    quoted=float(receipt["price"])
    if not all(math.isfinite(v) and v>0 for v in
               (source,anchor,age,market,quoted)):
        raise ValueError("Nonfinite pricing or time multipliers")
    multiplier=age*market
    tolerance=max(.05,quoted*1e-6)
    if abs(source-anchor)>max(.05,source*1e-6):
        raise ValueError("The transport anchor does not match this actual model quote")
    if abs(source*multiplier-quoted)>tolerance:
        raise ValueError("Selected-date quote did not pass exact transport parity")
    neural=float(attribution["neural_usd"])
    baseline=float(attribution["anchor_usd"])
    items=attribution["items"]
    if not all(math.isfinite(float(row["dollars"])) for row in items):
        raise ValueError("Nonfinite Captum contributions")
    if side=="ask":
        if abs(neural-source)>max(.05,source*1e-5):
            raise ValueError("Captum and the full Ask model disagree")
        xgb=0.
    else:
        prediction=saved["bid"]
        weight=float(prediction.get("neural_weight",.75))  # Verified original Bid uses 75/25.
        original_xgb=float(prediction["xgboost_lemon_price"])
        if not (math.isfinite(weight) and .5<=weight<=1. and
                math.isfinite(original_xgb) and original_xgb>0):
            raise ValueError("Unverified Bid component weights")
        xgb=(1.-weight)*original_xgb
        if abs(neural+xgb-source)>max(.05,source*1e-5):
            raise ValueError("DeepFM + original Lemon XGBoost does not match the Bid quote")
    signed_residual=(neural-baseline-sum(float(x["dollars"]) for x in items))*multiplier
    selected_items=[{"feature":row["feature"],
                     "dollars":float(row["dollars"])*multiplier}
                    for row in items]
    selected_baseline=baseline*multiplier
    selected_xgb=xgb*multiplier
    reconstructed=(selected_baseline+sum(x["dollars"] for x in selected_items)
                   +signed_residual+selected_xgb)
    if abs(reconstructed-quoted)>tolerance:
        raise ValueError("Selected-date explanation does not reconcile to the quote")
    return {
        "side":side,"date":str(receipt["date"]),"quote_usd":quoted,
        "reference_usd":selected_baseline,"items":selected_items,
        "xgboost_share_usd":selected_xgb,"integration_residual_usd":signed_residual,
        "age_factor":age,"market_factor":market,"multiplier":multiplier,
        "reconciliation_error_usd":abs(reconstructed-quoted),
        "bid_has_fixed_xgb":side=="bid",
    }


def render(root,saved,*,ask_champion=False,bid_champion=False,adjusted=None):
    import streamlit as st
    sides=[s for s in ("ask","bid") if s in saved and
           (s!="bid" or "bid_fields" in saved) and
           (adjusted is None or s in adjusted)]
    # An INT8 head can differ slightly from native Captum pricing, so
    # do not falsely reconcile a reference-weight attribution to a
    # differently quantized user quote. Bid Captum remains available.
    from pricing.ray_lemon75_ask_live import available as lemon_available
    if "ask" in sides and lemon_available(root):
        sides.remove("ask")
        st.caption("🔬 Lemon75 Ask uses audited INT8 ONNX inference. "
                   "Exact-price Captum is temporarily limited to native "
                   "models; we do not present native gradients as if they "
                   "exactly explained a quantized quote. "
                   "See the Lemon science panel for its methodology.")
    if not sides:
        if adjusted is not None:
            st.info("The selected-date price must be available before it can be explained.")
        return
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
            # Fixed 24-step precision: no slider and no lower-precision option.
            # Only 1 interpolation point is processed at a time.
            steps=INTEGRATION_STEPS
            st.caption("🔬 **24 Gauss–Legendre integration steps · fixed**. "
                       "RAM limiter: **internal_batch_size=1**, one vehicle, "
                       "no BGE backpropagation. No precision slider to babysit.")
            go=st.form_submit_button("🔬 Explain THIS price · Captum",
                type="primary",use_container_width=True)
        # A stored explanation belongs to ONE chosen market and the fixed
        # 24-step precision. Switching Ask/Bid must never show stale bars.
        rid=receipt_id(saved)+f":{side}:{steps}"
        if go:
            try:
                with st.spinner("Doing tiny numerical integrations, not inviting another model…"):
                    result=explain(root,saved,side,steps,
                        ask_champion=ask_champion,bid_champion=bid_champion)
                st.session_state["captum_lite_result"]={"id":rid,"data":result}
            except Exception as exc:
                # Do not swallow the real traceback: user-supplied Streamlit
                # logs must identify the failure without exposing advert text,
                # model weights or service credentials in the browser.
                import logging
                logging.getLogger("car_price_studio.captum").exception(
                    "On-demand Captum 24-step attribution failed: market=%s; error_type=%s",
                    side,type(exc).__name__)
                st.session_state.pop("captum_lite_result",None)
                detail=str(exc)
                if "Recalculate Ask once" in detail or "Recalculate Bid" in detail:
                    cause="Original vehicle representation is missing."
                    action="Calculate the vehicle price once more, then press Explain."
                elif "does not reproduce the saved price" in detail:
                    cause="Model parity check failed."
                    action="Your price is preserved; reload the page and run a new quote."
                elif "lane is busy" in detail:
                    cause="Explanation lane is busy."
                    action="Press Explain again once the other calculation finishes."
                elif isinstance(exc,ImportError):
                    cause="Captum or a required package is unavailable."
                    action="Check the Streamlit deployment dependency logs."
                elif isinstance(exc,MemoryError):
                    cause="Not enough memory for gradient attribution."
                    action="The price remains valid; check the hosting memory limit."
                elif "Numerical attribution is invalid" in detail:
                    cause="The model returned non-finite or invalid gradient values."
                    action="This car cannot be explained reliably with this method."
                else:
                    cause="A neural explanation calculation failed ("+type(exc).__name__+")."
                    action="Check the latest Streamlit log after clicking Explain."
                st.warning("⚠️ Captum couldn't produce a reliable explanation. "
                           +cause+" Your original price is unchanged.")
                st.caption(action+" Technical details are recorded in the server log.")
        current=st.session_state.get("captum_lite_result")
        if current and current.get("id")==rid:
            r=current["data"]
            if adjusted is None or side not in adjusted:
                st.warning("Recalculate a selected-date quote before viewing this explanation.")
                return
            if not r["adequate"]:
                st.warning("The 24-step attribution did not converge "
                           "sufficiently; individual driver figures are "
                           "hidden rather than inventing an explanation.")
                return
            try:
                dated=selected_date_breakdown(saved,side,r,adjusted[side])
            except (ValueError,TypeError,KeyError,OverflowError) as exc:
                import logging
                logging.getLogger("car_price_studio.captum").warning(
                    "Date-adjusted explanation rejected: %s",type(exc).__name__)
                st.warning("The explanation cannot be reconciled to the displayed "
                           "date-adjusted price. The quote is unchanged; "
                           "individual contribution figures are hidden.")
                return
            st.markdown("#### 📅 Why this price on "+dated["date"]+"?")
            st.metric("Explained "+("seller asking" if side=="ask" else "wholesale auction")+
                      " · selected-date scenario",
                      "$"+format(dated["quote_usd"],",.0f"))
            st.caption("These are the SAME selected-date dollars as your price "
                       "above—not unadjusted training-era dollars. Vehicle ageing ×"+
                       format(dated["age_factor"],".3f")+"; market index ×"+
                       format(dated["market_factor"],".3f")+
                       ". Both multipliers apply to each dollar contribution.")
            st.markdown("**Historical model effects, rescaled to your selected date**")
            st.caption("Captum compares the actual car with standardized numeric "
                       "reference inputs; category inputs and BGE text stay fixed. "
                       "The reference is NOT an actual historic vehicle or transaction. "
                       "These rescaled Captum effects hold the chosen car's age "
                       "and market-adjustment multipliers fixed, so they are "
                       "not full end-to-end causal effects of changing car age.")
            for item in sorted(dated["items"],key=lambda x:-abs(x["dollars"])):
                left,right=st.columns([3,1])
                left.write(("🟢 " if item["dollars"]>=0 else "🟠 ")+item["feature"])
                right.metric("Selected-date effect",
                             "$"+format(item["dollars"],"+,.0f"))
            a,b=st.columns(2)
            a.metric("Reference · selected-day dollars",
                     "$"+format(dated["reference_usd"],",.0f"))
            if dated["bid_has_fixed_xgb"]:
                b.metric("Lemon XGBoost · fixed Bid share",
                         "$"+format(dated["xgboost_share_usd"],",.0f"))
                st.caption("Bid is a blend: Captum explains ONLY the DeepFM "
                           "numerical inputs. The original Lemon XGBoost "
                           "component is carried to the same date, but not "
                           "misrepresented as a Captum feature effect.")
            else:
                b.metric("Captum numerical features",
                         "$"+format(sum(x["dollars"] for x in dated["items"]),"+,.0f"))
            st.caption("Arithmetic: reference + feature effects"+
                       (" + XGBoost ensemble share" if side=="bid" else "")+
                       " + numerical integration remainder = displayed "+
                       ("Bid" if side=="bid" else "Ask")+" price.")
            st.caption("Integration remainder: $"+
                       format(dated["integration_residual_usd"],"+,.2f")+
                       " · date-price reconciliation error: $"+
                       format(dated["reconciliation_error_usd"],",.2f")+
                       " · "+str(r["steps"])+" sequential steps · batch **1**")
            st.info(r["scope"]+" "+r["warning"])
            st.caption("The age adjustment uses a learned depreciation curve "
                       "and the market adjustment an aggregate price index. "
                       "Neither validates a current individual-car transaction "
                       "or makes the price movement causal.")
