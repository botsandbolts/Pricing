"""Memory-bounded, on-demand Captum IG for LIVE car-pricing checkpoints.

Captum's 50-step default can construct large batches: this module forces
internal_batch_size=1, uses at most 16 Gauss-Legendre integration steps, and
attributes ONLY 7 Ask / 5 Bid numerical inputs. The pinned BGE encoder is
NEVER differentiated or re-instantiated, and categorical codes stay fixed.

No eager Torch/Captum import, alternate model weights, SHAP background data,
subprocesses, reloading/unloading, inference-mode wrappers around gradients,
training, or feature-level causal claims.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ASK_LABELS=(
    "Model year",
    "Vehicle age in 2021",
    "Mileage (log scale)",
    "Miles per model-year",
    "Advert length (log)",
    "Advert word count (log)",
    "Advert punctuation (log)",
)
BID_LABELS=(
    "Model year",
    "Age at historical auction",
    "Mileage (log scale)",
    "Annualized mileage (log)",
    "Pre-sale auction grade",
)
PRESETS={"⚡ Quick · 8 steps":8,"🔬 Careful · 16 steps":16}


def stamp(saved,market,steps):
    """Bind results to the exact submitted car, source model and price."""
    p={
        "market":market,"n_steps":steps,"fields":saved.get("fields"),
        "auction_features":saved.get("bid_fields"),
        "date":saved.get("date"),"notes":saved.get("notes"),
        "model":saved.get(market,{}).get("model"),
        "historical_price":saved.get(market,{}).get("price"),
    }
    return hashlib.sha256(json.dumps(p,sort_keys=True,default=str).encode()).hexdigest()


def _load_native(root,market,champion):
    # Exact serving cache functions, not independent model instances.
    if market=="ask":
        if champion:
            from pricing.neural_champion_live import load_ask
            return load_ask(root)
        from pricing.neural_ask_embedding_live import load
        return load(root)
    if market=="bid":
        if champion:
            from pricing.neural_champion_live import load_bid
            return load_bid(root)
        from pricing.neural_bid_blend_live import load
        return load(root)
    raise ValueError("Unsupported pricing market")


def explain(root,saved,market,*,champion,steps=8,model=None):
    """Explain one serving network's continuous inputs, single-row chunks.

    Ask output is the actual historical seller price. Bid output is the
    DeepFM BRANCH in historical wholesale dollars (NOT the 90/10 ensemble).
    Numerical standardized zeros are a disclosed median-feature reference,
    NOT a plausible reference vehicle or dataset expectation.
    """
    if steps not in (8,16):
        raise ValueError("Only memory-bounded 8/16-step Captum is supported")
    if market not in ("ask","bid") or market not in saved:
        raise ValueError("No checked Ask/Bid prediction to explain")
    import numpy as np
    import torch
    from captum.attr import IntegratedGradients

    source=model if model is not None else _load_native(root,market,champion)
    original=getattr(source,"original",source)
    if market=="ask":
        fields=saved["fields"]
        prepared=saved.get("ask_prepared")
        if prepared is None:
            # Old saved receipts only: reuse exactly the cached original BGE,
            # and store its tiny (1,384) text vector for the next run.
            prepared=source.prepare_text(saved.get("notes",""))
            saved["ask_prepared"]=prepared
        numeric,categorical,embedding=original.encode_inputs(fields,prepared)
        net=getattr(source,"model",None)
        if net is None:net=original.price_network
        categories=torch.from_numpy(np.asarray(categorical,dtype=np.int64))
        fixed_embedding=torch.from_numpy(np.asarray(embedding,dtype=np.float32))
        labels=ASK_LABELS
        expected=float(saved["ask"]["price"])

        def forward(x):
            # One-row explanation. The BGE output is a FIXED leaf tensor.
            batch_cats=categories.expand(x.shape[0],-1)
            batch_text=fixed_embedding.expand(x.shape[0],-1)
            log=net(x,batch_cats,batch_text)[:,0]
            return torch.exp(torch.clamp(log,math.log(100.),math.log(1e6)))
    else:
        if "bid_fields" not in saved:
            raise ValueError("No saved historical auction features for Bid")
        numeric,categorical,_,_=original.features(saved["bid_fields"])
        net=source.model
        categories=torch.from_numpy(np.asarray(categorical,dtype=np.int64))
        labels=BID_LABELS
        expected=float(saved["bid"]["deepfm_price"])

        def forward(x):
            batch_cats=categories.expand(x.shape[0],-1)
            log=net(x,batch_cats).reshape(-1)
            return torch.exp(torch.clamp(log,math.log(100.),math.log(1e6)))

    if not (math.isfinite(expected) and 100<=expected<=1e6):
        raise ValueError("Saved native model price is invalid")
    data=np.asarray(numeric,dtype=np.float32)
    if data.shape!=(1,len(labels)) or not np.isfinite(data).all():
        raise ValueError("Native feature geometry or values do not match release")
    inputs=torch.from_numpy(data.copy())
    baselines=torch.zeros_like(inputs)  # standardized train-only medians
    # DO NOT call torch.inference_mode here; Captum needs first derivatives.
    # No model parameters are mutated; price nets are already in eval mode.
    with torch.enable_grad():
        with torch.no_grad():
            actual=float(forward(inputs).item())
            reference=float(forward(baselines).item())
        if abs(actual-expected)>max(1.,expected*.0002):
            raise ValueError("Live network does not reproduce the saved price")
        ig=IntegratedGradients(forward,multiply_by_inputs=True)
        attribution,delta=ig.attribute(
            inputs,baselines=baselines,n_steps=steps,
            method="gausslegendre",internal_batch_size=1,
            return_convergence_delta=True)
    attributions=np.asarray(attribution.detach().cpu().numpy(),dtype=float)
    residual=float(delta.detach().cpu().reshape(-1)[0].item())
    if attributions.shape!=(1,len(labels)) or not np.isfinite(attributions).all() or not math.isfinite(residual):
        raise ValueError("Captum returned invalid attribution dimensions/dollars")
    allocated=float(attributions.sum())
    difference=actual-reference
    if abs(allocated-difference-residual)>max(.50,.0002*max(100.,actual)):
        raise ValueError("Captum's completeness delta did not reconcile")
    # Do not claim exact feature attribution if a short integration is coarse.
    tolerance=max(10.,.03*abs(difference),.002*actual)
    passed=abs(residual)<=tolerance
    return {
        "market":market,
        "model":saved[market].get("model","native verified model"),
        "method":"Captum Integrated Gradients · numerical features only",
        "n_steps":steps,
        "internal_batch_size":1,
        "quadrature":"gausslegendre",
        "observed_native_branch_usd":actual,
        "median_numeric_reference_usd":reference,
        "numeric_difference_usd":difference,
        "attribution_sum_usd":allocated,
        "convergence_delta_usd":residual,
        "completeness_passed":passed,
        "items":[{"feature":label,"dollars":float(x)}
                 for label,x in zip(labels,attributions[0])],
        "scope":(
            "Ask network, with BGE advert embedding and categorical IDs held fixed"
            if market=="ask" else
            "DeepFM neural BRANCH only, with categorical IDs fixed; Lemon XGBoost excluded"
        ),
        "baseline":"Train-standardized numerical zeros, holding the current BGE and categories fixed",
        "causal":False,
        "warning":"Local path attribution, not a repair valuation, transaction forecast, or full causal price explanation",
    }


def render(root,saved,*,ask_champion=False,bid_champion=False):
    """A compact native Streamlit workbench, never active before a real price."""
    import streamlit as st
    if not any(m in saved for m in ("ask","bid")):
        return
    st.markdown("""
<section aria-label="Captum Explainable AI" style="
background:linear-gradient(118deg,#18304e,#274957 65%,#21594e);
color:white;border:1px solid #7ca8b0;border-radius:18px;
padding:18px 22px;margin:20px 0 12px">
<div style="font-size:.75rem;font-weight:700;letter-spacing:.11em;color:#a8ebd7">
CAR PRICE STUDIO · AI EXPLAINABILITY · MADE IN GERMANY
</div><div style="font-size:1.5rem;font-weight:800;color:#fff;margin:6px 0">
🔍 Why this price? · The tiny detective
</div><div style="color:#e1f2f2;line-height:1.5">
The AI has opinions. We have receipts. No 50-step memory buffet.
</div></section>""",unsafe_allow_html=True)
    st.caption("Real Captum Integrated Gradients. Nothing runs until you click "
               "Explain. One model, one vehicle, one numerical feature vector.")
    available=[m for m in ("ask","bid")
               if m in saved and (m=="ask" or "bid_fields" in saved)]
    if not available:
        st.info("Run a verified asking or wholesale estimate to unlock explanations.")
        return
    with st.form("cps_lowmem_captum_controls",clear_on_submit=False):
        col1,col2=st.columns(2)
        market=col1.selectbox(
            "Explain this model",
            available,
            format_func=lambda m:"🏷 Seller Ask · entire ResMLP" if m=="ask"
                else "🔨 Wholesale Bid · DeepFM branch",
            key="cps_captum_market")
        mode=col2.selectbox(
            "CPU / detail budget",list(PRESETS),
            help="Quick is 8 sequential Gauss–Legendre steps. "
                 "Careful is 16. Both use exactly ONE item per internal batch.",
            key="cps_captum_mode")
        submit=st.form_submit_button(
            "⚡ Explain with Captum (1-at-a-time)",
            type="primary",use_container_width=True)
    steps=PRESETS[mode]
    current=stamp(saved,market,steps)
    if submit:
        try:
            with st.spinner("Questioning the neural network, one tiny step at a time…"):
                report=explain(root,saved,market,
                       champion=ask_champion if market=="ask" else bid_champion,
                       steps=steps)
            st.session_state["cps_captum_receipt"]={
                "stamp":current,"report":report}
        except (ImportError,OSError,ValueError,TypeError,KeyError,RuntimeError,
                MemoryError) as exc:
            st.session_state.pop("cps_captum_receipt",None)
            st.warning("Captum explanation unavailable for this appraisal. "
                       "Your validated model and price are unchanged.")
            st.caption("Try Quick mode or review the native model results above. "
                       "No substitute explanation has been invented.")
    state=st.session_state.get("cps_captum_receipt")
    if not state or state.get("stamp")!=current:
        with st.expander("🧠 How does the tiny detective save memory?"):
            st.markdown("**Captum runs only on your click.** We compute "
                        "8 or 16 steps with `internal_batch_size=1`, "
                        "use the native model already in memory, and attribute "
                        "just 7 Ask / 5 Bid numeric inputs. No BGE gradients, "
                        "large background datasets, model duplication or new "
                        "inference service.")
            st.caption("We use `method='gausslegendre'`, "
                       "`multiply_by_inputs=True` and "
                       "`return_convergence_delta=True`. "
                       "The delta measures integration error—not a model "
                       "confidence interval.")
        return
    report=state["report"]
    st.markdown("#### 🧾 Your neural network's explanation receipt")
    a,b=st.columns(2)
    a.metric("Historical native neural price",
             "${:,.0f}".format(report["observed_native_branch_usd"]))
    b.metric("Median-numeric reference",
             "${:,.0f}".format(report["median_numeric_reference_usd"]))
    if report["completeness_passed"]:
        st.success("✅ Numerical attribution check passed. "
                   "The only thing missing is a tiny detective hat.")
    else:
        st.warning("⚠️ This low-step approximation has a large convergence "
                   "error. Try **Careful · 16 steps** before interpreting the "
                   "individual feature amounts.")
    st.caption("Numeric contribution sum: ${:+,.0f} · "
               "Captum convergence delta: ${:+,.2f}".format(
                    report["attribution_sum_usd"],report["convergence_delta_usd"]))
    import pandas as pd
    frame=pd.DataFrame([
        {"Feature":p["feature"],
         "Effect vs reference ($)":round(p["dollars"],2),
         "Direction":"Raises model price" if p["dollars"]>0
                     else "Lowers model price" if p["dollars"]<0 else "Unchanged"}
        for p in report["items"]])
    frame=frame.reindex(frame["Effect vs reference ($)"].abs().sort_values(
                        ascending=False).index)
    if report["completeness_passed"]:
        st.bar_chart(frame.set_index("Feature")[["Effect vs reference ($)"]],
                     horizontal=True)
    st.dataframe(frame,hide_index=True,use_container_width=True)
    st.caption("**What this explains:** "+report["scope"]+". The reference "
               "sets numerical inputs to their training-standardized median, "
               "not necessarily a physically possible car. Other features "
               "and the original BGE description stay fixed.")
    if market=="bid":
        st.info("The total Bid price also contains Lemon XGBoost. "
                "These bars explain only the DeepFM neural component, "
                "not the whole 90/10 blend.")
    st.warning("These are model sensitivities, **not causal evidence**, "
               "negotiating discounts, realized sale prices or prediction intervals.")
    with st.expander("🔬 Scientific settings & validation"):
        st.json({k:report[k] for k in (
            "method","n_steps","internal_batch_size","quadrature",
            "numeric_difference_usd","attribution_sum_usd",
            "convergence_delta_usd","completeness_passed",
            "scope","causal")})
        st.caption("No original vehicle description or embedding is "
                   "included in the diagnostic receipt.")
