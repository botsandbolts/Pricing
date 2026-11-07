"""Auditable, model-faithful explanations for the TWO native pricing engines.

- Exact exhaustive 6-group interventional Shapley against a disclosed reference
  *configuration*, not conditional observational SHAP or a causal explanation.
- Captum Integrated Gradients for the continuous PyTorch branches (CPU/on demand).
- XGBoost native TreeSHAP for the deployed wholesale tree, in log USD and
  path-rescaled dollars. No surrogate, fitted explanation model or conversions.
- Genuine BGE re-embedding for each optional deleted listing phrase.
- Closed-form historical-age/market-index dollar bridge.

The user's actual model weights are used for every result. Never copy old
MiniLM/TreeSHAP explanations onto the neural BGE or DeepFM prediction.
"""
from __future__ import annotations

from itertools import combinations
from math import comb, exp, expm1, isfinite, log
import re

import numpy as np

MODEL_GROUPS=(
    ("Make, model & year",("manufacturer","model","year")),
    ("Mileage",("odometer",)),
    ("Body & drivetrain",("type","fuel","transmission","drive","cylinders")),
    ("Condition & title",("condition","title_status")),
    ("Location & color",("state","paint_color")),
)
ASK_GROUPS=MODEL_GROUPS+(("Advert description",("listing_text",)),)
BID_GROUPS=MODEL_GROUPS+(("Auction grade",("auction_grade",)),)
REFERENCE={
    "year":2010,"manufacturer":"Toyota","model":"Corolla",
    "odometer":95000.,"state":"ca","type":"sedan",
    "condition":"good","fuel":"gas","transmission":"automatic",
    "drive":"fwd","title_status":"clean","cylinders":"4 cylinders",
    "paint_color":"silver","auction_grade":"3",
    "listing_text":"Well maintained used car. Standard equipment.",
}
NUMERIC_ASK={
    "year":"Model year","age_2021":"Age in 2021",
    "log_miles":"Log mileage","miles_per_year":"Mileage per year",
    "log_description_chars":"Advert length",
    "log_description_words":"Advert words",
    "description_punctuation":"Advert punctuation",
}
NUMERIC_BID={
    "year":"Model year","age_at_auction":"Age at auction",
    "log_mileage":"Log mileage",
    "log_mileage_per_vehicle_age":"Annualized mileage",
    "auction_grade":"Pre-sale condition grade",
}


def benchmark_reference(actual,market,mode="same"):
    """Chosen *hypothetical* reference car; never claim population averages."""
    if market not in ("ask","bid") or mode not in ("same","benchmark"):
        raise ValueError("Unsupported explanation reference")
    ref=dict(REFERENCE)
    if mode=="same":
        for name in ("manufacturer","model","year","type"):
            if name in actual:ref[name]=actual[name]
    # The archive spans only 2014-2015 for Bid; hold the auction date
    # fixed, so hypothetical hybrid coalitions cannot move into the future.
    if market=="bid":
        ref["auction_date"]=actual["auction_date"]
        if "year" in actual and mode=="same":
            ref["year"]=actual["year"]
    return ref


def exact_group_shapley(predictor,actual,reference,groups,observed_price,
                        max_groups=7):
    """Exhaustive interventional SHAP for a transparent finite set of groups.

    Given f(x), a fixed benchmark x0 and n groups, enumerate 2**n coalitions.
    Attribution = sum_S (f(x_{S+i})-f(x_S)) / [n * C(n-1, |S|)].
    For these 6 groups the sum is exact (up to floating point), unlike
    independent leave-one-out deltas.
    """
    groups=tuple((str(name),tuple(keys)) for name,keys in groups)
    n=len(groups)
    if not 1<=n<=max_groups:
        raise ValueError("Explanation group budget exceeded")
    all_keys=[k for _,ks in groups for k in ks]
    if len(set(all_keys))!=len(all_keys):
        raise ValueError("An input belongs to two explanation groups")
    if not all(k in actual and k in reference for k in all_keys):
        raise ValueError("Missing required full-model explanation input")
    observed=float(observed_price)
    if not isfinite(observed) or observed<=0:
        raise ValueError("Cannot attribute an invalid model price")
    values={}
    for mask in range(1<<n):
        row=dict(reference)
        for i,(_,keys) in enumerate(groups):
            if mask & (1<<i):
                row.update({key:actual[key] for key in keys})
        value=float(predictor(row))
        if not isfinite(value) or not 0<value<=1e7:
            raise ValueError("Invalid native model coalition prediction")
        values[mask]=value
    target=values[(1<<n)-1]
    if abs(target-observed)>max(.5,.00002*observed):
        raise ValueError("Saved price is not reproduced by current native weights")
    attributions=[]
    for i,(label,keys) in enumerate(groups):
        total=0.
        for mask in range(1<<n):
            if mask & (1<<i):continue
            size=mask.bit_count()
            total+=(values[mask | (1<<i)]-values[mask])/(n*comb(n-1,size))
        attributions.append({
            "name":label,"fields":list(keys),"dollars":float(total),
            "direction":"increase" if total>0 else "decrease" if total<0 else "no change",
        })
    base=values[0]
    residual=base+sum(z["dollars"] for z in attributions)-target
    if abs(residual)>max(.05,1e-8*observed):
        raise ValueError("Grouped Shapley does not reconcile to actual model")
    return {
        "method":"Exhaustive grouped interventional Shapley",
        "groups":attributions,"base_usd":float(base),"target_usd":float(target),
        "reconciliation_error_usd":float(residual),
        "coalitions_evaluated":len(values),
        "reference_scope":"hypothetical comparator, never a typical or mean car",
        "unit":"original historical USD", "causal":False,
        "model_being_explained":"actual served native checkpoint",
    }


def explain_ask(model,fields,description,expected,mode="same"):
    """Reuse BGE embeddings (once per unique text), never 64 encoder passes."""
    actual={**fields,"listing_text":str(description or "")[:8000]}
    reference=benchmark_reference(actual,"ask",mode)
    embeddings={}
    for value in {actual["listing_text"],reference["listing_text"]}:
        embeddings[value]=model.prepare_text(value)
    def predict(row):
        prepared=embeddings[row["listing_text"]]
        return model.predict_prepared(row,prepared)["price"]
    result=exact_group_shapley(predict,actual,reference,ASK_GROUPS,expected)
    result.update(market="ask",reference=mode,
                  reference_description_redacted=True,
                  text_encoder_runs=len(embeddings))
    return result


def explain_bid(model,fields,expected,mode="same"):
    actual=dict(fields)
    reference=benchmark_reference(actual,"bid",mode)
    result=exact_group_shapley(
        lambda row:model.predict(row)["price"],
        actual,reference,BID_GROUPS,expected)
    result.update(market="bid",reference=mode,
                  reference_description_redacted=True,text_encoder_runs=0)
    return result


def historical_date_bridge(historical,receipt):
    """Exact dollar steps from the SAME stored TimeTransport receipt."""
    h=float(historical)
    aged=h*float(receipt["age_factor"])
    market=float(receipt["price"])-aged
    final=float(receipt["price"])
    if not all(map(isfinite,(h,aged,market,final))) or h<=0 or final<=0:
        raise ValueError("Invalid date-adjusted market receipt")
    if abs((h+(aged-h)+market)-final)>.01:
        raise ValueError("Date bridge failed price accounting")
    return {
        "historical_anchor_usd":h,
        "vehicle_age_effect_usd":aged-h,
        "market_index_effect_usd":market,
        "date_adjusted_usd":final,
        "date":receipt["date"],
        "age_multiplier":float(receipt["age_factor"]),
        "market_multiplier":float(receipt["market_factor"]),
        "source":"The SAME time-transport receipt shown in the calculator",
        "warning":"Arithmetic transport, not causal SHAP or verified current-market error",
    }


def bid_treeshap(model,fields,published_xgb_price,expected_alpha=None):
    """True XGBoost TreeSHAP; never attribute a fitted surrogate model.

    The tree predicts LOG USD. The signed contributions add to log-price;
    their dollar equivalents use the exact exponential path identity
    exp(b+sum(phi))-exp(b) = sum(phi)*[exp(b+sum(phi))-exp(b)]/sum(phi).
    These path-rescaled dollars are *not* SHAP values of the dollar model.
    """
    from pricing.hierarchy_sparse_live import design_matrix
    from pricing.neural_bid_blend_live import CATS,_norm
    original=getattr(model,"original",model)
    numeric,cat,raw,unknown=original.features(fields)
    pack={**{k:_norm(fields.get(k)) for k in CATS[:8]},
          "odometer":float(fields["odometer"]),
          "auction_grade":float(raw[-1]) if np.isfinite(raw[-1]) else None}
    booster=original.xgb
    matrix,_=design_matrix(booster,original.xgb_schema,None,pack,float(raw[1]))
    parts=np.asarray(booster.predict(matrix,pred_contribs=True))
    if parts.ndim!=2 or parts.shape[0]!=1 or parts.shape[1]<2:
        raise ValueError("Unexpected XGBoost contribution dimensions")
    raw_score=float(booster.predict(matrix,output_margin=True)[0])
    base_log=float(parts[0,-1])
    phi=np.asarray(parts[0,:-1],dtype=float)
    if not np.isfinite(phi).all() or abs(base_log+float(phi.sum())-raw_score)>.002:
        raise ValueError("Native TreeSHAP fails additive log-score parity")
    target=float(exp(min(raw_score,log(1e7))))
    expected=float(published_xgb_price)
    if abs(target-expected)>max(.5,expected*.0001):
        raise ValueError("TreeSHAP was not run on the displayed Lemon XGBoost")
    delta=raw_score-base_log
    scale=float(exp(base_log)*expm1(delta)/delta) if abs(delta)>1e-7 else float(exp(base_log))
    dollar=phi*scale
    baseline=float(exp(base_log))
    if abs(baseline+float(dollar.sum())-target)>max(.15,target*.00001):
        raise ValueError("TreeSHAP dollar bridge failed price reconciliation")
    names=matrix.feature_names or [f"feature_{i}" for i in range(len(phi))]
    if len(names)!=len(phi):raise ValueError("XGBoost feature name dimension mismatch")
    return {
        "method":"Native XGBoost TreeSHAP, additive in log price",
        "features":[{"name":str(name),"log_price_effect":float(v),
                     "path_rescaled_usd":float(d)} for name,v,d in zip(names,phi,dollar)],
        "tree_baseline_usd":baseline,
        "tree_prediction_usd":target,
        "log_margin":raw_score,
        "log_additivity_error":base_log+float(phi.sum())-raw_score,
        "dollar_reconciliation_error":baseline+float(dollar.sum())-target,
        "note":"Dollars are exact exponential path-rescaling of log TreeSHAP, not dollar-space SHAP or causal effects.",
        "model_being_explained":"native Lemon XGBoost from the active Bid ensemble",
    }


def captum_integrated_gradients(model,market,fields,reference,description="",
                                 steps=32):
    """Continuous input IG in dollars with categorical embeddings FIXED.

    Captum is imported only after the user requests technical diagnostics.
    A frozen BGE vector is used; we do NOT backprop through SentenceTransformer.
    Categorical variables require separate grouped Shapley, not fake integer IG.
    """
    from captum.attr import IntegratedGradients
    import torch
    steps=int(steps)
    if steps not in (16,32,64):raise ValueError("Unsupported IG calculation budget")
    if market=="ask":
        original=getattr(model,"original",model)
        text=original.prepare_text(description)
        reference_text=original.prepare_text(reference["listing_text"])
        num,cat,vec=original.encode_inputs(fields,text)
        num0,_,vec0=original.encode_inputs(reference,reference_text)
        net=getattr(model,"model",None)
        if net is None:net=original.price_network
        cats=torch.from_numpy(cat.astype(np.int64))
        x=(torch.from_numpy(num),torch.from_numpy(vec))
        baseline=(torch.from_numpy(num0),torch.from_numpy(vec0))
        def forward(numbers,text_vec):
            codes=cats.expand(numbers.shape[0],-1)
            log_price=net(numbers,codes,text_vec)[:,0]
            return torch.exp(torch.clamp(log_price,log(100),log(1e6)))
        numeric_names=list(NUMERIC_ASK)
    elif market=="bid":
        original=getattr(model,"original",model)
        num,cat,raw,unknown=original.features(fields)
        num0,_,_,_=original.features(reference)
        net=model.model
        cats=torch.from_numpy(cat.astype(np.int64))
        x=torch.from_numpy(num)
        baseline=torch.from_numpy(num0)
        def forward(numbers):
            codes=cats.expand(numbers.shape[0],-1)
            log_price=net(numbers,codes).reshape(-1)
            return torch.exp(torch.clamp(log_price,log(100),log(1e6)))
        numeric_names=list(NUMERIC_BID)
    else:
        raise ValueError("Unknown model branch")
    result,delta=IntegratedGradients(forward).attribute(
        x,baselines=baseline,n_steps=steps,internal_batch_size=16,
        return_convergence_delta=True)
    if market=="ask":
        numeric=result[0].detach().cpu().numpy()[0]
        text_dollars=float(result[1].sum().item())
    else:
        numeric=result.detach().cpu().numpy()[0]
        text_dollars=None
    with torch.inference_mode():
        price=float(forward(*(x if market=="ask" else (x,)))[0].item())
        ref=float(forward(*(baseline if market=="ask" else (baseline,)))[0].item())
    d=float(delta.flatten()[0].item())
    if not all(map(isfinite,[price,ref,d,*numeric])) or abs(d)>max(5.,.015*price):
        raise ValueError("Integrated gradients did not converge sufficiently; refuse unsupported chart")
    rows=[{"name":numname.replace("_"," ").title(),
           "dollars":float(value)} for numname,value in zip(numeric_names,numeric)]
    if text_dollars is not None:rows.append({"name":"BGE advert embedding (all 384 dimensions)",
                                           "dollars":text_dollars})
    total=sum(x["dollars"] for x in rows)
    if abs(ref+total-price)>max(5.,.015*price):
        raise ValueError("Captum IG failed its dollar completeness check")
    return {
        "method":"Captum Integrated Gradients (continuous numeric/text inputs only)",
        "market":market,"price_branch_usd":price,"reference_branch_usd":ref,
        "attributions":rows,"convergence_delta_usd":d,"steps":steps,
        "fixed_categorical_inputs":True,
        "text_encoder_trained":False,
        "scope":"Native neural branch ONLY; other ensemble weights excluded",
        "warning":"Fixed categorical embeddings; interpolation in normalized features or BGE space is not a realistic vehicle counterfactual.",
    }


def phrase_fragments(text,max_chunks=8):
    """Public text only, no raw car price or personal data stored in reports."""
    raw=str(text or "")[:8000]
    if not raw.strip():return []
    max_chunks=max(1,min(int(max_chunks),12))
    parts=list(re.finditer(r"[^.!?;\n]+[.!?;\n]*|\S+",raw))
    segments=[(m.start(),m.end(),m.group(0).strip()) for m in parts
              if m.group(0).strip()]
    if len(segments)>max_chunks:
        # Group adjacent sentences to stay within the bounded encoder budget.
        chunks=[]
        for q in np.array_split(np.arange(len(segments)),max_chunks):
            a=segments[int(q[0])][0];b=segments[int(q[-1])][1]
            chunks.append((a,b,raw[a:b].strip()))
        return chunks
    if len(segments)==1:
        words=list(re.finditer(r"\S+",raw))
        if len(words)>3:
            chunks=[]
            for q in np.array_split(np.arange(len(words)),min(max_chunks,6)):
                a=words[int(q[0])].start();b=words[int(q[-1])].end()
                chunks.append((a,b,raw[a:b].strip()))
            return chunks
    return segments[:max_chunks]


def phrase_removal_effects(model,fields,description,original_price,
                           max_chunks=8):
    """Leave-one-fragment-out model *sensitivity*, NOT Shapley or causality."""
    raw=str(description or "")[:8000]
    chunks=phrase_fragments(raw,max_chunks=max_chunks)
    if not chunks:return {"method":"Model phrase removal sensitivity","items":[],
                          "encoder_runs":0}
    original=float(original_price)
    check=float(model.predict(raw,fields)["price"])
    if abs(check-original)>max(.5,.00002*original):
        raise ValueError("Phrase analysis does not match actual displayed Ask price")
    out=[]
    for left,right,fragment in chunks:
        altered=(raw[:left]+raw[right:]).strip()
        price=float(model.predict(altered,fields)["price"])
        if not isfinite(price) or price<=0:raise ValueError("Invalid phrase counterfactual")
        out.append({
            "fragment":fragment[:160],"without_fragment_usd":price,
            "removal_delta_usd":price-original,
            "interpretation":"Change in prediction if this segment is removed",
        })
    return {"method":"Actual BGE re-embed + full neural network, leave-one-fragment-out",
            "items":out,"original_price_usd":original,
            "encoder_runs":len(chunks)+1,"causal":False,
            "warning":"Individual removal effects may overlap and DO NOT add up to total price. Never interpret an advert rewrite as a guaranteed transaction gain."}
