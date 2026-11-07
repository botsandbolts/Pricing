"""Short-lived neural inference worker. Launched by neural_isolated_serving.

In this process ONLY, load one existing verified model (native PyTorch), run
one actual price or three bounded sensitivities, write one JSON line and exit.
"""
import json
import math
import sys


def _ready(root,market):
    if market=="ask":
        from pricing.neural_champion_live import ask_available
        from pricing.neural_ask_embedding_live import available
        return ask_available(root),available(root)
    from pricing.neural_champion_live import bid_available
    from pricing.neural_bid_blend_live import available
    return bid_available(root),available(root)


def _model(root,market):
    champion,old=_ready(root,market)
    if not (champion or old):
        raise FileNotFoundError("No parity-verified native checkpoint for this market")
    if market=="ask":
        if champion:
            from pricing.neural_champion_live import CVaR95Asking
            return CVaR95Asking(root)
        from pricing.neural_ask_embedding_live import BGEDeepCrossAsk
        return BGEDeepCrossAsk(root)
    if champion:
        from pricing.neural_champion_live import CVaR95Wholesale
        return CVaR95Wholesale(root)
    from pricing.neural_bid_blend_live import VerifiedWholesaleBlend
    return VerifiedWholesaleBlend(root)


def _valid_price(value):
    p=float(value)
    if not math.isfinite(p) or not 100<=p<=1_000_000:
        raise ValueError("Model returned invalid historical price")
    return p


def evaluate(root,payload):
    import torch
    torch.set_num_threads(2)
    market=payload["market"]
    fields=dict(payload["fields"])
    text=str(payload.get("description") or "")[:8000]
    operation=payload["operation"]
    model=_model(root,market)
    if market=="ask":
        prepared=model.prepare_text(text)
        def predict(f):
            return model.predict_prepared(f,prepared)
    else:
        def predict(f):
            return model.predict(f)
    base=predict(fields)
    original=_valid_price(base["price"])
    if operation=="predict":
        return {"ok":True,"market":market,"operation":operation,
                "prediction":base}
    expected=payload.get("expected")
    if expected is None or abs(float(expected)-original)>max(.5,original*.00005):
        raise ValueError("Original estimate changed; request a fresh valuation")
    odometer=float(fields["odometer"])
    tests=[]
    for delta in (-10000.,10000.):
        next_miles=max(0.,min(2000000.,odometer+delta))
        if next_miles!=odometer:
            tests.append(("Mileage "+("%+d"%int(delta))+" mi",
                          {**fields,"odometer":next_miles}))
    if market=="ask":
        current=str(fields.get("title_status") or "unknown")
        alternative="clean" if current!="clean" else "rebuilt"
        tests.append(("Title status: "+alternative,
                      {**fields,"title_status":alternative}))
    else:
        original_grade=str(fields.get("auction_grade") or "unknown")
        new_grade="4" if original_grade!="4" else "3"
        tests.append(("Auction grade: "+new_grade,
                      {**fields,"auction_grade":new_grade}))
    out=[]
    for label,f in tests[:3]:
        estimate=_valid_price(predict(f)["price"])
        out.append({"scenario":label,
                    "historical_price_usd":estimate,
                    "difference_usd":estimate-original})
    return {"ok":True,"market":market,"operation":operation,
            "sensitivity":{
                "market":market,"model":base.get("model","Verified native neural model"),
                "baseline_usd":original,"changes":out,
                "calls":len(out)+1,
                "causal":False,
                "method":"Native model finite changes, reused Ask BGE embedding",
                "notice":"Scenario sensitivity is not a causal repair value or guaranteed transaction price."}}


def main():
    import pathlib
    try:
        payload=json.load(sys.stdin)
        if not isinstance(payload,dict) or payload.get("market") not in ("ask","bid")            or payload.get("operation") not in ("predict","sensitivity"):
            raise ValueError("Invalid worker request")
        reply=evaluate(pathlib.Path.cwd(),payload)
    except (MemoryError,ValueError,KeyError,TypeError,RuntimeError,
            OSError,ImportError) as exc:
        # Public responses exclude input text and internal stack traces.
        reply={"ok":False,"market":payload.get("market") if
               isinstance(locals().get("payload"),dict) else None,
               "operation":payload.get("operation") if
               isinstance(locals().get("payload"),dict) else None,
               "message":"Native model unavailable or unsupported input. "
                         "Retry or choose a different supported vehicle."}
    print("CPS_NATIVE_RESULT:"+json.dumps(reply,allow_nan=False,default=str),
          flush=True)


if __name__=="__main__":
    main()
