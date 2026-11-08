"""Actual in-process native serving smoke against the production Python runtime.

Unlike the browser-only CI, this loads the pinned public Hugging Face BGE
encoder and BOTH verified real checkpoint bundles, runs a real valuation,
and records peak RSS. No retraining, surrogate, model reloading or subprocesses
are used by the app. This script runs only in GitHub Actions, not Streamlit.
"""
import json
import logging
import math
import os
import resource
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CAR={
    "year":2015,"manufacturer":"Toyota","model":"Corolla","odometer":95000.0,
    "state":"ca","type":"sedan","condition":"good","fuel":"gas",
    "transmission":"automatic","drive":"fwd","title_status":"clean",
    "cylinders":"4 cylinders","paint_color":"white",
}
DESCRIPTION="Well maintained. Clean title. New tires."
AUCTION={**CAR,"auction_date":"2015-01-15","auction_grade":"3"}


def phase(stage,start=None):
    rss=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    ms=round(1000*(time.monotonic()-start)) if start is not None else None
    print(json.dumps({"stage":stage,"peak_rss_mb":rss,
                      "elapsed_ms":ms,"python":sys.version.split()[0]}),
          flush=True)


def main():
    os.environ.setdefault("OMP_NUM_THREADS","2")
    os.environ.setdefault("MKL_NUM_THREADS","2")
    os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
    import torch
    torch.set_num_threads(2)
    from pricing.neural_champion_live import (
        ask_available as old_ask_available,
        bid_available as old_bid_available,
        load_ask as old_load_ask,
        load_bid as old_load_bid)
    from pricing.overnight_bohb_live import (
        available as bohb_available,
        load_ask as bohb_load_ask,
        load_bid as bohb_load_bid)
    if bohb_available(ROOT,"ask") and bohb_available(ROOT,"bid"):
        ask_available=lambda _root:bohb_available(_root,"ask")
        bid_available=lambda _root:bohb_available(_root,"bid")
        load_ask=bohb_load_ask
        load_bid=bohb_load_bid
        release="overnight-BOHB-ask36-bid116"
    else:
        ask_available=old_ask_available
        bid_available=old_bid_available
        load_ask=old_load_ask
        load_bid=old_load_bid
        release="original-CVaR95-research-champions"
    assert ask_available(ROOT), "Original-test verified Ask weights unavailable"
    assert bid_available(ROOT), "Original-test verified Bid weights unavailable"
    print(json.dumps({"stage":"active_release","model":release}),flush=True)
    phase("start")
    started=time.monotonic()
    ask=load_ask(ROOT)
    phase("ask_load_complete",started)
    started=time.monotonic()
    prepared=ask.prepare_text(DESCRIPTION)
    phase("ask_bge_complete",started)
    started=time.monotonic()
    asking=ask.predict_prepared(CAR,prepared)
    phase("ask_price_complete",started)
    assert asking["market"]=="ask" and math.isfinite(float(asking["price"]))
    assert 100<float(asking["price"])<1_000_000
    started=time.monotonic()
    bid=load_bid(ROOT)
    phase("bid_load_complete",started)
    started=time.monotonic()
    wholesale=bid.predict(AUCTION)
    phase("bid_price_complete",started)
    assert wholesale["market"]=="bid" and math.isfinite(float(wholesale["price"]))
    assert 100<float(wholesale["price"])<1_000_000
    assert "deepfm_price" in wholesale and "xgboost_lemon_price" in wholesale
    # Probe the actual cached serving route a second time, WITHOUT reloading.
    again=ask.predict_prepared(CAR,prepared)
    assert math.isclose(float(again["price"]),float(asking["price"]),rel_tol=1e-6)
    phase("both_native_prices_verified")
    print("Native production Python appraisal PASS (private synthetic vehicle)",flush=True)


if __name__=="__main__":
    main()
