"""Offline BOHB release-contract checks. No original private data required."""
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from pricing.overnight_bohb_live import FOLDERS,WINNERS,available
from scripts.release_overnight_bohb import _check_metrics, _verify_select
from scripts.overnight_bohb_neural import build_ask,StructuredBid


def test_exact_release_identities_and_fail_closed_missing_bundle(tmp_path):
    assert WINNERS=={"ask":36,"bid":116}
    assert FOLDERS["ask"]!=FOLDERS["bid"]
    assert not available(tmp_path,"ask")
    assert not available(tmp_path,"bid")


def test_ask_native_structure_from_archived_configuration():
    cfg={"projection":64,"project_layers":2,"proj_dropout":0.,
         "cross_layers":4,"rank":32,"experts":4,"width":192,
         "blocks":1,"dropout":.18,"activation":"silu",
         "output_dropout":.15,"embed_dropout":.05,"cross_scale":.5}
    model=build_ask([10]*11,384,"ask_structure",cfg)
    with torch.inference_mode():
        p=model(torch.randn(2,7),torch.ones(2,11,dtype=torch.int64),
                torch.randn(2,384))
    assert p.shape==(2,2) and torch.isfinite(p).all()


def test_bid_native_structure_never_has_bge_encoder():
    cfg={"rank":16,"width1":192,"width2":128,"depth":2,
         "dropout":.03,"activation":"gelu","pairgate":False}
    model=StructuredBid([10]*10,9.,cfg)
    with torch.inference_mode():
        p=model(torch.randn(2,5),torch.ones(2,10,dtype=torch.int64))
    assert p.shape==(2,) and torch.isfinite(p).all()
    assert not hasattr(model,"encoder")
    assert all("bge" not in name.lower() for name,_ in model.named_modules())


def test_replay_metrics_fail_on_any_drift():
    metric={"CVaR95_absolute_USD":8436.36,"MAE_USD":1858.03,
            "RMSE_USD":2910.2}
    _check_metrics(metric,metric)
    with pytest.raises(ValueError,match="metrics"):
        _check_metrics(metric,{**metric,"CVaR95_absolute_USD":8400.})


def test_reject_missing_final_research_evidence(tmp_path):
    class Args:
        report=tmp_path/"report.json"
        winner=tmp_path/"candidate"
    Args.report.write_text(json.dumps({"status":"complete","records":[]}))
    Args.winner.mkdir()
    (Args.winner/"summary.json").write_text("{}")
    (Args.winner/"result.json").write_text("{}")
    with pytest.raises((ValueError,KeyError)):
        _verify_select("ask",Args)
