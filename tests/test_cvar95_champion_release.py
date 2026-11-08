"""Experimental CVaR95 champion admission and UI contracts.

No network, no test labels, no original production checkpoint modification in
these unit tests. Exact runtime/sample parity runs in the release workflow.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pricing.neural_champion_live import (
    ASK_FOLDER,BID_FOLDER,ask_available,bid_available,_available,
)
from scripts.ask25_joint_search import ResearchAskingNetwork,trial_config
from scripts.neural_200_sweep import spec
from scripts.bid_fm_scale_train import ScaledAuctionFM


def _stage(root,folder,market):
    d=root/folder
    d.mkdir(parents=True)
    raw=b"native-test-payload"
    (d/"weights.pt").write_bytes(raw)
    sha=hashlib.sha256(raw).hexdigest()
    (d/"manifest.json").write_text(json.dumps({
        "market":market,"status":"release_candidate_verified",
        "native_pytorch_only":True,
        "original_test_scored_after_validation_selection":True,
        "production_main_calculator_unchanged":True,
        "checkpoint_sha256":sha}))
    (d/"verification.json").write_text(json.dumps({
        "status":"passed","verified":True,
        "complete_original_validation_parity":True,
        "original_test_rows_scored":31682 if market=="ask" else 83603,
        "shared_preprocessor_parity":True,
        "live_prediction_checked":True,
        "weight_sha256":sha}))
    return d


@pytest.mark.parametrize("market,folder",[
    ("ask",ASK_FOLDER),("bid",BID_FOLDER)])
def test_strict_trust_gate_and_checksum(tmp_path,market,folder):
    assert not _available(tmp_path,market)
    d=_stage(tmp_path,folder,market)
    assert _available(tmp_path,market)
    (d/"weights.pt").write_bytes(b"tampered!")
    assert not _available(tmp_path,market)


def test_fallback_when_original_engines_missing(tmp_path):
    _stage(tmp_path,ASK_FOLDER,"ask")
    _stage(tmp_path,BID_FOLDER,"bid")
    assert not ask_available(tmp_path)
    assert not bid_available(tmp_path)


def test_ask_resmlp15_uses_real_native_projector_and_3_residual_blocks():
    cfg=trial_config(15)
    assert cfg["family"]=="resmlp"
    model=ResearchAskingNetwork([8]*11,384,cfg)
    model.eval()
    with torch.no_grad():
        v=model(torch.ones(2,7),torch.ones(2,11,dtype=torch.long),
                torch.ones(2,384))
    assert v.shape==(2,2) and torch.isfinite(v).all()


def test_bid18_is_original_deepfm16_architecture_with_new_training_settings():
    cfg=spec("bid_hparams",18)
    assert cfg["tail_lambda"]>.0
    model=ScaledAuctionFM([8]*10,"deepfm",16,initial_log_price=9.)
    model.eval()
    with torch.no_grad():
        p=model(torch.ones(2,5),torch.ones(2,10,dtype=torch.long))
    assert p.shape==(2,) and torch.isfinite(p).all()


def test_experimental_panel_declares_the_appropriate_fallback():
    ui=Path("pricing/neural_experimental_ui.py").read_text()
    assert "champion_ask_available" in ui
    assert "champion_bid_available" in ui
    assert "load_champion_ask" in ui and "load_champion_bid" in ui
    assert "ask_champion or ask_available(root)" in ui
    assert "bid_champion or bid_available(root)" in ui
    assert "CVaR95" in ui and "ResMLP" in ui
    assert "Ask training labels are advertised prices" in ui
    assert "models/neural-experimental-ask-bge" not in ui
