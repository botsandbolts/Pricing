"""Smoke and contract tests for the optional Neural Lab alpha serving path."""
from pathlib import Path
import numpy as np
import pytest
from pricing.neural_experimental_live import numerical,categoricals,available,FILES
from scripts.deepcross_aft_tune import TunedDeepCross,config
from scripts.neural_ask_prepare import CATS,NUMS

def test_original_feature_geometry_and_numeric_values():
    assert len(CATS)==11 and len(NUMS)==7
    f={"year":2015,"odometer":95000}
    n=numerical(f,"Good car!")
    assert n.shape==(7,) and np.isfinite(n).all()
    assert n[0]==2015 and n[1]==6
    assert np.isclose(n[2],np.log1p(95000))
    assert n[5]==np.log1p(2)
    assert n[6]==np.log1p(1)

def test_unsupported_generation_and_bad_mileage_rejected():
    with pytest.raises(ValueError):numerical({"year":2023,"odometer":12000},"")
    with pytest.raises(ValueError):numerical({"year":2015,"odometer":-1},"")
    with pytest.raises(ValueError):numerical({"year":2015},"")

def test_unknown_category_id_and_token_shapes():
    maps={c:{"known":2} for c in CATS}
    output=categoricals({"manufacturer":"known","condition":"brand-new"},maps)
    assert output.shape==(1,11)
    assert output[0,0]==2 and output[0,-1]==1

def test_fail_closed_uninstalled_bundle(tmp_path):
    assert available(tmp_path) is False
    for name in FILES[:-1]:
        (tmp_path/"models/neural-experimental-ask").mkdir(parents=True,exist_ok=True)
        (tmp_path/"models/neural-experimental-ask"/name).write_text("")
    assert available(tmp_path) is False

def test_selected_deepcross_has_valid_forward_geometry():
    import torch
    cards=[3]*11
    net=TunedDeepCross(cards,8,24,config("dropout12"))
    pred=net(torch.zeros((2,8)),torch.ones((2,11),dtype=torch.long),
             torch.zeros((2,24)))
    assert pred.shape==(2,2) and torch.isfinite(pred).all()
