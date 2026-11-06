"""Frozen auction split guardrails and architecture sanity checks."""
import numpy as np
import pytest
import torch
from scripts.ft_bid_tune import BASE,TRIALS,settings,model_for,ids_sha
from scripts.neural_ask_train import FTTransformer

def test_registered_grid_and_original_control():
    assert len(TRIALS)==29
    assert settings("control")==BASE
    assert isinstance(model_for("control",[5,7],4,1),FTTransformer)
    with pytest.raises(ValueError):
        settings("chosen_after_test")

def test_every_variant_legal():
    for name in TRIALS:
        c=settings(name)
        assert c["d"]%c["heads"]==0 and c["layers"]>0 and c["ff"]>0
        assert 0<=c["drop"]<1 and 0<=c["token_drop"]<1
        assert c["wd"]>=0 and c["huber"]>0

def test_architectural_variations_forward():
    for name in ("width32","depth3","heads8","postnorm","token12","wide64depth3"):
        net=model_for(name,[4,7,5],5,1)
        out=net(torch.zeros((2,5)),torch.ones((2,3),dtype=torch.long),torch.zeros((2,1)))
        assert out.shape==(2,2)
        assert torch.isfinite(out).all()

def test_holdout_fingerprints_are_order_invariant():
    assert ids_sha(["b","a"])==ids_sha(["a","b"])
    assert ids_sha(["a","b"])!=ids_sha(["a","c"])
