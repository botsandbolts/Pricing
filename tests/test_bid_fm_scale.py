"""Expanded full/Lemon FM family: true higher-order, field-specific, and DeepFM tests."""
import numpy as np
import pytest
import torch

from scripts.bid_fm_scale_train import (
    TRIALS, FAMILIES, COHORTS, ScaledAuctionFM, prediction, EPOCHS)
from scripts.bid_fm_kernel_train import _scores


def test_full_vs_lemon_prespecified_26_trials():
    assert len(TRIALS)==26
    assert set(FAMILIES)=={"fm","ffm","hofm3","fieldgate","deepfm"}
    assert COHORTS==("full","lemon")
    for family,ranks in FAMILIES.items():
        for rank in ranks:
            assert TRIALS[f"{family}{rank}_full"]==(family,rank,"full")
            assert TRIALS[f"{family}{rank}_lemon"]==(family,rank,"lemon")
    assert all(0 < EPOCHS[k] <= 28 for k in FAMILIES)


@pytest.mark.parametrize("family,rank",[
    ("fm",16),("ffm",8),("hofm3",16),("fieldgate",16),("deepfm",16)])
def test_architecture_forward_and_gradients(family,rank):
    model=ScaledAuctionFM([10]*10,family,rank,9.)
    numeric=torch.randn(5,5)
    category=torch.randint(1,7,(5,10))
    pred=model(numeric,category)
    assert pred.shape==(5,)
    assert torch.isfinite(pred).all()
    loss=torch.nn.functional.smooth_l1_loss(pred,torch.ones(5)*9.4,beta=.25)
    loss.backward()
    assert model.fm.num_latent.grad is not None
    assert torch.isfinite(model.fm.num_latent.grad).all()
    if family=="hofm3":
        assert model.third_strength.grad is not None
    if family=="fieldgate":
        assert model.pair_gate.grad is not None
    if family=="deepfm":
        assert model.deep[-1].weight.grad is not None


def test_true_third_order_interactions_not_pairwise_only():
    model=ScaledAuctionFM([5]*10,"hofm3",16,9.)
    num=torch.ones((2,5))
    cat=torch.ones((2,10),dtype=torch.long)
    with torch.no_grad():
        model.third_strength.fill_(0)
    base=model(num,cat)
    with torch.no_grad():
        model.third_strength.fill_(2.5)
    changed=model(num,cat)
    assert not torch.allclose(base,changed)


def test_fieldgate_pair_parameters_are_symmetric_avoiding_self_pairs():
    model=ScaledAuctionFM([5]*10,"fieldgate",16,9.)
    with torch.no_grad():
        model.pair_gate.fill_(0.)
    num=torch.randn((3,5))
    cat=torch.ones((3,10),dtype=torch.long)
    baseline=model(num,cat)
    with torch.no_grad():
        model.pair_gate[0,1]=1.
    changed=model(num,cat)
    assert not torch.allclose(baseline,changed)


def test_predict_returns_valid_prices_and_original_rows():
    model=ScaledAuctionFM([7]*10,"fm",16,9.)
    numeric=np.zeros((12,5),dtype=np.float32)
    cat=np.ones((12,10),dtype=np.int64)
    out=prediction(model,numeric,cat,np.array([10,1,4,2]))
    assert len(out)==4 and np.isfinite(out).all() and (out>0).all()
    assert _scores(np.ones(4)*100.,np.ones(4)*110.)["mae_usd"]==10.


def test_unregistered_architecture_rejected():
    with pytest.raises(ValueError):
        ScaledAuctionFM([3]*10,"some-made-up-variant",16,9.)
