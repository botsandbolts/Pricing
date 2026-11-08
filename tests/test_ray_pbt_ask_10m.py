"""Synthetic QA for the targeted ten-minute Ray PBT rerun."""
from types import SimpleNamespace

import pytest
import torch

import scripts.ray_pbt_ask as core
from scripts.ray_pbt_ask_10m import (
    nearby,safe_options,valid_neighborhood,patch_targeted_search
)
from scripts.overnight_bohb_losses import regression_loss


@pytest.mark.parametrize("family",core.ARCHITECTURES)
def test_previous_five_hour_bug_impossible_in_focused_mutations(family):
    options=valid_neighborhood(family)
    assert set(options)==set(core.MUTABLE)
    assert all(isinstance(v,list) and v for v in options.values())
    assert all(0<x<1 for x in options["tau"])
    assert all(.01<=x<=.30 for x in options["tail_fraction"])
    assert all(0<=x<=2 for x in options["tail_strength"])
    assert all(x in core.LOSS_FAMILIES for x in options["loss"])
    assert options["lr"] and options["wd"]


@pytest.mark.parametrize("family",core.ARCHITECTURES)
def test_every_focused_option_remains_valid_under_real_loss(family):
    opts=valid_neighborhood(family)
    base={k:v[len(v)//2] for k,v in opts.items()}
    base.update(architecture=family,seed=17,epochs_per_step=2)
    for k,candidates in opts.items():
        for v in candidates:
            config={**base,k:v}
            assert core.validate_config(config)
            pred=torch.tensor([[9.2,-.1],[9.8,.3],[10.1,-1.1]],
                              requires_grad=True)
            truth=torch.tensor([9.1,10.,9.9])
            loss=regression_loss(pred,truth,core.loss_hp(config),"ask")
            assert torch.isfinite(loss)
            loss.backward()
            assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("family",core.ARCHITECTURES)
def test_real_ray_search_space_and_mutations_stay_categorical(monkeypatch,family):
    from ray import tune
    original=core._pbt_space
    original_m=core._pbt_mutations
    try:
        patch_targeted_search(family)
        opts=valid_neighborhood(family)
        mutations=core._pbt_mutations(tune)
        assert mutations==opts
        assert all(isinstance(v,list) for v in mutations.values())
        fake=SimpleNamespace(features="/tmp/features.npz",embeddings="/tmp/bge",
                             epochs_per_step=2)
        space=core._pbt_space(tune,family,fake)
        assert space["architecture"]==family
        assert set(mutations).issubset(space.keys())
        for _ in range(30):
            sampled={k:v.sample() if hasattr(v,"sample") else v
                     for k,v in space.items()}
            assert core.validate_config(sampled)
    finally:
        core._pbt_space=original
        core._pbt_mutations=original_m


def test_narrow_search_rejects_unregistered_architecture():
    with pytest.raises(ValueError):
        nearby("wrong")
