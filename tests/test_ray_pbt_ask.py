"""Fast, fully synthetic regression checks; never require historical test data."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.ray_pbt_ask import (
    ARCHITECTURES,LOSS_FAMILIES,MUTABLE,architecture_params,
    install_dynamic_hparams,loss_hp,validate_config,validate_run_settings,
)
from scripts.ask25_joint_search import ResearchAskingNetwork
from scripts.overnight_bohb_losses import regression_loss


def small_config(architecture="resmlp"):
    return {
        "architecture":architecture,"seed":17,"epochs_per_step":2,
        "lr":.0007,"wd":.003,"batch":512,"dropout":.15,
        "noise":.004,"clip":1.,"loss":"soft_cvar_dollar",
        "tail_strength":.35,"tail_fraction":.05,
        "tail_temperature":.09,"usd_scale":10000.,
        "beta":.35,"tau":.5,
    }


@pytest.mark.parametrize("architecture",ARCHITECTURES)
def test_both_architecture_islands_have_compatible_frozen_bge_inputs(architecture):
    torch.manual_seed(13)
    p=architecture_params(architecture)
    assert p["family"]==("resmlp" if architecture=="resmlp" else "dcnv2")
    m=ResearchAskingNetwork([8]*11,384,p)
    x=m(torch.randn(3,7),torch.ones(3,11,dtype=torch.int64),
        torch.randn(3,384))
    assert x.shape==(3,2) and torch.isfinite(x).all()


@pytest.mark.parametrize("name",LOSS_FAMILIES)
def test_all_mutable_loss_families_have_finite_trainable_gradients(name):
    torch.manual_seed(13)
    hp=loss_hp({**small_config(),"loss":name})
    preds=torch.randn(19,2,requires_grad=True)
    truth=torch.randn(19)+9.5
    value=regression_loss(preds,truth,hp,"ask")
    assert bool(torch.isfinite(value).item())
    value.backward()
    assert torch.isfinite(preds.grad).all()


def test_dropout_lr_and_decay_mutate_without_weight_shape_change():
    c=small_config()
    m=ResearchAskingNetwork([8]*11,384,architecture_params("resmlp"))
    optimizer=torch.optim.AdamW(m.parameters(),lr=.001)
    before={k:tuple(x.shape) for k,x in m.state_dict().items()}
    install_dynamic_hparams(m,optimizer,c)
    after={k:tuple(x.shape) for k,x in m.state_dict().items()}
    assert before==after and optimizer.param_groups[0]["lr"]==c["lr"]
    assert optimizer.param_groups[0]["weight_decay"]==c["wd"]
    assert all(mod.p==c["dropout"] for mod in m.modules()
               if isinstance(mod,torch.nn.Dropout))


def test_invalid_config_and_wild_memory_budget_rejected():
    c=small_config()
    assert validate_config(c)
    with pytest.raises(ValueError):
        validate_config({**c,"batch":4096})
    with pytest.raises(ValueError):
        validate_config({**c,"architecture":"unverified-transformer"})
    args=SimpleNamespace(architecture="resmlp",population=6,workers=2,
                         epochs_per_step=3,perturb_steps=3,steps=18)
    validate_run_settings(args)
    with pytest.raises(ValueError):
        validate_run_settings(SimpleNamespace(**{**vars(args),"population":2}))


def test_pbt_backend_uses_real_ray_checkpointed_trainable():
    from ray import tune
    from scripts.ray_pbt_ask import _ray_trainable
    cls=_ray_trainable()
    assert issubclass(cls,tune.Trainable)
    assert all(callable(getattr(cls,m,None)) for m in
               ("setup","step","save_checkpoint","load_checkpoint","reset_config"))
