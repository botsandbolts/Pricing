"""AFT likelihood and research specialization tests, no private data required."""
import math
import pytest
import torch
from scripts.ask_interval_aft_likelihood import (
    lognormal_aft_nll,exact_ask_aft_nll,
)
from scripts.ray_aft_ask_focus import (
    LOSS_FAMILIES,regression_loss,loss_hp,validate_config,
)

def example_config(loss="aft"):
    return {
        "architecture":"resmlp","seed":123,
        "lr":.0007,"wd":.003,"batch":512,"dropout":.1,
        "noise":.004,"clip":1.,
        "loss":loss,"tail_strength":.4,"tail_fraction":.05,
        "tail_temperature":.1,"usd_scale":10000.,
        "beta":.3,"tau":.5,"sigma_floor":.07,
    }

def test_exact_asking_label_matches_standard_lognormal_distribution():
    raw=torch.tensor([[9.3,.0],[10.2,-.8]],dtype=torch.float64,requires_grad=True)
    observed=torch.tensor([10000.,23000.],dtype=torch.float64)
    actual=exact_ask_aft_nll(raw,observed,sigma_floor=.07)
    sigma=torch.nn.functional.softplus(raw[:,1])+.07
    expected=-torch.distributions.LogNormal(raw[:,0],sigma).log_prob(observed).mean()
    assert torch.allclose(actual,expected,atol=1e-9,rtol=1e-9)
    actual.backward()
    assert torch.isfinite(raw.grad).all()

def test_verified_censoring_types_have_finite_gradients():
    # The following are SYNTHETIC measured bounds; the current Ask dataset
    # contains no corresponding censoring metadata.
    raw=torch.tensor([[9.6,.2],[9.8,.1],[10.,-.1],[10.2,.2]],
                     dtype=torch.float64,requires_grad=True)
    lo=torch.tensor([11000.,0.,18000.,17000.],dtype=torch.float64)
    hi=torch.tensor([11000.,20000.,float("inf"),24000.],dtype=torch.float64)
    v=lognormal_aft_nll(raw,lo,hi)
    assert torch.isfinite(v)
    v.backward()
    assert torch.isfinite(raw.grad).all()

def test_invalid_and_uninformative_bounds_are_never_fabricated():
    raw=torch.zeros((1,2))
    for lo,hi in [(-1.,3000.),(3000.,1000.),(0.,float("inf")),
                  (3000.,0.),(1000.,float("nan"))]:
        with pytest.raises(ValueError):
            lognormal_aft_nll(raw,torch.tensor([lo]),torch.tensor([hi]))

@pytest.mark.parametrize("loss",("aft","hybrid_aft_dollar"))
def test_specialist_uses_exact_aft_and_evolves_scale(loss):
    assert set(LOSS_FAMILIES)=={"aft","hybrid_aft_dollar"}
    config=example_config(loss)
    validate_config(config)
    hp=loss_hp(config)
    assert hp["sigma_floor"]==.07
    raw=torch.tensor([[9.5,-1.],[9.3,.2],[10.,0.]],requires_grad=True)
    target=torch.tensor([9.6,9.4,9.9])
    v=regression_loss(raw,target,hp,"ask")
    assert torch.isfinite(v)
    v.backward()
    assert torch.isfinite(raw.grad).all()

def test_invalid_sigma_floor_fails_before_training():
    config=example_config()
    with pytest.raises(ValueError):
        validate_config({**config,"sigma_floor":.5})
