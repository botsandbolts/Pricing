"""Synthetic safety regression tests for independent half-hour BOHB Ask AFT."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.bohb_aft_ask_30m import (
    BUDGETS,KINDS,SEED,bohb_generator,choose_next,
    configspace,normalise_config,weighted_exact_aft_loss,
)


def test_real_hpbandster_kde_generator_and_three_real_fidelities():
    from hpbandster.optimizers.config_generators.bohb import BOHB
    assert BUDGETS==(3,9,27)
    gen=bohb_generator([])
    assert isinstance(gen,BOHB)
    cfg,_=gen.get_config(budget=3.)
    x=normalise_config(cfg)
    assert x["loss"] in KINDS and x["batch"] in (512,768,1024)
    assert .01<=x["sigma_floor"]<=.2
    assert len(configspace().get_hyperparameters())>=12


@pytest.mark.parametrize("loss",KINDS)
def test_weighted_exact_aft_and_robust_losses_have_finite_gradients(loss):
    gen=bohb_generator([])
    cfg,_=gen.get_config(budget=3.)
    hp=normalise_config(cfg)
    hp.update(loss=loss,aft_weight=.65,tail_weight=.25,sigma_floor=.06)
    pred=torch.tensor([[9.1,-.3],[9.4,-.7],[10.5,-1.2],[10.,.2]],
                      requires_grad=True)
    log_y=torch.log(torch.tensor([10000.,12000.,23000.,17000.]))
    v=weighted_exact_aft_loss(pred,log_y,hp)
    assert torch.isfinite(v)
    v.backward()
    assert torch.isfinite(pred.grad).all()


def test_pure_aft_is_exact_log_normal_likelihood_up_to_target_constant():
    gen=bohb_generator([])
    c=normalise_config(gen.get_config(budget=3.)[0])
    c.update(loss="aft",sigma_floor=.06,tail_weight=0.,aft_weight=1.)
    pred=torch.tensor([[9.2,-.4],[9.6,.2]],dtype=torch.float64)
    observed=torch.tensor([10000.,15000.],dtype=torch.float64)
    actual=weighted_exact_aft_loss(pred,observed.log(),c)
    sigma=torch.nn.functional.softplus(pred[:,1])+.06
    expected=-torch.distributions.LogNormal(pred[:,0],sigma).log_prob(observed).mean()
    expected-=observed.log().mean()+math.log(2*math.pi)/2
    assert torch.allclose(actual,expected,rtol=1e-9,atol=1e-9)


def test_halving_promotes_better_completed_train_only_candidate():
    gen=bohb_generator([])
    c=normalise_config(gen.get_config(budget=3.)[0])
    d={**c,"lr":c["lr"]*1.001}
    e={**c,"wd":c["wd"]*1.001}
    rows=[]
    for i,(params,score) in enumerate(((c,19000.),(d,20000.),(e,21000.))):
        rows.append({"trial":i,"epochs":3,"params":params,
                     "status":"complete","inner_metrics":{
                         "CVaR95_absolute_USD":score}})
    new=bohb_generator(rows)
    hp,budget,parent=choose_next(rows,new)
    assert budget==9 and parent==0 and hp==c


def test_no_fabricated_censoring_bounds_and_invalid_config_rejected():
    gen=bohb_generator([])
    c=normalise_config(gen.get_config(budget=3.)[0])
    with pytest.raises(ValueError):normalise_config({**c,"sigma_floor":2.})
    with pytest.raises(ValueError):normalise_config({**c,"loss":"imaginary-censored-sale"})
    with pytest.raises(ValueError):normalise_config({**c,"batch":8192})
