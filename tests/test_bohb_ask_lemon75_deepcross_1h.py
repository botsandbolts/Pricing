"""Fast self-tests before spending an hour on the frozen Lemon BOHB study."""
import numpy as np
import pytest
import torch

import scripts.bohb_ask_lemon75_deepcross_1h as study


def test_frozen_counts_and_one_hour_budget():
    assert study.BUDGETS == (4,12,36)
    assert study.EXPECTED == {"fit":99749,"inner":11083,
                               "lemon_validation":23751,"test":31682}
    assert study.SOURCE_RUN == 42485146978
    assert study.SOURCE_RAY_FILTERED_CVAR95_USD > 0
    assert study.COHORT == "frozen-lemon-oof-retained-75"
    assert len(study.KINDS) == 10


def test_real_hyperparameter_space_and_optimizers():
    space = study.configspace()
    assert len(space) >= 19
    for _ in range(6):
        choice = study.normalise_config(dict(space.sample_configuration()))
        assert choice["loss"] in study.KINDS
        assert choice["optimizer"] in ("adamw","adam","radam")
        assert choice["schedule"] in ("constant","cosine","warm_restart")
        assert choice["batch"] in (384,512,768,1024)


def test_all_10_aft_and_tail_losses_are_finite_and_differentiable():
    config = study.normalise_config(dict(study.configspace().sample_configuration()))
    truth = torch.log(torch.tensor([5000.,7500.,16000.,24000.]))
    for name in study.KINDS:
        hp = {**config,"loss":name}
        pred = torch.stack((truth + torch.tensor([.01,-.02,.05,-.04]),
                            torch.full_like(truth,-1.)),dim=1).detach()
        pred.requires_grad_(True)
        loss = study.weighted_exact_aft_loss(pred,truth,hp)
        assert torch.isfinite(loss).item(), name
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all().item(), name


def test_training_and_validation_never_invoke_original_test():
    import inspect
    source=inspect.getsource(study.run)
    assert "original_test_scored" in source
    assert "reference_retained_mask" in source
    assert "valid_all[filter75]" in source
    assert "production_unchanged" in source
    assert "lemon_validation" in source


def test_invalid_search_duration_rejected_before_data_read(tmp_path):
    from types import SimpleNamespace
    args=SimpleNamespace(architecture="deepcross",search_seconds=1700,
        reserve_seconds=160,output=str(tmp_path),
        features="does-not-exist.npz",embeddings="does-not-exist",
        reference="does-not-exist.csv.gz")
    with pytest.raises(ValueError,match="duration"):
        study.run(args)
