"""Synthetic integrity checks for Ray PBT Lemon 75% Ask experiment."""
import numpy as np
import pytest
from scripts.ray_pbt_ask_lemon_1h import (
    TOTAL_TRAIN,RETAINED_TRAIN,ORIGINAL_VALIDATION,ORIGINAL_TEST,
    TRIMMED_TRAIN,lemon_indices
)
from scripts.ray_pbt_ask_10m import valid_neighborhood
from scripts.ray_pbt_ask import ARCHITECTURES,LOSS_FAMILIES


def synthetic():
    phase=np.array(["train"]*TOTAL_TRAIN+
                   ["validation"]*ORIGINAL_VALIDATION+
                   ["test"]*ORIGINAL_TEST)
    frozen=np.ones(len(phase),dtype=bool)
    frozen[:TRIMMED_TRAIN]=False
    return phase,frozen


def test_exact_frozen_lemon_and_holdout_isolation():
    phase,mask=synthetic()
    fit,inside=lemon_indices(phase,mask)
    assert len(fit)+len(inside)==RETAINED_TRAIN
    assert len(fit)==99749
    assert len(inside)==11083
    assert np.all(mask[fit]) and np.all(mask[inside])
    assert set(fit).isdisjoint(inside)
    assert max(inside)<TOTAL_TRAIN
    assert mask[phase=="validation"].all() and mask[phase=="test"].all()


def test_rejects_any_holdout_filter_or_modified_train_count():
    phase,mask=synthetic()
    bad=mask.copy()
    bad[TOTAL_TRAIN]=False
    with pytest.raises(ValueError,match="must not be filtered"):
        lemon_indices(phase,bad)
    bad=mask.copy()
    bad[1]=True
    with pytest.raises(ValueError,match="audited retained 75"):
        lemon_indices(phase,bad)


@pytest.mark.parametrize("arch",ARCHITECTURES)
def test_one_hour_study_has_safe_mutations(arch):
    opts=valid_neighborhood(arch)
    assert all(0<x<1 for x in opts["tau"])
    assert all(0.01<=x<=0.30 for x in opts["tail_fraction"])
    assert set(opts["loss"]).issubset(set(LOSS_FAMILIES))
    assert all(0.00003<=x<=0.004 for x in opts["lr"])
    assert all(0<=x<=0.3 for x in opts["wd"])
