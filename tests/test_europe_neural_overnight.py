"""Crash-prevention and numerical-safety checks for all overnight networks."""
from pathlib import Path
import json
import numpy as np
import pytest
import torch

from scripts.europe_neural_overnight import (
    CATEGORICAL, NUMERIC, DEADLINE, FAMILIES, LOSSES,
    Prepared, DeepTabular, base_config, objective, safe_config,
    smoke, stats_of_train, read_arrays,
)


def test_5_architectures_times_10_losses_finite_forward_backward():
    smoke()


def test_prepare_unknown_categories_and_missing_numerics():
    a=np.full((4,len(NUMERIC)+len(CATEGORICAL)),np.nan,dtype=np.float32)
    a[0,0]=2020
    a[1,0]=2019
    a[0,len(NUMERIC)]=2.
    mean,std=stats_of_train(a)
    pack=Prepared(a,np.array([2000.,4000.,8000.,16000.]),
                  mean,std,[3]*len(CATEGORICAL))
    assert pack.num.shape==(4,2*len(NUMERIC))
    assert pack.cat.shape==(4,len(CATEGORICAL))
    assert pack.cat[0,0].item()==3
    assert pack.cat[1,0].item()==0
    assert torch.isfinite(pack.num).all()
    assert torch.isfinite(pack.log_y).all()


def test_crash_prone_hyperparameters_rejected():
    c=base_config("fttransformer")
    for param,bad in [("batch",2048),("dim",16),("layers",3),
                      ("dropout",.8),("lr",.1),("sigma",.001),
                      ("tau",2.),("wd",1.),("train_fraction",.05)]:
        trial={**c,param:bad}
        with pytest.raises(ValueError):
            safe_config(trial)


def test_no_test_file_allowed_on_neural_worker(tmp_path):
    folder=Path(tmp_path)
    (folder/"test.npz").write_bytes(b"sealed")
    (folder/"manifest.json").write_text(json.dumps({
        "deduplication_applied":False,"market":"ask","censored_rows":0,
        "feature_names":list(NUMERIC+CATEGORICAL),
        "test_sealed_until_final_aggregation":True,
        "counts":{"train":1602247,"validation":200039},
    }))
    with pytest.raises(ValueError,match="sealed test"):
        read_arrays(folder)


def test_overnight_deadline_is_6am_cet_november_10():
    from zoneinfo import ZoneInfo
    local=DEADLINE.astimezone(ZoneInfo("Europe/Amsterdam"))
    assert (local.year,local.month,local.day,local.hour,local.minute)==(2026,11,10,6,0)
