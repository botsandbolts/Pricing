"""Guard against data leakage, accidental model/target switches and test-based tuning."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.deepcross_aft_tune import CONFIGS, FEATURES, build, config, sha_ids, verify_features
from scripts.deepcross_aft_report import choose_winner
from scripts.neural_ask_train import DeepCross, objective


def test_grid_and_reproducible_baseline():
    assert len(CONFIGS) == 18
    assert FEATURES == ("svd", "minilm")
    assert len(CONFIGS) * len(FEATURES) == 36
    assert isinstance(build("baseline", [20, 30], 7, 24), DeepCross)
    torch.manual_seed(42)
    model = build("baseline", [20, 30], 7, 24).eval()
    x = (torch.randn(4, 7), torch.zeros(4, 2, dtype=torch.long), torch.randn(4, 24))
    with torch.no_grad():
        assert model(*x).shape == (4, 2)


@pytest.mark.parametrize("variant", list(CONFIGS))
def test_all_architectural_and_optimizer_variants(variant):
    specs = config(variant)
    assert specs["lr"] > 0 and specs["weight_decay"] >= 0
    assert 1 <= specs["cross_layers"] <= 6
    assert 1 <= specs["experts"] <= 4
    assert specs["batch_size"] >= 256
    model = build(variant, [12, 15], 5, 4).eval()
    with torch.no_grad():
        prediction = model(torch.rand(3,5), torch.ones(3,2,dtype=torch.long),torch.rand(3,4))
    assert prediction.shape == (3,2)
    assert torch.isfinite(prediction).all()
    err=objective(prediction,torch.log(torch.tensor([8000.,15000.,18000.])),"aft")
    assert torch.isfinite(err)


def test_validation_only_selects_champion():
    rows=pd.DataFrame([
        {"variant":"fast","feature_variant":"svd","partition":"validation","mae_usd":100},
        {"variant":"slow","feature_variant":"minilm","partition":"validation","mae_usd":120},
        {"variant":"fast","feature_variant":"svd","partition":"test","mae_usd":6000},
        {"variant":"slow","feature_variant":"minilm","partition":"test","mae_usd":10},
    ])
    winner=choose_winner(rows)
    assert winner.variant=="fast"
    rows.loc[rows.partition=="test","mae_usd"]=[999999.,1.]
    assert choose_winner(rows).variant=="fast"


def test_original_holdout_integrity_and_price_source(tmp_path):
    values={"full_train":147777,"validation":31668,"test":31682,
            "no_hierarchy":True,"price_derived_inputs":False,
            "holdout_row_sha256":{"validation":sha_ids(["v"]),"test":sha_ids(["t"])}}
    path=tmp_path/"features.npz"
    path.with_suffix(".json").write_text(json.dumps(values))
    assert verify_features(path,"svd")["no_hierarchy"]
    with pytest.raises(ValueError,match="MiniLM"):
        verify_features(path,"minilm")
    values.update(variant="minilm-2-oob-price",minilm_internal_stopper_labels_used=False,
                  minilm_validation_test_labels_used=False)
    path.with_suffix(".json").write_text(json.dumps(values))
    assert verify_features(path,"minilm")["variant"]=="minilm-2-oob-price"
    values["minilm_validation_test_labels_used"]=True
    path.with_suffix(".json").write_text(json.dumps(values))
    with pytest.raises(ValueError,match="MiniLM"):
        verify_features(path,"minilm")


def test_unknown_variant_refused():
    with pytest.raises(ValueError,match="Unknown"):
        config("quantile-loss-instead-of-aft")
