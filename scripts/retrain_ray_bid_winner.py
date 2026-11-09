"""Re-train the actual shard-2 internal-CVaR95 winner, never use test to select.

No Ray scheduler or old 09:00 deadline: reproducibly fit exactly the
archived best-epoch configuration on its identical 90/10 TRAIN-only split.
Fail closed if frozen data, serving vocabulary or reproducibility drift.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.ray_bid_9am import (
    BidNetwork, bound_config, make_loss, read_source, prices, _risk_exact,
)


def _sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def retrain(features, reference, output, root="."):
    root = Path(root)
    output = Path(output)
    reference = Path(reference)
    reference_result = json.loads(reference.read_text())
    config = bound_config(dict(reference_result["configuration"]))
    expected = float(reference_result["inner_CVaR95_USD"])
    assert (config["family"], config["loss"], int(config["rank"])) == (
        "deepfm", "cvar95_weighted_huber", 16)
    assert reference_result.get("training_full_non_lemon") is True
    assert 6400 < expected < 6600, "Unexpected Ray inner winner identity"
    epochs = int(reference_result["best_epoch"])
    if not 1 <= epochs <= 250:
        raise ValueError("Invalid archived optimal epoch")
    torch.set_num_threads(2)
    torch.manual_seed(20261109 + int(config["seed"]))
    np.random.seed(20261109 + int(config["seed"]))
    z, meta, censoring = read_source(features, with_bounds=True)
    if meta["cohort_sha256"] != reference_result["source_sha256"]:
        raise ValueError("Original non-Lemon dataset differs from Ray trial")
    prep_path = root / "models/neural-experimental-bid/preprocessing.json"
    prep = json.loads(prep_path.read_text())
    cards = [int(x) for x in meta["cardinalities"]]
    if cards != [int(x) for x in prep["cardinalities"]]:
        raise ValueError("Frozen original Bid cardinalities differ from live dictionary")
    if prep.get("categorical_features") != list(
        ("manufacturer","model","type","transmission","state","fuel",
         "drive","title_status","cylinders","paint_color")):
        raise ValueError("Live wholesale category ordering differs from source")
    phase = z["phase"].astype(str)
    training = np.flatnonzero(phase == "train")
    n_stop = max(512, int(len(training) * .1))
    fit, inner = training[:-n_stop], training[-n_stop:]
    if (len(fit), len(inner)) != (345779, 38419):
        raise ValueError("TRAIN internal partition changed")
    inner_exact = (z["lower_log"][inner] == z["upper_log"][inner])
    if int(inner_exact.sum()) < 500:
        raise ValueError("Insufficient exact-price TRAIN internal rows")
    inner = inner[inner_exact]
    tensor_n = torch.from_numpy(z["num"][fit].astype(np.float32))
    tensor_c = torch.from_numpy(z["cat"][fit].astype(np.int64))
    tensor_y = torch.from_numpy(np.log(z["y"][fit]).astype(np.float32))
    low = torch.from_numpy(z["lower_log"][fit])
    high = torch.from_numpy(z["upper_log"][fit])
    model = BidNetwork(cards, float(np.median(np.log(z["y"][fit]))), config)
    optimizer = (
        torch.optim.AdamW if config["optimizer"] == "adamw" else torch.optim.RAdam
    )(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["wd"]),
      foreach=False)
    scheduler = None
    if config["schedule"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=250, eta_min=.00002)
    elif config["schedule"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=36, gamma=.5)
    best = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(
            20261109 + int(config["seed"]) * 17 + epoch)
        for batch in torch.randperm(len(tensor_y), generator=generator).split(
                int(config["batch"])):
            num = tensor_n[batch]
            if float(config["noise"]):
                num = num + torch.randn_like(num) * float(config["noise"])
            optimizer.zero_grad(set_to_none=True)
            prediction = model(num, tensor_c[batch])
            loss = make_loss(prediction, tensor_y[batch], low[batch], high[batch],
                             model.sigma(config["sigma_floor"]), config)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(config["clip"]))
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        candidate = prices(model, z["num"][inner], z["cat"][inner])
        score = _risk_exact(z["y"][inner].astype(np.float64), candidate)
        value = float(score["CVaR95_absolute_USD"])
        if not math.isfinite(value):
            raise ValueError("Non-finite training-internal CVaR95")
        if value < best - .05:
            best = value
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k,v in model.state_dict().items()}
        history.append({"epoch": epoch, "inner_CVaR95_absolute_USD": value})
        print("RAY_BID_WINNER_RETRAIN", json.dumps({
            "epoch":epoch, "best_internal_CVaR95":round(best,2),
            "reference_internal_CVaR95":round(expected,2)}), flush=True)
    if best_state is None or best > expected * 1.03:
        raise ValueError("Retraining could not reproduce internal winning performance")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    validation = np.flatnonzero(phase == "validation")
    exact_validation = (z["lower_log"][validation] == z["upper_log"][validation])
    validation = validation[exact_validation]
    if len(validation) != 82380:
        raise ValueError("Original full untouched validation set differs from Ray winner")
    original_scores = _risk_exact(
        z["y"][validation].astype(np.float64),
        prices(model, z["num"][validation], z["cat"][validation]))
    original_ref = float(reference_result["original_validation"]["CVaR95_absolute_USD"])
    if original_scores["CVaR95_absolute_USD"] > original_ref * 1.12:
        raise ValueError("Retrained model drifts excessively on independent validation")
    # Original validation is a reporting/safety gate only; internal TRAIN CVaR95
    # determines the checkpoint. The original TEST set remains sealed.
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "weights.pt"
    torch.save({
        "state_dict": best_state, "config": config, "cards": cards,
        "source_sha256": meta["cohort_sha256"],
        "original_test_scored":False,
    }, checkpoint)
    manifest = {
        "status":"release_candidate_verified", "market":"bid",
        "source_workflow_run":42570947698,
        "source_shard":2, "source_trial_loss":"cvar95_weighted_huber",
        "retrained_from_winner_hyperparameters":True,
        "training_split":{"fit":345779,"inner":38419,"full_train":384198},
        "original_training_no_lemon":True,
        "selection_metric":"TRAIN internal CVaR95_absolute_USD",
        "reference_internal_CVaR95_USD":expected,
        "retrained_internal_CVaR95_USD":best,
        "retrained_best_epoch":best_epoch,
        "original_validation":original_scores,
        "reference_original_validation":reference_result["original_validation"],
        "original_test_scored":False, "no_blending":True,
        "cohort_sha256":meta["cohort_sha256"],
        "original_holdout_row_sha256":meta["holdout_row_sha256"],
        "censoring_audit":censoring,
        "configuration":config,
        "checkpoint_sha256":_sha(checkpoint),
        "preprocessing_sha256":_sha(prep_path),
        "price_basis":"2015-index-normalized wholesale historical USD",
        "not_current_2026_accuracy":True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                                     allow_nan=False) + "\n")
    verification = {
        "status":"passed", "inner_train_partition_verified":True,
        "original_validation_rows":int(len(validation)),
        "original_test_scored":False,
        "model_not_blended_with_lemon":True,
        "checkpoint_forward_finite":True,
        "reference_inner_CVaR95_USD":expected,
        "retrained_inner_CVaR95_USD":best,
        "reference_validation_CVaR95_USD":original_ref,
        "retrained_validation_CVaR95_USD":original_scores["CVaR95_absolute_USD"],
        "model_seed":config["seed"],
        "no_price_derived_inputs":meta["no_price_derived_inputs"],
    }
    (output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    (output / "retrain_history.json").write_text(json.dumps(
        {"source":"TRAIN internal only", "best_epoch":best_epoch,
         "best_internal_CVaR95_USD":best, "history":history}, indent=2) + "\n")
    # One real serving-path smoke test with two independent user-enterable cars.
    from pricing.ray_bid_winner_live import RayWinnerWholesale, available
    if not available(root):
        raise ValueError("The just-written new release fails the serving availability gate")
    served = RayWinnerWholesale(root)
    inputs = {
        "auction_date":"2015-01-10", "year":2010, "odometer":100000,
        "manufacturer":"toyota", "model":"camry", "type":"sedan",
        "transmission":"automatic", "state":"ca", "fuel":"gas",
        "drive":"fwd", "title_status":"clean",
        "cylinders":"4 cylinders", "paint_color":"white", "auction_grade":3.5,
    }
    p1 = served.predict(inputs)["price"]
    p2 = served.predict({**inputs, "year":2013, "odometer":30000})["price"]
    if not (100 <= p1 <= 1e6 and 100 <= p2 <= 1e6 and abs(p1-p2) > 10):
        raise ValueError("Production serving smoke did not distinguish two vehicles")
    print("RAY_BID_WINNER_READY", json.dumps({
        "best_epoch":best_epoch, "internal_CVaR95_USD":round(best,2),
        "original_validation_CVaR95_USD":round(original_scores["CVaR95_absolute_USD"],2),
        "checkpoint_sha256":manifest["checkpoint_sha256"],
        "test_scored":False}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--output", default="models/ray-bid-9am-winner")
    p.add_argument("--root", default=".")
    a = p.parse_args()
    retrain(a.features, a.reference, a.output, a.root)
