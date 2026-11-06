"""Expanded, bounded FM family sweep on exact wholesale FULL vs LEMON cohorts.

Five genuine model families; rank and training cohort prespecified. Every trial
uses original VIN-disjoint auctions, and publishes validation ONLY. The original
test is accessed solely in the independent reporting job.

1. FM: ordinary second-order latent pair factors.
2. FFM: partner-field-specific embedding tables, 105 field pairs.
3. HOFM3: second- and third-order ANOVA interactions using symmetric sums.
4. Field-weighted FM: learnable explicit field-pair gates on top of FM.
5. DeepFM: same FM plus a small jointly trained nonlinear feature branch.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.bid_fm_kernel_train import (
    AuctionFactorization, NUMERIC_COUNT, FIELD_COUNT,
    SEED, _scores, y_price, partition, source,
)

FAMILIES = {
    "fm": (16, 32, 64, 128),
    "ffm": (8, 16, 32),
    "hofm3": (16, 32),
    "fieldgate": (16, 32),
    "deepfm": (16, 32),
}
COHORTS = ("full", "lemon")
TRIALS = {
    f"{family}{rank}_{cohort}": (family, rank, cohort)
    for family, ranks in FAMILIES.items()
    for rank in ranks
    for cohort in COHORTS
}
assert len(TRIALS) == 26
EPOCHS = {"fm": 28, "ffm": 22, "hofm3": 25, "fieldgate": 25, "deepfm": 25}
PATIENCE = 5
SEED = 20261106


class ScaledAuctionFM(nn.Module):
    """All alternatives contain identical underlying main and pairwise effects."""
    def __init__(self, cardinalities, family, rank, initial_log_price=9.):
        super().__init__()
        if family not in FAMILIES or rank not in FAMILIES[family]:
            raise ValueError("Unregistered family or factor rank")
        self.family = family
        self.rank = rank
        self.fm = AuctionFactorization(
            cardinalities, rank, field_aware=(family == "ffm"),
            initial_log_price=initial_log_price)
        if family == "hofm3":
            # Elementary symmetric polynomial e3 computes the full sum of
            # distinct three-field factor products, O(number_fields * rank).
            # Start tiny: the third-order interactions must earn their place.
            self.third_strength = nn.Parameter(torch.full((rank,), .05))
        if family == "fieldgate":
            # Upper-triangular field-pair gates, initialized at zero.
            self.pair_gate = nn.Parameter(torch.zeros(FIELD_COUNT, FIELD_COUNT))
        if family == "deepfm":
            width = FIELD_COUNT * rank
            self.deep = nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, 128), nn.GELU(), nn.Dropout(.16),
                nn.Linear(128, 64), nn.GELU(), nn.Dropout(.12),
                nn.Linear(64, 1))
            # Initial deep correction zero, preserving pure-FM starting behavior.
            nn.init.zeros_(self.deep[-1].weight)
            nn.init.zeros_(self.deep[-1].bias)

    def latent(self, numeric, categories):
        """Same original FM latent factors, no price-derived inputs."""
        v = self.fm
        n = numeric.unsqueeze(-1) * v.num_latent.unsqueeze(0)
        c = torch.stack([
            table(categories[:, j])
            for j, table in enumerate(v.cat_latent)], dim=1)
        return torch.cat((n, c), dim=1)

    def forward(self, numeric, categories):
        base = self.fm(numeric, categories)
        if self.family in ("fm", "ffm"):
            return base
        v = self.latent(numeric, categories)
        if self.family == "hofm3":
            s1 = v.sum(dim=1)
            s2 = (v * v).sum(dim=1)
            s3 = (v * v * v).sum(dim=1)
            e3 = (s1.pow(3) - 3. * s1 * s2 + 2. * s3) / 6.
            return base + (e3 * self.third_strength).sum(dim=1)
        if self.family == "fieldgate":
            weights = torch.triu(torch.tanh(self.pair_gate), diagonal=1)
            # Field-specific pair weights, not separate partner embeddings.
            dot = torch.matmul(v, v.transpose(1, 2))
            return base + torch.sum(dot * weights.unsqueeze(0), dim=(1, 2))
        if self.family == "deepfm":
            return base + self.deep(v.flatten(start_dim=1)).squeeze(-1)
        raise ValueError("Unknown factorization model variant")


@torch.no_grad()
def prediction(model, num, cat, rows, batch=8192):
    model.eval()
    chunks = []
    for start in range(0, len(rows), batch):
        ids = rows[start:start + batch]
        raw = model(torch.from_numpy(num[ids].astype(np.float32)),
                    torch.from_numpy(cat[ids].astype(np.int64)))
        chunks.append(raw.cpu().numpy())
    return y_price(np.concatenate(chunks))


def run(trial, features, output, max_seconds=4800):
    if trial not in TRIALS:
        raise ValueError("Unknown prespecified factorization model")
    family, rank, cohort = TRIALS[trial]
    z, provenance = source(features)
    fit, inner, val, _test = partition(z, cohort)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(2)
    numeric = z["num"].astype(np.float32)
    cats = z["cat"].astype(np.int64)
    y = z["y"].astype(np.float64)
    model = ScaledAuctionFM(
        provenance["cardinalities"], family, rank,
        initial_log_price=float(np.median(np.log(y[fit]))))
    learning_rate = .0015 if family == "ffm" else .0025
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=.005)
    training = TensorDataset(
        torch.from_numpy(numeric[fit]), torch.from_numpy(cats[fit]),
        torch.from_numpy(np.log(y[fit]).astype(np.float32)))
    batch = 1024 if family == "ffm" else 2048
    generator = torch.Generator().manual_seed(SEED)
    batches = DataLoader(
        training, shuffle=True, batch_size=batch, num_workers=0,
        generator=generator)
    best = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    start = time.monotonic()
    completed_epochs = 0
    for epoch in range(1, EPOCHS[family] + 1):
        model.train()
        total_loss = 0.
        count = 0
        for numerical, categorical, log_price in batches:
            optimizer.zero_grad(set_to_none=True)
            estimate = model(numerical, categorical)
            loss = nn.functional.smooth_l1_loss(
                estimate, log_price, beta=.25)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite FM/FFM neural loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
            optimizer.step()
            total_loss += float(loss.detach()) * len(log_price)
            count += len(log_price)
            if time.monotonic() - start > max_seconds:
                print("HARD_TIME_BUDGET", trial, "epoch", epoch, flush=True)
                break
        completed_epochs += 1
        inner_pred = prediction(model, numeric, cats, inner)
        score = _scores(y[inner], inner_pred)["mae_usd"]
        print(json.dumps({
            "trial":trial, "epoch":epoch,
            "train_loss":total_loss / max(count, 1),
            "inner_mae_usd":score,
            "elapsed_seconds":round(time.monotonic() - start, 1)}),
            flush=True)
        if score < best - .5:
            best = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE or time.monotonic() - start > max_seconds:
            break
    if best_state is None:
        raise ValueError("No valid trained factorization checkpoint")
    model.load_state_dict(best_state)
    val_pred = prediction(model, numeric, cats, val)
    scores = _scores(y[val], val_pred)
    pd.DataFrame({
        "row_id":z["ids"][val].astype(str),
        "split":"validation",
        "observed_price":y[val],
        "predicted_price":val_pred,
    }).to_csv(output / "validation.csv.gz", index=False)
    torch.save({
        "state_dict":best_state,
        "trial":trial, "family":family,
        "rank":rank, "cohort":cohort,
        "cards":provenance["cardinalities"],
        "initial_log_price":float(model.fm.base.detach()),
        "seed":SEED, "schema":"same 5 numeric + 10 categorical auction fields",
    }, output / "weights.pt")
    report = {
        "status":"completed",
        "trial":trial, "market":"bid",
        "family":family, "rank":rank, "cohort":cohort,
        "sample_size_training":len(fit) + len(inner),
        "fitting_rows":len(fit), "stopper_rows":len(inner),
        "original_train_total":384198,
        "original_train_lemon":288148,
        "validation_rows":len(val), "untouched_test_rows":83603,
        "test_evaluated":False, "no_hierarchy":True,
        "no_price_derived_inputs":True,
        "no_description":True,
        "original_holdout_row_sha256":provenance["holdout_row_sha256"],
        "cohort_sha256":provenance["cohort_sha256"],
        "cardinalities":provenance["cardinalities"],
        "validation":scores,
        "stopping_mae_usd":best,
        "best_epoch":best_epoch,
        "completed_epochs":completed_epochs,
        "training_seconds":time.monotonic() - start,
        "architecture":"regularized 2nd-order FM + "
          + {
            "fm":"no higher-order correction",
            "ffm":"partner-field-specific factors",
            "hofm3":"ANOVA rank-3 latent correction",
            "fieldgate":"learned partner-field-pair gates",
            "deepfm":"small joint nonlinear residual branch",
          }[family],
        "loss":"Huber on log observed wholesale dollar sale price",
        "selection":"Original validation dollar MAE; original test not scored",
        "deployed":False,
    }
    (output / "result.json").write_text(json.dumps(
        report, indent=2, allow_nan=False))
    print("SCALED_FM_VALIDATION", trial, scores["mae_usd"], flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trial", choices=tuple(TRIALS), required=True)
    p.add_argument("--features", default="training-cache/neural-bid/features.npz")
    p.add_argument("--output", required=True)
    p.add_argument("--max-seconds", type=float, default=4800)
    a = p.parse_args()
    run(a.trial, a.features, a.output, max_seconds=a.max_seconds)
