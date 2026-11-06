"""Regularized CPU neural asking-price sweep: 3 architectures x 3 honest losses x 2 train cohorts.

All outcomes are exact observed advertised asking prices. 'aft' is log-normal
exact-event likelihood (not made-up censoring). Internal temporal early stopping
is restricted to TRAIN; canonical validation chooses the winner, TEST audits it.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

ARCHS = ("resmlp", "deepcross", "fttransformer")
LOSSES = ("huber", "quantile50", "aft")
COHORTS = ("full", "lemon")
SEED = 20261106


def seed_everything(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)


class CarEmbedding(nn.Module):
    def __init__(self, sizes, n_num, n_text, dropout):
        super().__init__()
        self.tables = nn.ModuleList()
        dims = []
        for size in sizes:
            dim = min(24, max(4, int(round(math.sqrt(int(size)) * 1.5))))
            self.tables.append(nn.Embedding(int(size), dim, padding_idx=0))
            dims.append(dim)
        self.dropout = nn.Dropout(dropout)
        self.width = n_num + n_text + sum(dims)

    def forward(self, num, cat, txt):
        embedded = [emb(cat[:, j]) for j, emb in enumerate(self.tables)]
        return torch.cat([num, txt, *[self.dropout(e) for e in embedded]], 1)


class ResidualBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.path = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                  nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(width, width), nn.Dropout(dropout))

    def forward(self, x):
        return x + 0.5 * self.path(x)


class ResMLP(nn.Module):
    def __init__(self, sizes, n_num, n_text):
        super().__init__()
        self.encoder = CarEmbedding(sizes, n_num, n_text, .16)
        self.body = nn.Sequential(nn.Linear(self.encoder.width, 256),
                                  nn.LayerNorm(256), nn.GELU(),
                                  ResidualBlock(256, .22),
                                  ResidualBlock(256, .22),
                                  ResidualBlock(256, .22),
                                  nn.LayerNorm(256))
        self.out = nn.Linear(256, 2)

    def forward(self, n, c, t):
        return self.out(self.body(self.encoder(n, c, t)))


class DeepCross(nn.Module):
    """Low-rank explicit crosses plus independent residual nonlinear deep branch."""
    def __init__(self, sizes, n_num, n_text):
        super().__init__()
        self.encoder = CarEmbedding(sizes, n_num, n_text, .18)
        d = self.encoder.width
        self.u = nn.ModuleList(nn.Linear(d, 32, bias=False) for _ in range(3))
        self.v = nn.ModuleList(nn.Linear(32, d) for _ in range(3))
        self.deep = nn.Sequential(nn.Linear(d, 256), nn.LayerNorm(256),
                                  nn.GELU(), nn.Dropout(.25),
                                  ResidualBlock(256, .25),
                                  ResidualBlock(256, .25))
        self.out = nn.Sequential(nn.LayerNorm(d + 256),
                                 nn.Linear(d + 256, 128), nn.GELU(),
                                 nn.Dropout(.22), nn.Linear(128, 2))

    def forward(self, n, c, t):
        x0 = self.encoder(n, c, t)
        x = x0
        for u, v in zip(self.u, self.v):
            x = x + x0 * torch.tanh(v(u(x)))
        return self.out(torch.cat([x, self.deep(x0)], 1))


class FTTransformer(nn.Module):
    """Contextual feature tokens with learned numeric tokenizers and CLS attention."""
    def __init__(self, sizes, n_num, n_text):
        super().__init__()
        d = 48
        self.embeddings = nn.ModuleList([nn.Embedding(int(s), d, padding_idx=0) for s in sizes])
        self.num_weight = nn.Parameter(torch.empty(n_num, d))
        self.num_bias = nn.Parameter(torch.zeros(n_num, d))
        nn.init.normal_(self.num_weight, std=.05)
        self.text = nn.Sequential(nn.Linear(n_text, d), nn.LayerNorm(d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=4, dim_feedforward=192,
                                           dropout=.18, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Sequential(nn.Linear(d, 96), nn.GELU(), nn.Dropout(.2),
                                 nn.Linear(96, 2))

    def forward(self, n, c, t):
        tokens = [e(c[:, j]).unsqueeze(1) for j, e in enumerate(self.embeddings)]
        numeric = n.unsqueeze(-1) * self.num_weight + self.num_bias
        tokens += [numeric, self.text(t).unsqueeze(1)]
        body = torch.cat([self.cls.expand(n.shape[0], -1, -1), *tokens], dim=1)
        return self.out(self.norm(self.encoder(body)[:, 0, :]))


def build_model(arch, card, n_num, n_text):
    constructors = {"resmlp": ResMLP, "deepcross": DeepCross,
                    "fttransformer": FTTransformer}
    if arch not in constructors:
        raise ValueError("Unknown architecture")
    return constructors[arch](card, n_num, n_text)


def objective(pred, target, loss_name):
    mu = pred[:, 0]
    err = target - mu
    if loss_name == "huber":
        return nn.functional.smooth_l1_loss(mu, target, beta=.25)
    if loss_name == "quantile50":
        return torch.maximum(.5 * err, -.5 * err).mean()
    if loss_name == "aft":
        sigma = (nn.functional.softplus(pred[:, 1]) + .06).clamp(max=3.)
        # Exact observations: normal log-likelihood on log(positive asking price).
        # Interval likelihoods are NOT used because there are NO observed bounds.
        return (.5 * (err / sigma).square() + sigma.log()).mean()
    raise ValueError("Invalid loss")


def metrics(y, predicted):
    y = np.asarray(y, float)
    p = np.asarray(predicted, float)
    if len(y) == 0 or len(y) != len(p) or not np.isfinite(p).all():
        raise ValueError("Invalid heldout predictions")
    difference = p - y
    return {"rows": int(len(y)),
            "mae_usd": float(np.abs(difference).mean()),
            "rmse_usd": float(np.sqrt(np.mean(difference ** 2))),
            "mean_bias_usd": float(difference.mean()),
            "median_absolute_percentage_error": float(np.median(np.abs(difference) / y)),
            "rmse_log": float(np.sqrt(np.mean((np.log(y) - np.log(p)) ** 2))),
            "spearman": float(pd.Series(y).corr(pd.Series(p), method="spearman"))}


@torch.no_grad()
def predict(model, dataset, indexes, batch_size):
    loader = DataLoader(Subset(dataset, indexes.tolist()), batch_size=batch_size,
                        num_workers=0, shuffle=False)
    model.eval()
    parts = []
    for n, c, t, _ in loader:
        raw = model(n, c, t)[:, 0].detach().cpu().numpy()
        parts.append(np.exp(np.clip(raw, np.log(100), np.log(1000000))))
    return np.concatenate(parts).astype(np.float64)


def run(a):
    if a.arch not in ARCHS or a.loss not in LOSSES or a.cohort not in COHORTS:
        raise ValueError("Unknown experiment arm")
    seed_everything()
    with np.load(a.data, allow_pickle=False) as data:
        num = data["num"].astype(np.float32)
        cat = data["cat"].astype(np.int64)
        txt = data["txt"].astype(np.float32)
        y = data["y"].astype(np.float64)
        phase = data["phase"].astype(str)
        ids = data["ids"].astype(str)
        lemon = data["lemon"].astype(bool)
    provenance = json.loads(Path(a.data).with_suffix(".json").read_text())
    train = np.flatnonzero((phase == "train") & (lemon if a.cohort == "lemon" else True))
    val = np.flatnonzero(phase == "validation")
    test = np.flatnonzero(phase == "test")
    if len(train) != provenance["lemon_train" if a.cohort == "lemon" else "full_train"]:
        raise ValueError("Unexpected training cohort size")
    if len(val) != provenance["validation"] or len(test) != provenance["test"]:
        raise ValueError("Unexpected original heldout cohort")
    # Original train is chronological. The last 10% of the selected train
    # serves ONLY for early stopping, not main validation or test.
    stopper = max(512, int(len(train) * .1))
    fit, internal = train[:-stopper], train[-stopper:]
    if np.intersect1d(fit, internal).size or np.intersect1d(train, test).size:
        raise ValueError("Split leakage")
    dataset = TensorDataset(torch.from_numpy(num), torch.from_numpy(cat),
                            torch.from_numpy(txt), torch.from_numpy(np.log(y).astype(np.float32)))
    card = provenance["cardinalities"]
    model = build_model(a.arch, card, num.shape[1], txt.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=.00065 if a.arch == "fttransformer" else .001,
                                   weight_decay=.025)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs, eta_min=.00006)
    batches = DataLoader(Subset(dataset, fit.tolist()), shuffle=True,
                         batch_size=a.batch_size, num_workers=0, drop_last=False)
    best_score = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, a.epochs + 1):
        model.train()
        total, nrows = 0., 0
        for numbers, categories, text, truth in batches:
            optimizer.zero_grad(set_to_none=True)
            if a.arch != "fttransformer":
                # Bounded label-free input jitter is a mild regularizer.
                numbers = numbers + torch.randn_like(numbers) * .008
                text = text + torch.randn_like(text) * .008
            raw = model(numbers, categories, text)
            loss = objective(raw, truth, a.loss)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite neural loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach()) * len(truth)
            nrows += len(truth)
        schedule.step()
        pstop = predict(model, dataset, internal, a.batch_size)
        score = metrics(y[internal], pstop)["mae_usd"]
        print(json.dumps({"arch": a.arch, "loss": a.loss, "cohort": a.cohort,
                          "epoch": epoch, "train_loss": total / nrows,
                          "inner_stopper_mae_usd": round(score, 3)}), flush=True)
        if score < best_score - .1:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= a.patience:
                break
    if best_state is None:
        raise ValueError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    pv = predict(model, dataset, val, a.batch_size)
    pt = predict(model, dataset, test, a.batch_size)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame({"row_id": np.concatenate([ids[val], ids[test]]),
                         "split": ["validation"] * len(val) + ["test"] * len(test),
                         "observed_price": np.concatenate([y[val], y[test]]),
                         "predicted_price": np.concatenate([pv, pt])})
    rows.to_csv(out / "predictions.csv.gz", index=False)
    torch.save({"state_dict": model.state_dict(), "architecture": a.arch,
                "loss": a.loss, "cohort": a.cohort,
                "cardinalities": card, "numeric_dimensions": num.shape[1],
                "text_dimensions": txt.shape[1]}, out / "weights.pt")
    result = {"status": "completed", "market": "ask", "architecture": a.arch,
              "loss": a.loss, "train_cohort": a.cohort,
              "objective": "observed asking log-dollar price",
              "aft_censoring": "zero genuinely censored rows; exact log-normal likelihood" if a.loss == "aft" else None,
              "train_available": int(len(train)), "train_fit": int(len(fit)),
              "train_early_stopper": int(len(internal)),
              "best_epoch": best_epoch, "inner_stopper_mae_usd": best_score,
              "weight_decay": .025, "gradient_norm_clip": 1.,
              "original_validation": metrics(y[val], pv),
              "original_test": metrics(y[test], pt),
              "holdout_row_sha256": provenance["holdout_row_sha256"],
              "canonical_cohort_sha256": provenance["cohort_sha256"],
              "selection": "Independent train-only internal stopper; cross-arm selection on original validation; test is never used for training/early stopping",
              "hierarchical_correction": False}
    (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", choices=ARCHS, required=True)
    p.add_argument("--loss", choices=LOSSES, required=True)
    p.add_argument("--cohort", choices=COHORTS, required=True)
    p.add_argument("--data", default="training-cache/neural-ask/features.npz")
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=768)
    run(p.parse_args())
