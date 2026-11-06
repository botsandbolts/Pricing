"""18 reproducible DeepCross/AFT architecture and optimizer variants, two feature sets.

Historical Craigslist ask data: authentic 211127-row canonical cohort, immutable
chronological holdouts. Internal stopper and all normalization fit exclusively on
original training. The only loss is exact log-normal AFT; zero synthetic censoring.

All variants use the SAME training subset/early stopper. The baseline uses the
unchanged original DeepCross implementation; other architectures vary narrowly.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from scripts.neural_ask_train import (
    DeepCross, CarEmbedding, ResidualBlock, metrics, objective, predict,
    seed_everything,
)

# Name -> changes relative to prior successful 3-cross, rank32, width256 model.
# Do not grow this dictionary after inspecting original test results.
CONFIGS = {
    "baseline": {},
    "depth1": {"cross_layers": 1},
    "depth5": {"cross_layers": 5},
    "rank16": {"rank": 16},
    "rank64": {"rank": 64},
    "moe2": {"experts": 2},
    "moe4": {"experts": 4},
    "wide384": {"width": 384},
    "compact128": {"width": 128},
    "blocks3": {"blocks": 3},
    "dropout12": {"dropout": .12},
    "dropout35": {"dropout": .35},
    "lr06": {"lr": .0006},
    "lr15": {"lr": .0015},
    "wd08": {"weight_decay": .08},
    "batch1024": {"batch_size": 1024},
    "crosshalf": {"cross_scale": .5},
    "silu": {"activation": "silu"},
}
FEATURES = ("svd", "minilm")


def config(name):
    if name not in CONFIGS:
        raise ValueError(f"Unknown predeclared DeepCross tuning arm {name}")
    result = dict(cross_layers=3, rank=32, experts=1, width=256, blocks=2,
                  dropout=.25, embed_dropout=.18, output_dropout=.22,
                  lr=.001, weight_decay=.025, batch_size=768, cross_scale=1.,
                  activation="gelu", epochs=35, patience=7, sigma_floor=.06)
    result.update(CONFIGS[name])
    return result


class TunedDeepCross(nn.Module):
    """DCNv2-inspired low-rank cross layers; optional learned MoE per layer."""
    def __init__(self, sizes, n_num, n_text, settings):
        super().__init__()
        c = settings
        self.encoder = CarEmbedding(sizes, n_num, n_text, c["embed_dropout"])
        d = self.encoder.width
        self.layers = nn.ModuleList()
        for _ in range(c["cross_layers"]):
            experts = nn.ModuleList([
                nn.Sequential(nn.Linear(d, c["rank"], bias=False),
                              nn.Linear(c["rank"], d))
                for _ in range(c["experts"])
            ])
            gate = nn.Linear(d, c["experts"]) if c["experts"] > 1 else nn.Identity()
            self.layers.append(nn.ModuleDict({"experts": experts, "gate": gate}))
        self.cross_scale = c["cross_scale"]
        activation = nn.SiLU if c["activation"] == "silu" else nn.GELU
        width = c["width"]
        deep = [nn.Linear(d, width), nn.LayerNorm(width), activation(), nn.Dropout(c["dropout"])]
        deep.extend(ResidualBlock(width, c["dropout"]) for _ in range(c["blocks"]))
        self.deep = nn.Sequential(*deep)
        self.out = nn.Sequential(nn.LayerNorm(d + width), nn.Linear(d + width, 128),
                                 activation(), nn.Dropout(c["output_dropout"]),
                                 nn.Linear(128, 2))

    def forward(self, n, c, t):
        x0 = self.encoder(n, c, t)
        x = x0
        for layer in self.layers:
            experts = layer["experts"]
            if len(experts) == 1:
                correction = torch.tanh(experts[0](x))
            else:
                weight = torch.softmax(layer["gate"](x), dim=-1)
                correction = sum(weight[:, j:j+1] * torch.tanh(expert(x))
                                 for j, expert in enumerate(experts))
            x = x + self.cross_scale * x0 * correction
        return self.out(torch.cat((x, self.deep(x0)), dim=1))


def build(name, cards, n_num, n_text):
    # Baseline is literally the previous implementation (same init order and seed).
    return DeepCross(cards, n_num, n_text) if name == "baseline" else TunedDeepCross(
        cards, n_num, n_text, config(name))


def sha_ids(ids):
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def verify_features(data_path, feature_variant):
    manifest = json.loads(Path(data_path).with_suffix(".json").read_text())
    if (manifest.get("full_train"), manifest.get("validation"), manifest.get("test")) != (147777, 31668, 31682):
        raise ValueError("Not the original canonical asking-price dataset")
    if manifest.get("no_hierarchy") is not True:
        raise ValueError("No hierarchical residual correction permitted in tuning")
    if feature_variant == "minilm":
        if (manifest.get("variant") != "minilm-2-oob-price" or
            manifest.get("minilm_internal_stopper_labels_used") is not False or
            manifest.get("minilm_validation_test_labels_used") is not False):
            raise ValueError("MiniLM source must be the audited two-out-of-bag version")
    elif manifest.get("price_derived_inputs") is not False:
        raise ValueError("SVD source must have no price-derived features")
    return manifest


def run(a):
    s = config(a.variant)
    if a.features not in FEATURES:
        raise ValueError("Invalid pretrained text feature family")
    seed_everything()
    manifest = verify_features(a.data, a.features)
    with np.load(a.data, allow_pickle=False) as f:
        num, cat, txt = f["num"].astype(np.float32), f["cat"].astype(np.int64), f["txt"].astype(np.float32)
        y, phase, ids = f["y"].astype(np.float64), f["phase"].astype(str), f["ids"].astype(str)
    if len(ids) != 211127 or len(np.unique(ids)) != len(ids):
        raise ValueError("Missing or duplicate canonical vehicles")
    tr = np.flatnonzero(phase == "train")
    va = np.flatnonzero(phase == "validation")
    te = np.flatnonzero(phase == "test")
    if (len(tr), len(va), len(te)) != (147777, 31668, 31682):
        raise ValueError("Original chronological holdouts changed")
    if any(sha_ids(ids[phase == p]) != manifest["holdout_row_sha256"][p]
           for p in ("validation", "test")):
        raise ValueError("Holdout identity fingerprint mismatch")
    if not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("Invalid observed asking prices")
    fit, internal = tr[:-14777], tr[-14777:]
    if len(fit) != 133000 or np.intersect1d(fit, internal).size:
        raise ValueError("Prior neural early-stopper split drift")
    dataset = TensorDataset(torch.from_numpy(num), torch.from_numpy(cat),
                            torch.from_numpy(txt), torch.from_numpy(np.log(y).astype(np.float32)))
    net = build(a.variant, manifest["cardinalities"], num.shape[1], txt.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=s["lr"], weight_decay=s["weight_decay"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=s["epochs"], eta_min=.00006)
    train = DataLoader(Subset(dataset, fit.tolist()), shuffle=True, batch_size=s["batch_size"], num_workers=0)
    best = float("inf")
    state = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, s["epochs"]+1):
        net.train()
        loss_sum, rows = 0., 0
        for numerics, categories, text, target in train:
            opt.zero_grad(set_to_none=True)
            numerics = numerics + torch.randn_like(numerics) * .008
            text = text + torch.randn_like(text) * .008
            raw = net(numerics, categories, text)
            loss = objective(raw, target, "aft")
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite exact-observation log-normal AFT loss")
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.)
            opt.step()
            loss_sum += float(loss.detach()) * len(target)
            rows += len(target)
        schedule.step()
        pred = predict(net, dataset, internal, s["batch_size"])
        score = metrics(y[internal], pred)["mae_usd"]
        print(json.dumps({"variant":a.variant,"features":a.features,"epoch":epoch,
                          "train_loss":loss_sum/rows,"internal_mae_usd":score}),flush=True)
        if score < best - .1:
            best, state, best_epoch, stale = score, copy.deepcopy(net.state_dict()), epoch, 0
        else:
            stale += 1
            if stale >= s["patience"]:
                break
    if state is None:
        raise ValueError("No valid neural checkpoint")
    net.load_state_dict(state)
    vpred = predict(net, dataset, va, s["batch_size"])
    tpred = predict(net, dataset, te, s["batch_size"])
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id":np.concatenate((ids[va], ids[te])),
                  "split":["validation"]*len(va)+["test"]*len(te),
                  "observed_price":np.concatenate((y[va],y[te])),
                  "predicted_price":np.concatenate((vpred,tpred))}).to_csv(out/"predictions.csv.gz",index=False)
    torch.save({"state_dict":state,"architecture":"deepcross","variant":a.variant,
                "feature_variant":a.features,"settings":s,"cardinalities":manifest["cardinalities"],
                "numeric_dimensions":num.shape[1],"text_dimensions":txt.shape[1]},out/"weights.pt")
    rec = {"status":"completed","market":"ask","architecture":"deepcross","loss":"aft",
           "variant":a.variant,"feature_variant":a.features,"settings":s,
           "original_train_rows":len(tr),"train_fit":len(fit),"train_internal_stopper":len(internal),
           "best_epoch":best_epoch,"internal_mae_usd":best,
           "original_validation":metrics(y[va],vpred),"original_test":metrics(y[te],tpred),
           "holdout_row_sha256":manifest["holdout_row_sha256"],
           "canonical_cohort_sha256":manifest["cohort_sha256"],
           "selection_policy":"validation MAE only; original test never used for hyperparameter selection",
           "outlier_policy":"full original training, no Lemon trimming",
           "text_provenance":manifest.get("variant","train-only TF-IDF+SVD 24"),
           "aft":"exact observed log-normal; no invented censoring",
           "no_hierarchy":True}
    (out/"result.json").write_text(json.dumps(rec,indent=2,allow_nan=False))
    print(json.dumps({"completed":a.variant,"text":a.features,"validation_mae":rec["original_validation"]["mae_usd"],
                     "test_mae":rec["original_test"]["mae_usd"]}),flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant",required=True,choices=tuple(CONFIGS))
    p.add_argument("--features",required=True,choices=FEATURES)
    p.add_argument("--data",required=True)
    p.add_argument("--output",required=True)
    run(p.parse_args())
