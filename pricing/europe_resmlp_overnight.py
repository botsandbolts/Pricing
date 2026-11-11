"""Frozen 11-Nov-2026 European ResMLP / Weighted Tail-Huber ASK inference.

This is an exact architecture and input-transform reproduction of the
overnight CVaR95 best checkpoint from Actions run 42892216866. The model
was fitted on recently verified *observed* advertised ASK prices, NOT on
Eurostat-adjusted November labels. NEVER add a second HICP post-multiplier.

The source dataset is CC BY-NC-SA 4.0; code is independently MIT licensed.
"""
from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from pricing.europe_model import (
    CATEGORICAL, FEATURES, NUMERIC, EuropeanModelUnavailable,
    encode_vehicle, ROOT as OLD_ROOT,
)

ROOT = OLD_ROOT.parent / "europe-resmlp-overnight"
SOURCE_RUN = 42892216866
SOURCE_ARTIFACT = 13827450754
SUPPORTED_COUNTRIES = ("de", "fr", "nl", "be", "at", "es", "it", "pl")


def _architecture(levels, cfg, median_log):
    import torch
    from torch import nn

    if cfg.get("family") != "resmlp" or cfg.get("loss") != "weighted_tail_huber":
        raise EuropeanModelUnavailable("Unsupported overnight ResMLP configuration.")

    class ResidualBlock(nn.Module):
        def __init__(self, width, dropout, activation):
            super().__init__()
            act = nn.GELU if activation == "gelu" else nn.SiLU
            self.net = nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, width), act(),
                nn.Dropout(dropout), nn.Linear(width, width), nn.Dropout(dropout),
            )

        def forward(self, x):
            return x + .4 * self.net(x)

    class ResMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.mean_log = float(median_log)
            dim = int(cfg["dim"])
            width = int(cfg["width"])
            layers = int(cfg["layers"])
            dropout = float(cfg["dropout"])
            act = nn.GELU if cfg["activation"] == "gelu" else nn.SiLU
            self.embeddings = nn.ModuleList([
                nn.Embedding(int(level) + 1, dim) for level in levels
            ])
            self.num_weight = nn.Parameter(torch.empty(len(NUMERIC), dim))
            self.num_bias = nn.Parameter(torch.empty(len(NUMERIC), dim))
            dim_in = len(NUMERIC) * 2 + len(CATEGORICAL) * dim
            self.proj = nn.Sequential(
                nn.LayerNorm(dim_in), nn.Linear(dim_in, width),
                act(), nn.Dropout(dropout),
            )
            self.blocks = nn.Sequential(*[
                ResidualBlock(width, dropout, cfg["activation"])
                for _ in range(layers)
            ])
            self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

        def forward(self, num, cat):
            import torch
            catemb = torch.stack([
                embedding(cat[:, i])
                for i, embedding in enumerate(self.embeddings)
            ], dim=1)
            full = torch.cat([num, catemb.flatten(1)], dim=1)
            raw = self.head(self.blocks(self.proj(full))).squeeze(-1)
            return self.mean_log + raw

    return ResMLP()


def _prepared(rows, pack, schema):
    import torch
    n = len(NUMERIC)
    mean = np.asarray(pack["train_mean"], dtype=np.float32)
    std = np.asarray(pack["train_std"], dtype=np.float32)
    if (mean.shape != (n,) or std.shape != (n,) or
            not np.isfinite(mean).all() or not np.isfinite(std).all() or
            not np.all(std > 0)):
        raise EuropeanModelUnavailable("Frozen ResMLP numeric normalizer invalid.")
    raw = np.asarray(rows[:, :n], dtype=np.float32)
    finite = np.isfinite(raw)
    standardized = np.clip((np.where(finite, raw, mean) - mean) / std,
                           -7., 7.).astype(np.float32)
    num = np.concatenate([standardized, finite.astype(np.float32)], axis=1)
    raw_cat = np.asarray(rows[:, n:], dtype=np.float32)
    categories = np.where(np.isfinite(raw_cat), raw_cat, -1).astype(np.int64) + 1
    for j, key in enumerate(CATEGORICAL):
        cap = len(schema["category_vocabulary"][key])
        categories[:, j] = np.where(
            (categories[:, j] >= 1) & (categories[:, j] <= cap),
            categories[:, j], 0,
        )
    return (
        torch.from_numpy(np.ascontiguousarray(num)),
        torch.from_numpy(np.ascontiguousarray(categories)),
    )


@lru_cache(maxsize=1)
def load_bundle(path=None):
    import torch
    root = Path(path) if path is not None else ROOT
    paths = [root / name for name in (
        "champion.pt", "comparison.json", "schema.json", "release_manifest.json"
    )]
    if not all(p.is_file() for p in paths):
        raise EuropeanModelUnavailable(
            "New European ResMLP model package is incomplete; valuation disabled."
        )
    manifest = json.loads((root / "release_manifest.json").read_text())
    comparison = json.loads((root / "comparison.json").read_text())
    schema = json.loads((root / "schema.json").read_text())
    best = comparison["overall_champion"]
    expected = manifest.get("checkpoint_sha256")
    if (
        manifest.get("source_run_id") != SOURCE_RUN
        or manifest.get("source_artifact_id") != SOURCE_ARTIFACT
        or manifest.get("model") != "ResMLP Weighted Tail-Huber"
        or schema.get("trained_model") != "resmlp"
        or schema.get("train_only_vocabulary") is not True
        or schema.get("feature_names") != FEATURES
        or schema.get("source_run_id") != SOURCE_RUN
        or schema.get("source_training_target") != "original_price_eur"
        or schema.get("automatic_eurostat_adjustment") is not False
        or best.get("architecture") != "resmlp"
        or best.get("loss") != "weighted_tail_huber"
        or best.get("origin") != "eu-overnight-20261111-wave1-0"
        or best.get("neural_epoch") != 76
        or comparison.get("compared_models") != 30
        or best.get("kind") != "neural"
        or best.get("sealed_test_sha256") != schema.get("test_split_sha256")
        or manifest.get("schema_training_sha256") != schema.get("training_split_sha256")
        or manifest.get("schema_validation_sha256") != schema.get("validation_split_sha256")
        or set(schema.get("category_vocabulary", {}).get("seller_country", {}))
           != set(SUPPORTED_COUNTRIES)
    ):
        raise EuropeanModelUnavailable(
            "ResMLP checkpoint or canonical country vocabulary not verified."
        )
    if not expected or hashlib.sha256(paths[0].read_bytes()).hexdigest() != expected:
        raise EuropeanModelUnavailable("ResMLP checkpoint SHA256 mismatch.")
    pack = torch.load(paths[0], weights_only=True, map_location="cpu")
    if (
        pack["config"] != best["config"]
        or pack["manifest_train_sha"] != schema["training_split_sha256"]
        or pack["manifest_valid_sha"] != schema["validation_split_sha256"]
        or pack.get("test_scored") is not False
    ):
        raise EuropeanModelUnavailable(
            "Frozen ResMLP tensor is not the selected validation champion."
        )
    model = _architecture(
        [len(schema["category_vocabulary"][x]) for x in CATEGORICAL],
        pack["config"], pack["train_median_log"],
    )
    try:
        model.load_state_dict(pack["state_dict"], strict=True)
    except RuntimeError as exc:
        raise EuropeanModelUnavailable("ResMLP layer or tensor mismatch.") from exc
    model._frozen_normalization = {
        "train_mean": pack["train_mean"], "train_std": pack["train_std"],
    }
    model.eval()
    return model, schema, comparison


def available(path=None):
    root = Path(path) if path is not None else ROOT
    return all((root / name).is_file() for name in
               ("champion.pt", "comparison.json", "schema.json", "release_manifest.json"))


def batch_estimate(vehicles, bundle=None):
    import torch
    if not vehicles:
        return np.asarray([], dtype=np.float64)
    model, schema, _ = bundle if bundle is not None else load_bundle()
    pack = model._frozen_normalization
    for v in vehicles:
        if v.get("seller_country", "").strip().lower() not in SUPPORTED_COUNTRIES:
            raise ValueError("Choose one of the eight supported European countries.")
    x = np.concatenate([encode_vehicle(v, schema) for v in vehicles], axis=0)
    num, cat = _prepared(x, pack, schema)
    result = []
    with torch.inference_mode():
        for i in range(0, len(vehicles), 256):
            out = model(num[i:i + 256], cat[i:i + 256])
            price = torch.exp(out.clamp(math.log(100.), math.log(1_000_000.)))
            result.append(price.cpu().numpy())
    result = np.concatenate(result).astype(np.float64)
    if not np.isfinite(result).all():
        raise EuropeanModelUnavailable("ResMLP predicted nonfinite ASK amounts.")
    return np.clip(result, 100., 1_000_000.)


def estimate(vehicle, bundle=None):
    return float(batch_estimate([vehicle], bundle=bundle)[0])


def explain(vehicle, bundle=None):
    """Actual counterfactual network calls, not purported SHAP explanations."""
    trained = bundle if bundle is not None else load_bundle()
    original = estimate(vehicle, bundle=trained)
    variants = []
    km = float(vehicle["mileage_km"])
    year = int(vehicle["reg_year"])
    if km < 2_000_000:
        variants.append(("10% more kilometres", {
            **vehicle, "mileage_km": min(2_000_000., km * 1.1 + (1000. if km == 0 else 0.))
        }))
    if year < 2026:
        variants.append(("One year newer registration", {**vehicle, "reg_year": year + 1}))
    if year > 1950:
        variants.append(("One year older registration", {**vehicle, "reg_year": year - 1}))
    hp = vehicle.get("power_hp")
    if hp not in (None, "") and 20 <= float(hp) < 1090:
        variants.append(("10% more horsepower", {
            **vehicle, "power_hp": min(1200., float(hp) * 1.10)
        }))
    if not variants:
        return []
    predictions = batch_estimate([x for _, x in variants], bundle=trained)
    return [(label, float(v - original)) for (label, _), v in zip(variants, predictions)]
