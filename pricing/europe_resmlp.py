"""Frozen European ResMLP ASK inference, byte-verified against Ray winner.

The production weights are taken from GitHub Actions run 42679651225,
not retrained, synthesized, or re-selected on the untouched holdout.
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
    ROOT as EUROPE_MARKET_ROOT, encode_vehicle,
)

NEURAL_ROOT = EUROPE_MARKET_ROOT.parent / "europe-resmlp"
RUN_ID = 42679651225
ARTIFACT_ID = 13736822556


def _architecture(levels, config, median_log):
    """Exactly reproduce the winning ResMLP state-dict shapes and forward pass."""
    import torch
    from torch import nn

    class ResidualBlock(nn.Module):
        def __init__(self, width, dropout, activation):
            super().__init__()
            act = nn.GELU if activation == "gelu" else nn.SiLU
            self.net = nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, width), act(),
                nn.Dropout(dropout), nn.Linear(width, width), nn.Dropout(dropout)
            )

        def forward(self, x):
            return x + .4 * self.net(x)

    class DeepTabular(nn.Module):
        def __init__(self):
            super().__init__()
            self.mean_log = float(median_log)
            dim, width = int(config["dim"]), int(config["width"])
            self.embeddings = nn.ModuleList(
                [nn.Embedding(level + 1, dim) for level in levels]
            )
            self.num_weight = nn.Parameter(torch.empty(len(NUMERIC), dim))
            self.num_bias = nn.Parameter(torch.empty(len(NUMERIC), dim))
            act = nn.GELU if config["activation"] == "gelu" else nn.SiLU
            self.proj = nn.Sequential(
                nn.LayerNorm(len(NUMERIC) * 2 + len(CATEGORICAL) * dim),
                nn.Linear(len(NUMERIC) * 2 + len(CATEGORICAL) * dim, width),
                act(), nn.Dropout(float(config["dropout"]))
            )
            self.blocks = nn.Sequential(*[
                ResidualBlock(width, float(config["dropout"]), config["activation"])
                for _ in range(int(config["layers"]))
            ])
            self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

        def forward(self, num, cat):
            categorical = torch.stack([
                embedding(cat[:, i])
                for i, embedding in enumerate(self.embeddings)
            ], dim=1)
            full = torch.cat([num, categorical.flatten(1)], dim=1)
            raw = self.head(self.blocks(self.proj(full))).squeeze(-1)
            return self.mean_log + raw

    return DeepTabular()


def _prepared(rows, pack, schema):
    """Training-equivalent centering, missingness flags and categorical +1."""
    import torch

    numeric_count = len(NUMERIC)
    mean = np.asarray(pack["train_mean"], dtype=np.float32)
    std = np.asarray(pack["train_std"], dtype=np.float32)
    if mean.shape != (numeric_count,) or std.shape != (numeric_count,) or not np.all(std > 0):
        raise EuropeanModelUnavailable("Invalid frozen neural normalization statistics.")
    raw = np.asarray(rows[:, :numeric_count], dtype=np.float32)
    observed = np.isfinite(raw)
    filled = np.where(observed, raw, mean)
    standardized = np.clip((filled - mean) / std, -7., 7.).astype(np.float32)
    numeric = np.concatenate([standardized, observed.astype(np.float32)], axis=1)
    raw_cat = np.asarray(rows[:, numeric_count:], dtype=np.float32)
    categories = np.where(np.isfinite(raw_cat), raw_cat, -1).astype(np.int64) + 1
    for index, key in enumerate(CATEGORICAL):
        capacity = len(schema["category_vocabulary"][key])
        categories[:, index] = np.where(
            (categories[:, index] >= 1) & (categories[:, index] <= capacity),
            categories[:, index], 0
        )
    return (torch.from_numpy(np.ascontiguousarray(numeric)),
            torch.from_numpy(np.ascontiguousarray(categories)))


@lru_cache(maxsize=1)
def load_bundle(path=None):
    import torch

    root = Path(path) if path is not None else NEURAL_ROOT
    weight_file = root / "champion.pt"
    metadata_file = root / "comparison.json"
    manifest_file = root / "release_manifest.json"
    schema_file = EUROPE_MARKET_ROOT / "schema.json"
    if not all(p.is_file() for p in (weight_file, metadata_file, manifest_file, schema_file)):
        raise EuropeanModelUnavailable("Verified ResMLP release bundle is unavailable.")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    schema = json.loads(schema_file.read_text(encoding="utf-8"))
    comparison = json.loads(metadata_file.read_text(encoding="utf-8"))
    champion = comparison["overall_champion"]
    expected = manifest.get("checkpoint_sha256")
    if (manifest.get("source_run_id") != RUN_ID
        or manifest.get("source_artifact_id") != ARTIFACT_ID
        or champion.get("architecture") != "resmlp"
        or champion.get("loss") != "tail_mae_eur"
        or champion.get("kind") != "neural"
        or manifest.get("model") != "ResMLP Tail-MAE"
        or schema.get("train_only_vocabulary") is not True
        or schema.get("feature_names") != FEATURES
        or comparison.get("test_rows") != 201085
        or champion.get("sealed_test_sha256") != schema.get("test_split_sha256")
        or manifest.get("schema_training_sha256") != schema.get("training_split_sha256")):
        raise EuropeanModelUnavailable("Neural model provenance or European vocabulary mismatch.")
    if not expected or hashlib.sha256(weight_file.read_bytes()).hexdigest() != expected:
        raise EuropeanModelUnavailable("ResMLP checksum mismatch; pricing disabled.")
    pack = torch.load(weight_file, map_location="cpu", weights_only=True)
    if (pack.get("manifest_train_sha") != schema["training_split_sha256"]
        or pack.get("manifest_valid_sha") != schema["validation_split_sha256"]
        or pack.get("config") != champion["config"]
        or pack.get("test_scored") is not False):
        raise EuropeanModelUnavailable("Neural checkpoint differs from the frozen search winner.")
    levels = [len(schema["category_vocabulary"][field]) for field in CATEGORICAL]
    model = _architecture(levels, pack["config"], pack["train_median_log"])
    try:
        model.load_state_dict(pack["state_dict"], strict=True)
    except RuntimeError as exc:
        raise EuropeanModelUnavailable("ResMLP architecture/checkpoint mismatch.") from exc
    model.eval()
    model._frozen_normalization = {"train_mean": pack["train_mean"], "train_std": pack["train_std"]}
    return (model, schema, comparison)


def available(path=None):
    root = Path(path) if path is not None else NEURAL_ROOT
    return all((root / name).is_file() for name in
               ("champion.pt", "comparison.json", "release_manifest.json"))


def batch_estimate(vehicles, bundle=None):
    import torch

    if not vehicles:
        return np.asarray([], dtype=float)
    model, schema, _ = bundle if bundle is not None else load_bundle()
    pack = model._frozen_normalization
    rows = np.concatenate([encode_vehicle(vehicle, schema) for vehicle in vehicles], axis=0)
    numerical, categorical = _prepared(rows, pack, schema)
    results = []
    with torch.inference_mode():
        for start in range(0, len(vehicles), 256):
            prediction = model(numerical[start:start + 256], categorical[start:start + 256])
            values = torch.exp(prediction.clamp(math.log(100.), math.log(1_000_000.)))
            results.append(values.cpu().numpy())
    values = np.concatenate(results).astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise EuropeanModelUnavailable("ResMLP produced a non-finite estimate.")
    return np.clip(values, 100., 1_000_000.)


def estimate(vehicle, bundle=None):
    return float(batch_estimate([vehicle], bundle=bundle)[0])


def explain(vehicle, bundle=None):
    """Explicit one-factor what-if changes, NEVER fabricated SHAP attribution."""
    model_bundle = bundle if bundle is not None else load_bundle()
    baseline = estimate(vehicle, bundle=model_bundle)
    candidates = []
    mileage = float(vehicle["mileage_km"])
    year = int(vehicle["reg_year"])
    if mileage < 2_000_000:
        candidates.append(("10% more kilometres",
            {**vehicle, "mileage_km": min(2_000_000., mileage * 1.10 + (1000 if mileage == 0 else 0))}))
    if year < 2026:
        candidates.append(("One year newer registration",
                           {**vehicle, "reg_year": year + 1}))
    if year > 1950:
        candidates.append(("One year older registration",
                           {**vehicle, "reg_year": year - 1}))
    power = vehicle.get("power_hp")
    if power not in (None, "") and 20 <= float(power) < 1090:
        candidates.append(("10% more horsepower",
                           {**vehicle, "power_hp": min(1200., float(power) * 1.10)}))
    if not candidates:
        return []
    values = batch_estimate([item for _, item in candidates], bundle=model_bundle)
    return [(name, float(value - baseline))
            for (name, _), value in zip(candidates, values)]
