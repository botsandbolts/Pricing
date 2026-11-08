"""Frozen Granite INT8 ONNX 384D feature loader for Lemon75 BOHB.

Reuse original structured 2021 Ask labels/IDs and a frozen training-only
Lemon OOF membership mask. The Granite embedding artifact is price-free,
immutable, row-ordered and SHA checked. Validation and test never enter
training or BOHB's internal CVaR95 selection.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.ray_pbt_ask_lemon_1h import lemon_indices
from scripts.granite_lemon75_encode import (
    MODEL_ID, MODEL_REV, INT8_SHA256, VARIANT, DIM, N, MAX_TOKENS)

def fingerprint(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def frozen_source(features, embeddings, candidate=VARIANT):
    if candidate != VARIANT:
        raise ValueError("Granite trial cannot silently fall back to BGE")
    features = Path(features)
    embedding_dir = Path(embeddings)
    source = json.loads(features.with_suffix(".json").read_text())
    manifest = json.loads((embedding_dir/"manifest.json").read_text())
    required = ("status","variant","model_id","model_revision",
                "model_onnx_sha256","encoder_inference","row_count",
                "dimensionality","max_tokens","canonical_cohort_sha256")
    actual = tuple(manifest.get(key) for key in required)
    expected = ("complete",VARIANT,MODEL_ID,MODEL_REV,INT8_SHA256,"INT8 ONNX",
                N,DIM,MAX_TOKENS,source["cohort_sha256"])
    if actual != expected or manifest["holdout_row_sha256"]!=source["holdout_row_sha256"]:
        raise ValueError("Wrong or unaudited Granite INT8 ONNX cohort")
    if (manifest["price_derived_features"] is not False
            or manifest["fine_tuned_on_price"] is not False
            or manifest["test_labels_used_in_encoder"] is not False):
        raise ValueError("Price-derived Granite embeddings are forbidden")
    path = embedding_dir/"embeddings.npy"
    if fingerprint(path)!=manifest["embedding_file_sha256"]:
        raise ValueError("Granite input embeddings checksum changed")
    values = np.load(path, mmap_mode="r")
    if values.shape!=(N,DIM) or values.dtype != np.float16 or not np.isfinite(values).all():
        raise ValueError("Wrong pretrained Granite embedding shape")
    with np.load(features, allow_pickle=False) as file:
        z = {k:file[k] for k in ("num","cat","y","phase","ids","lemon")}
    phase = z["phase"].astype(str)
    if (len(z["ids"]) != N or
            (int((phase=="train").sum()),int((phase=="validation").sum()),
             int((phase=="test").sum()))!=(147777,31668,31682)):
        raise ValueError("Canonical original Ask cohort changed")
    if z["num"].shape != (N,7) or z["cat"].shape != (N,11):
        raise ValueError("Original tabular features must match BGE trial")
    if len(np.unique(z["ids"])) != N:
        raise ValueError("Original row ID collision")
    if not np.isfinite(z["y"]).all() or (z["y"]<=0).any():
        raise ValueError("Invalid price target in original frozen Ask data")
    encoder = {
        "embedding_file_sha256": manifest["embedding_file_sha256"],
        "model_revision": MODEL_REV,
        "encoder_backend": "ONNX INT8 CPU",
        "quantized_checkpoint_sha256": INT8_SHA256,
        "dimensionality": DIM,
    }
    return z, values, source, encoder


def frozen_lemon_data(config):
    z, vector, source, encoder = frozen_source(
        config["features"],config["embeddings"],VARIANT)
    fit,inner = lemon_indices(z["phase"],z["lemon"])
    cohorts = {}
    for name, index in (("fit",fit),("inner",inner)):
        cohorts[name] = {
            "numeric":torch.from_numpy(z["num"][index].astype(np.float32)),
            "category":torch.from_numpy(z["cat"][index].astype(np.int64)),
            "vector":torch.from_numpy(vector[index].astype(np.float32)),
            "log_target":torch.from_numpy(np.log(z["y"][index]).astype(np.float32)),
            "truth":z["y"][index].astype(np.float64),
        }
    meta = {
        "cohort_sha256":source["cohort_sha256"],
        "holdout_row_sha256":source["holdout_row_sha256"],
        "embedding_sha256":encoder["embedding_file_sha256"],
        "encoder_revision":encoder["model_revision"],
        "encoder_onnx_int8_sha256":INT8_SHA256,
        "cardinalities":source["cardinalities"],
        "fit_rows":len(fit),"train_only_inner_rows":len(inner),
        "retained_train_rows":110832,"trimmed_train_rows":36945,
        "original_validation_rows":31668,"original_test_rows":31682,
        "selection_uses_original_validation":False,"test_scored":False,
    }
    return cohorts,meta
