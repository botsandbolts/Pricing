"""BOHB English BGE Ask winner, original native price head and pinned encoder.

The research winner is reproducible only with the frozen BGE text embeddings
and the original 2021 preprocessor. Head weights remain FP32 to preserve price
parity and support gradient-based explanations. INT8 is gated independently.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np

FOLDER = Path("models/neural-bge-bohb-ask")
WEIGHTS_SHA256 = "d5e669d33fb34a379ca02837989beca757d43a1586cc99fa86e5c58d64a45c39"
COHORT_SHA256 = "746bc9f8e02e1e708ea5e1d9479408effb131eb6b1d1ea7710380701e62dcf12"
EMBED_SHA256 = "d619d6b8b4bb17475b500dbfb7835aeb125609c3ed54dbd6113264a2a7569a33"


def _sha(path):
    with Path(path).open("rb") as fd:
        return hashlib.file_digest(fd, "sha256").hexdigest()


def manifest(root):
    folder = Path(root) / FOLDER
    try:
        meta = json.loads((folder / "manifest.json").read_text())
        if not (
            meta.get("status") == "passed"
            and meta.get("source_run_id") == 42504289617
            and meta.get("source_slot") == 5
            and meta.get("cohort_sha256") == COHORT_SHA256
            and meta.get("embedding_sha256") == EMBED_SHA256
            and meta.get("weights_sha256") == WEIGHTS_SHA256
            and meta.get("cleaned_validation", {}).get("CVaR95_absolute_USD") == 8740.063663572575
            and meta.get("test_scored") is False
            and _sha(folder / "weights.pt") == WEIGHTS_SHA256
        ):
            return None
        return meta
    except (OSError, ValueError, KeyError, TypeError):
        return None


def available(root):
    return manifest(root) is not None


class BGEBOHBAsking:
    def __init__(self, root):
        import torch
        from scripts.ask25_joint_search import ResearchAskingNetwork
        from scripts.ray_pbt_ask import architecture_params
        from pricing.neural_ask_embedding_live import load as bge_load
        self.root = Path(root)
        self.manifest = manifest(root)
        if self.manifest is None:
            raise FileNotFoundError("BGE BOHB release has not passed the audit gate")
        self.original = bge_load(root)
        checkpoint = torch.load(self.root / FOLDER / "weights.pt",
                                map_location="cpu", weights_only=True)
        expected = architecture_params("deepcross")
        if (checkpoint.get("architecture") != expected
            or checkpoint.get("cohort_sha256") != COHORT_SHA256
            or checkpoint.get("embedding_sha256") != EMBED_SHA256
            or checkpoint.get("training_cohort") != "frozen-lemon-oof-retained-75"
            or checkpoint.get("test_scored") is not False
            or checkpoint.get("best_record", {}).get("inner_metrics", {}).get(
                "CVaR95_absolute_USD") != 7941.740520305461
        ):
            raise ValueError("BGE worker-5 checkpoint is not the audited winner")
        self.model = ResearchAskingNetwork(self.original.cards, 384, expected)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval()

    def prepare_text(self, description):
        return self.original.prepare_text(description)

    def predict_prepared(self, fields, prepared):
        import torch
        numeric, category, vector = self.original.encode_inputs(fields, prepared)
        with torch.inference_mode():
            raw = float(self.model(
                torch.from_numpy(np.asarray(numeric, dtype=np.float32)),
                torch.from_numpy(np.asarray(category, dtype=np.int64)),
                torch.from_numpy(np.asarray(vector, dtype=np.float32)))[0, 0].item())
        price = float(np.exp(np.clip(raw, math.log(100), math.log(1000000))))
        if not math.isfinite(price) or price <= 0:
            raise ValueError("Invalid BGE BOHB asking prediction")
        return {"market": "ask", "price": price,
                "model": "BGE English · BOHB DeepCross · 9 Nov 2026",
                "historical_anchor": "2021-05-01",
                "text_price_heads": 0,
                "current_market_accuracy_validated": False}

    def predict(self, description, fields, explain=False):
        return self.predict_prepared(fields, self.prepare_text(description))


def load(root):
    import streamlit as st
    folder = Path(root) / FOLDER
    if not available(root):
        raise FileNotFoundError("Audited BGE winner weights missing")
    signature = tuple((name, (folder / name).stat().st_size,
                       (folder / name).stat().st_mtime_ns)
                      for name in ("manifest.json", "weights.pt"))

    @st.cache_resource(max_entries=1, show_spinner=False)
    def _cache(path, stamp):
        return BGEBOHBAsking(path)

    return _cache(str(root), signature)
