"""Granite R2 97M multilingual Ask: pinned INT8 ONNX encoder + audited native DeepCross head.

No fallback to English BGE, no float-precision encoder, no test leakage. Checkpoint
must be installed by the audited GitHub Actions release workflow. The 97M encoder
is fetched by immutable Hugging Face revision and its ONNX bytes SHA-256 checked.
Only the compact neural price head remains FP32 for exact training parity.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path("models/neural-granite-bohb-ask")
MODEL = "ibm-granite/granite-embedding-97m-multilingual-r2"
REVISION = "c61e626a6255c490879d0af885078b61929d51f6"
ONNX_FILE = "onnx/model_quint8_avx2.onnx"
ONNX_SHA256 = "a6022dd8220ea6f6595562a1328ee216f4a94faa55362f2f4747c80f1e78772e"
COHORT_SHA256 = "746bc9f8e02e1e708ea5e1d9479408effb131eb6b1d1ea7710380701e62dcf12"
EMBEDDING_SHA256 = "7f2be5c0e2199ca8ca23c480140b6f8b6707b119594efb9f3278c415e7b8e5f2"
CHECKPOINT_SHA256 = "bc26b6ae67a5e43558248c9780f1667cdae8c609f5f24e1e07902db5ba2ca094"
CLEANED_CVAR95 = 9392.757506502196


def _sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def manifest(root):
    folder = Path(root) / ROOT
    try:
        path = folder / "manifest.json"
        weights = folder / "weights.pt"
        if not path.is_file() or not weights.is_file():
            return None
        meta = json.loads(path.read_text())
        if not (meta.get("status") == "passed"
                and meta.get("source_run_id") == 42518049406
                and meta.get("source_artifact_id") == 13649875732
                and meta.get("source_slot") == 8
                and meta.get("encoder") == MODEL
                and meta.get("encoder_revision") == REVISION
                and meta.get("encoder_sha256") == ONNX_SHA256
                and meta.get("cohort_sha256") == COHORT_SHA256
                and meta.get("embedding_sha256") == EMBEDDING_SHA256
                and meta.get("checkpoint_sha256") == CHECKPOINT_SHA256
                and meta.get("cleaned_validation", {}).get("CVaR95_absolute_USD") == CLEANED_CVAR95
                and meta.get("original_test_scored") is False
                and _sha(weights) == CHECKPOINT_SHA256):
            return None
        return meta
    except (OSError, ValueError, TypeError):
        return None


def available(root):
    return manifest(root) is not None


class _GraniteInt8Encoder:
    """Use precisely the published SentenceTransformer ONNX encoder used at training."""

    def __init__(self):
        from huggingface_hub import hf_hub_download
        from sentence_transformers import SentenceTransformer
        graph = Path(hf_hub_download(repo_id=MODEL, filename=ONNX_FILE,
                                     revision=REVISION))
        if _sha(graph) != ONNX_SHA256:
            raise ValueError("The IBM Granite INT8 ONNX file changed from training")
        self.model = SentenceTransformer(
            MODEL, revision=REVISION, trust_remote_code=False,
            device="cpu", backend="onnx",
            model_kwargs={"file_name": ONNX_FILE})
        self.model.max_seq_length = 256
        if self.model.get_sentence_embedding_dimension() != 384:
            raise ValueError("Granite R2 must produce exactly 384 coordinates")
        encoder = self.model[0]
        if not (getattr(encoder, "backend", None) == "onnx"
                or "ORT" in str(type(getattr(encoder, "auto_model", None)))):
            raise ValueError("Granite fell back to a non-ONNX encoder")

    def encode(self, clean_text):
        vector = np.asarray(self.model.encode(
            [clean_text], batch_size=1, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)
        if vector.shape != (1, 384) or not np.isfinite(vector).all():
            raise ValueError("Granite ONNX output shape/finite check failed")
        if not np.allclose(np.linalg.norm(vector, axis=1), 1, atol=.006):
            raise ValueError("Granite embedding normalization changed")
        # The training shards stored 384D float16 embeddings, then restored FP32.
        return vector.astype(np.float16).astype(np.float32)


class GraniteBOHBAsking:
    def __init__(self, root):
        import torch
        from pricing.neural_ask_embedding_live import NUMS, CATS
        from scripts.ask25_joint_search import ResearchAskingNetwork
        from scripts.ray_pbt_ask import architecture_params

        root = Path(root)
        self.manifest = manifest(root)
        if self.manifest is None:
            raise FileNotFoundError("Unverified Granite BOHB release")
        prep = json.loads((root / "models/neural-experimental-ask-bge" /
                           "preprocessing.json").read_text())
        if tuple(prep["numeric_names"]) != NUMS or tuple(prep["categorical_names"]) != CATS:
            raise ValueError("Granite input features differ from the trained cohort")
        self.median = np.asarray(prep["numeric_median"], dtype=np.float64)
        self.scale = np.asarray(prep["numeric_scale"], dtype=np.float64)
        self.maps = prep["category_maps"]
        self.cards = list(prep["cardinalities"])
        checkpoint = torch.load(root / ROOT / "weights.pt", map_location="cpu",
                                weights_only=True)
        expected = architecture_params("deepcross")
        if (checkpoint.get("architecture") != expected
                or checkpoint.get("cohort_sha256") != COHORT_SHA256
                or checkpoint.get("embedding_sha256") != EMBEDDING_SHA256
                or checkpoint.get("training_cohort") != "frozen-lemon-oof-retained-75"
                or checkpoint.get("test_scored") is not False
                or checkpoint.get("best_record", {}).get("inner_metrics", {}).get(
                    "CVaR95_absolute_USD") != 8638.459999648085):
            raise ValueError("Granite checkpoint is not the verified worker-8 winner")
        self.price_network = ResearchAskingNetwork(self.cards, 384, expected)
        self.price_network.load_state_dict(checkpoint["state_dict"], strict=True)
        self.price_network.eval()
        self.encoder = _GraniteInt8Encoder()
        self.encoder_backend = "IBM Granite R2 97M · verified INT8 ONNX CPU"

    def prepare_text(self, description):
        from scripts.common import clean
        raw = str(description or "")[:8000]
        return {"embedding": self.encoder.encode(clean(raw)), "raw": raw}

    def predict_prepared(self, fields, prepared):
        import torch
        from pricing.neural_ask_embedding_live import numerical, categoricals
        num = numerical(fields, prepared["raw"])
        numeric = np.clip((np.where(np.isfinite(num), num, self.median) -
                           self.median) / self.scale, -8, 8).astype(np.float32)
        cat = categoricals(fields, self.maps)
        embedding = np.asarray(prepared["embedding"], dtype=np.float32)
        if numeric.shape != (7,) or cat.shape != (1, 11) or embedding.shape != (1, 384):
            raise ValueError("The Granite pricing feature shape changed")
        with torch.inference_mode():
            raw = float(self.price_network(
                torch.from_numpy(numeric.reshape(1, 7)),
                torch.from_numpy(cat.astype(np.int64)),
                torch.from_numpy(embedding))[0, 0].item())
        price = float(np.exp(np.clip(raw, math.log(100), math.log(1000000))))
        if not math.isfinite(price) or price <= 0:
            raise ValueError("Granite price head returned an invalid price")
        return {"market": "ask", "price": price,
                "model": "Granite R2 97M · INT8 ONNX + DeepCross · 9 Nov 2026",
                "historical_anchor": "2021-05-01",
                "text_price_heads": 0,
                "current_market_accuracy_validated": False}

    def predict(self, description, fields, explain=False):
        return self.predict_prepared(fields, self.prepare_text(description))


def load(root):
    import streamlit as st
    folder = Path(root) / ROOT
    if not available(root):
        raise FileNotFoundError("Granite multilingual release has not passed its gate")
    signature = tuple((name, (folder / name).stat().st_size,
                       (folder / name).stat().st_mtime_ns)
                      for name in ("weights.pt", "manifest.json"))

    @st.cache_resource(max_entries=1, show_spinner=False)
    def _cache(path, fingerprint):
        return GraniteBOHBAsking(path)

    return _cache(str(root), signature)
