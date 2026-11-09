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
    """Exact pinned IBM QInt8 graph, native ONNX Runtime, no Optimum dependency.

    The published SentenceTransformer model has CLS-only pooling. We call
    the identical tokenizer and ONNX graph directly, then reproduce the
    normalize_embeddings=True and float16 storage rounding used in training.
    CI compares output to the official SentenceTransformer ONNX reference.
    """

    def __init__(self):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        graph = Path(hf_hub_download(
            repo_id=MODEL, filename=ONNX_FILE, revision=REVISION))
        if _sha(graph) != ONNX_SHA256:
            raise ValueError("IBM Granite QInt8 graph checksum mismatch")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL, revision=REVISION, trust_remote_code=False,
            use_fast=True)
        config = json.loads(Path(hf_hub_download(
            repo_id=MODEL, filename="1_Pooling/config.json",
            revision=REVISION)).read_text())
        if not (config.get("word_embedding_dimension") == 384
                and config.get("pooling_mode_cls_token") is True
                and config.get("pooling_mode_mean_tokens") is False
                and config.get("pooling_mode_lasttoken") is False):
            raise ValueError("Granite's official CLS pooling definition changed")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(graph), sess_options=opts, providers=["CPUExecutionProvider"])
        if self.session.get_providers() != ["CPUExecutionProvider"]:
            raise ValueError("Granite must run on CPU ONNX Runtime")
        self.input_names = tuple(item.name for item in self.session.get_inputs())
        if not {"input_ids", "attention_mask"}.issubset(self.input_names):
            raise ValueError("Unexpected IBM Granite ONNX input contract")
        self.output_names = tuple(item.name for item in self.session.get_outputs())

    def encode(self, clean_text):
        tokens = self.tokenizer(
            [str(clean_text)], max_length=256, truncation=True,
            padding=True, return_tensors="np")
        inputs = {}
        for name in self.input_names:
            if name in tokens:
                inputs[name] = np.asarray(tokens[name], dtype=np.int64)
            elif name == "token_type_ids":
                inputs[name] = np.zeros_like(
                    np.asarray(tokens["input_ids"], dtype=np.int64))
            else:
                raise ValueError("Unrecognized required ONNX input: " + name)

        values = self.session.run(None, inputs)
        hidden = np.asarray(values[0], dtype=np.float32)
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[-1] != 384:
            raise ValueError("Granite QInt8 did not return token embeddings")
        # 1_Pooling/config.json: CLS token, no mean/last-token pooling.
        vec = hidden[:, 0, :]
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        if not np.isfinite(norm).all() or (norm < 1e-10).any():
            raise ValueError("Granite QInt8 returned an invalid embedding")
        normalized = vec / norm
        # Research shards store float16 vectors restored to FP32 at inference.
        output = normalized.astype(np.float16).astype(np.float32)
        if output.shape != (1, 384) or not np.isfinite(output).all():
            raise ValueError("Granite QInt8 output failed shape / finite check")
        return output


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
        shape_keys=("family","project_dim","project_layers","project_dropout",
                    "cross_layers","rank","experts","width","blocks","dropout","activation")
        if (any(checkpoint.get("architecture",{}).get(k)!=expected.get(k) for k in shape_keys)
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
