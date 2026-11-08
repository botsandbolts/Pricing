"""Fail-closed serving for the 8 November 2026 Ray Lemon75 DeepCross Ask.

The full neural PRICE network executes in ONNX Runtime with INT8 dynamic
quantized matrix weights. We never relabel the shared BGE text encoder as
INT8 unless its separate parity-gated release is installed. The saved native
checkpoint is used on demand for Captum (ONNX INT8 cannot provide gradients).
All price models reuse the original verified frozen BGE and categorical/
numerical preprocessing. Older proven releases are preserved as fallback.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

FOLDER = "models/neural-ray-lemon75-ask"
REQUIRED = ("head-int8.onnx", "weights.pt", "manifest.json", "verification.json")
STUDY_RUN = 42485146978
STUDY_ARTIFACT = 13621540013
RELEASE_DAY = "2026-11-08"


def _hash(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def manifest(root):
    folder = Path(root) / FOLDER
    if not all((folder / name).is_file() for name in REQUIRED):
        return None
    try:
        m = json.loads((folder / "manifest.json").read_text())
        v = json.loads((folder / "verification.json").read_text())
        if not (
            m.get("status") == "passed"
            and m.get("market") == "ask"
            and m.get("release_date") == RELEASE_DAY
            and m.get("study_run_id") == STUDY_RUN
            and m.get("artifact_id") == STUDY_ARTIFACT
            and m.get("architecture") == "deepcross"
            and m.get("training_cohort") == "frozen-lemon-oof-retained-75"
            and m.get("quantization") == "onnxruntime-dynamic-int8-matmul-gemm"
            and m.get("lemon_retained_train_rows") == 110832
            and m.get("filtered_validation_rows") == 23751
            and m.get("original_validation_rows") == 31668
            and m.get("filtered_validation_is_conditional") is True
            and m.get("original_test_scored") is False
            and m.get("native_head_for_explanations") is True
            and v.get("status") == "passed"
            and v.get("verified") is True
            and v.get("complete_original_validation_replay") is True
            and v.get("filtered_validation_replay") is True
            and v.get("native_vs_int8_parity") is True
            and v.get("production_source_unchanged") is True
            and _hash(folder / "head-int8.onnx") == m.get("onnx_sha256")
            and _hash(folder / "weights.pt") == m.get("native_sha256")
        ):
            return None
        return m
    except (OSError, ValueError, TypeError, KeyError):
        return None


def available(root):
    from pricing.neural_ask_embedding_live import available as original
    return original(root) and manifest(root) is not None


class RayLemonDeepCrossAsk:
    """Exact frozen BGE/preprocessing with an audited dynamic-INT8 neural head."""

    def __init__(self, root):
        import onnxruntime as ort
        from pricing.neural_ask_embedding_live import load as original_load
        self.root = Path(root)
        self.manifest = manifest(root)
        if self.manifest is None or not available(root):
            raise FileNotFoundError("Audited Lemon75 INT8 Ask release not installed")
        self.original = original_load(root)
        option = ort.SessionOptions()
        option.intra_op_num_threads = 2
        option.inter_op_num_threads = 1
        option.enable_mem_pattern = False
        option.enable_cpu_mem_arena = False
        option.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.root / FOLDER / "head-int8.onnx"),
            sess_options=option, providers=["CPUExecutionProvider"])
        if {input_.name for input_ in self.session.get_inputs()} != {
                "numeric", "category", "vector"}:
            raise ValueError("Quantized price head has an incompatible input signature")
        if len(self.session.get_outputs()) != 1:
            raise ValueError("Quantized price head has an incompatible output signature")
        self._explanation_model = None

    def prepare_text(self, description):
        return self.original.prepare_text(description)

    def predict_prepared(self, fields, prepared):
        numeric, category, vector = self.original.encode_inputs(fields, prepared)
        value = self.session.run(None, {
            "numeric": np.asarray(numeric, dtype=np.float32),
            "category": np.asarray(category, dtype=np.int64),
            "vector": np.asarray(vector, dtype=np.float32)})[0]
        raw = float(np.asarray(value).reshape(1, -1)[0, 0])
        price = float(np.exp(np.clip(raw, math.log(100.), math.log(1_000_000.))))
        if not math.isfinite(price) or price <= 0:
            raise ValueError("ONNX Lemon75 DeepCross returned an invalid price")
        return {
            "market": "ask", "price": price,
            "model": "08 Nov 2026 · Lemon75 DeepCross · INT8 ONNX",
            "historical_anchor": "2021-05-01",
            "text_price_heads": 0,
            "price_predictions_from": "Full verified ONNX INT8 DeepCross price head",
            "current_market_accuracy_validated": False,
        }

    def predict(self, description, fields, explain=False):
        return self.predict_prepared(fields, self.prepare_text(description))

    @property
    def model(self):
        """Native saved weights for on-click Captum, not for normal pricing."""
        if self._explanation_model is None:
            import torch
            from scripts.ask25_joint_search import ResearchAskingNetwork
            from scripts.ray_pbt_ask import architecture_params
            pack = torch.load(self.root / FOLDER / "weights.pt",
                              map_location="cpu", weights_only=True)
            expected = architecture_params("deepcross")
            if (pack.get("architecture") != expected
                    or pack.get("training_cohort") != "frozen-lemon-oof-retained-75"
                    or pack.get("original_test_scored") is not False):
                raise ValueError("Native explanation checkpoint does not match INT8 head")
            network = ResearchAskingNetwork(self.original.cards, 384, expected)
            network.load_state_dict(pack["state_dict"], strict=True)
            network.eval()
            self._explanation_model = network
        return self._explanation_model


def load(root):
    import streamlit as st
    folder = Path(root) / FOLDER
    if not available(root):
        raise FileNotFoundError("Lemon75 DeepCross INT8 release has not passed audit")
    stamp = tuple((name, (folder / name).stat().st_size,
                   (folder / name).stat().st_mtime_ns) for name in REQUIRED)
    @st.cache_resource(show_spinner=False, max_entries=1)
    def _cache(origin, fingerprint):
        return RayLemonDeepCrossAsk(origin)
    return _cache(str(root), stamp)
