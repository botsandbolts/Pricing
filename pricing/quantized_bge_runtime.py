"""Verified INT8 ONNX inference for the *original* frozen BGE-small encoder.

A serving BACKEND change only: tokenization, CLS pooling, L2 normalization,
256-token truncation and FP16-rounding all reproduce the trained encoder.
Never quantizes the two price heads used for Captum gradients.

The INT8 ONNX weights are generated once by CI from the pinned BAAI model
revision. If and only if a parity+RAM release manifest is installed, choose
the ONNX backend instead of loading a duplicate PyTorch/SentenceTransformer
encoder. No subprocesses, per-request model swaps or hidden surrogates.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

RELPATH=Path("models/bge-small-int8-onnx")
SOURCE="BAAI/bge-small-en-v1.5"
REVISION="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
WEIGHTS="model_int8.onnx"
TOKENIZER="tokenizer.json"
MANIFEST="manifest.json"
MAX_TOKENS=256


def manifest_path(root):
    return Path(root)/RELPATH/MANIFEST


def candidate_manifest(root,model_id=SOURCE,revision=REVISION):
    """Cheap metadata check, no encoder loading or external requests."""
    path=manifest_path(root)
    if not path.is_file():
        return None
    try:
        m=json.loads(path.read_text())
        if (
            m.get("status")!="passed"
            or m.get("source_model")!=model_id
            or m.get("source_revision")!=revision
            or m.get("format")!="onnx-int8-dynamic-avx2"
            or m.get("dimension")!=384 or m.get("max_tokens")!=MAX_TOKENS
            or m.get("price_head_unchanged") is not True
            or m.get("quantization_did_not_use_prices") is not True
            or m.get("max_price_drift_passed") is not True
            or m.get("memory_reduction_passed") is not True
        ):
            return None
        folder=path.parent
        if not all((folder/f).is_file() for f in (WEIGHTS,TOKENIZER)):
            return None
        return m
    except (OSError,ValueError,TypeError):
        return None


def preferred(root,model_id=SOURCE,revision=REVISION):
    """Default AUTO only after CI promotion; manual native override for recovery."""
    mode=os.environ.get("PRICING_BGE_BACKEND","auto").strip().lower()
    if mode=="native":
        return False
    if mode not in ("auto","int8"):
        raise ValueError("Invalid PRICING_BGE_BACKEND mode")
    ready=candidate_manifest(root,model_id,revision) is not None
    if mode=="int8" and not ready:
        raise FileNotFoundError("Requested INT8 BGE release is not verified")
    return ready


class VerifiedInt8BGE:
    """Small tokenizers+onnxruntime adapter matching SentenceTransformer.encode."""

    def __init__(self,root,model_id=SOURCE,revision=REVISION):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        manifest=candidate_manifest(root,model_id,revision)
        if manifest is None:
            raise FileNotFoundError("Verified INT8 release manifest is missing")
        folder=Path(root)/RELPATH
        for name,key in ((WEIGHTS,"weight_sha256"),(TOKENIZER,"tokenizer_sha256")):
            with (folder/name).open("rb") as source:
                digest=hashlib.file_digest(source,"sha256").hexdigest()
            if digest != manifest.get(key):
                raise ValueError("INT8 BGE checksum mismatch: "+name)
        self.max_seq_length=MAX_TOKENS
        self.tokenizer=Tokenizer.from_file(str(folder/TOKENIZER))
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS)
        settings=ort.SessionOptions()
        settings.intra_op_num_threads=2
        settings.inter_op_num_threads=1
        settings.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        settings.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Fewer persistent allocator pools; stable when serving many short texts.
        settings.enable_mem_pattern=False
        settings.enable_cpu_mem_arena=False
        self.session=ort.InferenceSession(
            str(folder/WEIGHTS),sess_options=settings,
            providers=["CPUExecutionProvider"])
        self.inputs={inp.name for inp in self.session.get_inputs()}
        supported={"input_ids","attention_mask","token_type_ids"}
        if not {"input_ids","attention_mask"}<=self.inputs or not self.inputs<=supported:
            raise ValueError("INT8 BGE ONNX input signature changed")
        self.outputs=[out.name for out in self.session.get_outputs()]

    def get_embedding_dimension(self):
        return 384

    def get_sentence_embedding_dimension(self):
        # Compatibility alias for older SentenceTransformer integrations.
        return self.get_embedding_dimension()

    def eval(self):
        return self

    def encode(self,texts,*,batch_size=1,convert_to_numpy=True,
               normalize_embeddings=True,show_progress_bar=False,**kwargs):
        if batch_size != 1 or not convert_to_numpy or not normalize_embeddings:
            raise ValueError("INT8 BGE serving must use one L2-normalized row")
        if isinstance(texts,str):
            texts=[texts]
        rows=[]
        for text in texts:
            token=self.tokenizer.encode(str(text))
            payload={
                "input_ids":np.asarray([token.ids],dtype=np.int64),
                "attention_mask":np.asarray([token.attention_mask],dtype=np.int64),
                "token_type_ids":np.asarray([token.type_ids],dtype=np.int64),
            }
            output=self.session.run(None,{k:v for k,v in payload.items()
                                          if k in self.inputs})
            token_hidden=next(
                (np.asarray(t) for t in output
                 if getattr(t,"ndim",0)==3 and t.shape[0]==1 and t.shape[-1]==384),
                None)
            if token_hidden is None:
                raise ValueError("INT8 BGE ONNX has no expected token embeddings")
            # Pinned BGE 1_Pooling is CLS, followed by 2_Normalize.
            cls=np.asarray(token_hidden[0,0],dtype=np.float32)
            norm=float(np.linalg.norm(cls))
            if not np.isfinite(norm) or norm<1e-9:
                raise ValueError("INT8 BGE produced an invalid CLS vector")
            rows.append(cls/norm)
        result=np.stack(rows).astype(np.float32)
        if result.shape!=(len(texts),384) or not np.isfinite(result).all():
            raise ValueError("INT8 BGE output dimension or finiteness mismatch")
        return result
