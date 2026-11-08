"""Create audited, price-free Granite 97M R2 multilingual INT8 ONNX embeddings.

The encoder is the published IBM AVX2 QUInt8 ONNX checkpoint, NOT fp32 PyTorch
and NOT a newly quantized/exported model. We use the same 211127 canonical
vehicle descriptions, row IDs, OOF Lemon mask and 256-token BGE comparison
limit. Ten independent encoding shards, then row-identity-checked assembly.
No price labels or text-price features are inputs to the text encoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os

import numpy as np
import pandas as pd

from scripts.common import clean
from scripts.raw_embedding_ask_encode import canonical, boundaries, id_hash

MODEL_ID = "ibm-granite/granite-embedding-97m-multilingual-r2"
MODEL_REV = "c61e626a6255c490879d0af885078b61929d51f6"
FILE = "onnx/model_quint8_avx2.onnx"
INT8_SHA256 = "a6022dd8220ea6f6595562a1328ee216f4a94faa55362f2f4747c80f1e78772e"
VARIANT = "granite-97m-multilingual-r2-int8-onnx"
DIM = 384
MAX_TOKENS = 256
CHUNKS = 10
N = 211127


def fingerprint(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_encoder():
    from huggingface_hub import hf_hub_download
    from sentence_transformers import SentenceTransformer
    import onnx
    import torch

    torch.set_num_threads(2)
    weight = Path(hf_hub_download(repo_id=MODEL_ID, filename=FILE, revision=MODEL_REV))
    if fingerprint(weight) != INT8_SHA256:
        raise ValueError("Published INT8 ONNX SHA256 mismatch")
    graph = onnx.load(str(weight), load_external_data=False)
    qweights = sum(x.data_type in (onnx.TensorProto.INT8, onnx.TensorProto.UINT8)
                   for x in graph.graph.initializer)
    qops = sum(x.op_type in ("MatMulInteger", "QLinearMatMul", "DynamicQuantizeLinear",
                            "QLinearConv", "QuantizeLinear", "DequantizeLinear")
               for x in graph.graph.node)
    if not (qweights > 0 or qops > 0):
        raise ValueError("Requested ONNX graph has no genuine 8-bit weights or quant ops")
    del graph
    model = SentenceTransformer(
        MODEL_ID, revision=MODEL_REV, trust_remote_code=False, device="cpu",
        backend="onnx", model_kwargs={"file_name": FILE})
    model.max_seq_length = MAX_TOKENS
    if model.get_sentence_embedding_dimension() != DIM:
        raise ValueError("Granite must emit exactly 384D embeddings")
    transformer = model[0]
    if not (getattr(transformer, "backend", None) == "onnx"
            or "ORT" in type(getattr(transformer, "auto_model", None)).__name__
            or "ORT" in str(type(getattr(transformer, "auto_model", None)))):
        raise ValueError("Encoder selected a non-ONNX runtime backend")
    return model, {"model_id": MODEL_ID, "model_revision": MODEL_REV,
                   "model_onnx_file": FILE, "model_onnx_sha256": INT8_SHA256,
                   "int8_weight_tensors": int(qweights),
                   "int8_quant_nodes": int(qops),
                   "encoder_runtime": "onnxruntime-cpu-prequantized-quint8-avx2"}


def encode(model, texts, batch=1):
    """Use consistent per-description inference for batch-sensitive INT8 ONNX.

    We measured max absolute delta .0418 and min cosine .9783 when switching
    batch=1 versus batch=4. Do not weaken tolerance or train on inconsistent
    embeddings. Serving and training use batch_size=1.
    """
    if batch != 1:
        raise ValueError("Granite INT8 ONNX requires deterministic per-row inference")
    vectors = []
    for text in texts:
        item = np.asarray(model.encode(
            [text], batch_size=1, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)
        if item.shape != (1,DIM):
            raise ValueError("Granite ONNX vector shape changed")
        vectors.append(item[0])
    v = np.asarray(vectors, dtype=np.float32)
    if v.shape != (len(texts), DIM) or not np.isfinite(v).all():
        raise ValueError("Wrong Granite embedding shape or nonfinite result")
    if not np.allclose(np.linalg.norm(v, axis=1), 1., atol=.006):
        raise ValueError("Granite embeddings are not L2-normalized")
    return v


def smoke():
    model, meta = load_encoder()
    inputs = [
        "2019 Honda Civic, low mileage, clean title.",
        "Fahrzeug mit gepflegtem Motor und Klimaanlage.",
        "Un coche usado en buenas condiciones.",
        "Продается автомобиль с небольшим пробегом."
    ]
    a = encode(model, inputs, batch=1)
    b = np.concatenate([encode(model, [t], batch=1) for t in inputs], 0)
    cos = np.sum(a*b, axis=1)
    delta = float(np.max(np.abs(a-b)))
    if delta > .005 or float(cos.min()) < .999:
        raise ValueError(f"Repeated single-row INT8 ONNX inference differs {delta} / {cos.min()}")
    print("GRANITE_INT8_ONNX_SMOKE_OK", json.dumps({**meta,
          "dimensions": DIM, "comparison_max_tokens": MAX_TOKENS,
          "inference_batch_size": 1, "max_repeat_absolute_difference": delta,
          "min_repeat_cosine_similarity": float(cos.min()),
          "previous_unacceptable_batch4_cosine": 0.9782596826553345}),
          flush=True)


def chunk(args):
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    frame, ids, source = canonical(args.parquet, args.features)
    start, end = boundaries(len(ids), args.chunk)
    model, meta = load_encoder()
    text = frame.text.iloc[start:end].fillna("").astype(str).tolist()
    check_text = [clean(text[i]) for i in np.linspace(
        0, len(text)-1, min(4, len(text)), dtype=int)]
    b1 = encode(model, check_text, batch=1)
    b2 = np.concatenate([encode(model, [t], batch=1) for t in check_text], 0)
    delta = float(np.max(np.abs(b1-b2)))
    cos = float(np.min((b1*b2).sum(axis=1)))
    if delta > .005 or cos < .999:
        raise ValueError("Repeated individual Granite INT8 ONNX inference mismatch")
    whole = np.lib.format.open_memmap(
        folder/"embeddings.npy", mode="w+", dtype=np.float16,
        shape=(end-start, DIM))
    for i in range(0, len(text), 256):
        clean_text = [clean(x) for x in text[i:i+256]]
        whole[i:i+len(clean_text)] = encode(model, clean_text, batch=1).astype(np.float16)
        if i % 2048 == 0:
            print("GRANITE_INT8_ENCODE_PROGRESS", json.dumps({
                "chunk": args.chunk, "rows_done": min(i+256, len(text)),
                "chunk_rows": len(text)}), flush=True)
    whole.flush()
    del whole
    manifest = {**meta, "status": "complete_chunk", "variant": VARIANT,
        "chunk": args.chunk, "chunks": CHUNKS, "start": start, "end": end,
        "rows": end-start, "total_rows": N, "dimensionality": DIM,
        "max_tokens": MAX_TOKENS, "normalization": "L2",
        "embedding_precision": "float16", "encoder_inference": "INT8 ONNX",
        "inference_batch_size": 1,
        "price_labels_used_to_encode": False,
        "text_price_heads": 0, "fine_tuned_on_price": False,
        "input_text_cleaner": "scripts.common.clean (same as BGE)",
        "source_row_sha256": id_hash(ids[start:end]),
        "canonical_cohort_sha256": source["cohort_sha256"],
        "holdout_row_sha256": source["holdout_row_sha256"],
        "inference_batch_size": 1, "max_repeat_absolute_difference": delta,
        "min_repeat_cosine_similarity": cos,
        "embedding_file_sha256": fingerprint(folder/"embeddings.npy")}
    (folder/"chunk.json").write_text(json.dumps(manifest, indent=2))
    print("GRANITE_INT8_CHUNK_COMPLETE", json.dumps(manifest), flush=True)


def assemble(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    with np.load(args.features, allow_pickle=False) as f:
        ids = f["ids"].astype(str)
        phase = f["phase"].astype(str)
    source = json.loads(Path(args.features).with_suffix(".json").read_text())
    if len(ids) != N or (int(sum(phase=="train")), int(sum(phase=="validation")),
                         int(sum(phase=="test"))) != (147777, 31668, 31682):
        raise ValueError("Frozen original Ask cohort changed")
    matrix = np.lib.format.open_memmap(
        root/"embeddings.npy", mode="w+", dtype=np.float16, shape=(N,DIM))
    hashes = []
    for index in range(CHUNKS):
        path = Path(args.parts)/f"granite-int8-encoded-{index}"
        manifest = json.loads((path/"chunk.json").read_text())
        start,end = boundaries(N,index)
        required = ("status", "variant", "chunk", "chunks", "start", "end",
                    "rows", "total_rows", "dimensionality", "max_tokens",
                    "model_id", "model_revision", "model_onnx_sha256",
                    "source_row_sha256", "canonical_cohort_sha256",
                    "encoder_inference", "price_labels_used_to_encode")
        actual = tuple(manifest.get(key) for key in required)
        expected = ("complete_chunk", VARIANT,index,CHUNKS,start,end,end-start,N,
                    DIM,MAX_TOKENS,MODEL_ID,MODEL_REV,INT8_SHA256,
                    id_hash(ids[start:end]),source["cohort_sha256"],
                    "INT8 ONNX",False)
        if actual != expected or manifest["holdout_row_sha256"] != source["holdout_row_sha256"]:
            raise ValueError(f"Granite ONNX source provenance mismatch in chunk {index}")
        if manifest["text_price_heads"] or manifest["fine_tuned_on_price"]:
            raise ValueError("Detected target-derived embedding feature")
        part = path/"embeddings.npy"
        if fingerprint(part) != manifest["embedding_file_sha256"]:
            raise ValueError("Granite output chunk checksum mismatch")
        vec = np.load(part, mmap_mode="r")
        if vec.shape!=(end-start,DIM) or vec.dtype!=np.float16 or not np.isfinite(vec).all():
            raise ValueError("Wrong Granite chunk geometry or nonfinite values")
        matrix[start:end] = vec
        hashes.append(manifest["embedding_file_sha256"])
    matrix.flush()
    del matrix
    result = {"status": "complete", "variant": VARIANT,
        "model_id": MODEL_ID, "model_revision": MODEL_REV,
        "model_onnx_file": FILE, "model_onnx_sha256": INT8_SHA256,
        "encoder_inference": "INT8 ONNX", "embedding_file_sha256":
            fingerprint(root/"embeddings.npy"),
        "row_count": N, "dimensionality": DIM, "max_tokens": MAX_TOKENS,
        "chunks": CHUNKS, "chunk_sha256": hashes,
        "canonical_cohort_sha256": source["cohort_sha256"],
        "holdout_row_sha256": source["holdout_row_sha256"],
        "price_derived_features": False, "fine_tuned_on_price": False,
        "test_labels_used_in_encoder": False,
        "normalization": "L2", "embedding_precision": "float16",
        "input_text_cleaner": "scripts.common.clean"}
    (root/"manifest.json").write_text(json.dumps(result,indent=2))
    print("GRANITE_INT8_ASSEMBLY_COMPLETE",json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--features", default="training-cache/neural-ask/features.npz")
    p.add_argument("--parquet", default="training-cache/neural-ask/canonical-ask.parquet")
    p.add_argument("--chunk", type=int)
    p.add_argument("--parts", default="parts")
    p.add_argument("--output", default="training-cache/granite")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--assemble", action="store_true")
    args=p.parse_args()
    if args.smoke:
        smoke()
    elif args.assemble:
        assemble(args)
    elif args.chunk is not None and 0 <= args.chunk < CHUNKS:
        chunk(args)
    else:
        p.error("Provide --smoke, --assemble or --chunk 0..9")


if __name__=="__main__":
    main()
