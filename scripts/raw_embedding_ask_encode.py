"""Frozen, price-free sentence embeddings for the ORIGINAL asking-price cohort.

No text-price heads. No labels enter encoder selection, tokenization, pooling,
or normalization. All 211127 immutable source rows are encoded (including
holdouts) by the same pinned pretrained model; no text-encoder fine-tuning.

A batch-1 vs batch-16 consistency test precedes artifact publication. Source
listing text is not uploaded to any third-party inference API.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common import clean

SPECS = {
    "minilm6-128": ("sentence-transformers/all-MiniLM-L6-v2", 128, 384),
    "minilm12-128": ("sentence-transformers/all-MiniLM-L12-v2", 128, 384),
    "bge-128": ("BAAI/bge-small-en-v1.5", 128, 384),
    "bge-256": ("BAAI/bge-small-en-v1.5", 256, 384),
    "gte-128": ("thenlper/gte-small", 128, 384),
    "gte-256": ("thenlper/gte-small", 256, 384),
    "distilroberta-128": ("sentence-transformers/all-distilroberta-v1", 128, 768),
    "distilroberta-256": ("sentence-transformers/all-distilroberta-v1", 256, 768),
}


def id_hash(ids):
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode("utf-8")).hexdigest()


def canonical(parquet, features):
    source = json.loads(Path(features).with_suffix(".json").read_text())
    if (source.get("full_train"), source.get("validation"), source.get("test")) != (147777, 31668, 31682):
        raise ValueError("Only the published original asking-price cohort is permitted")
    with np.load(features, allow_pickle=False) as original:
        ids = original["ids"].astype(str)
        phase = original["phase"].astype(str)
    frame = pd.read_parquet(parquet, columns=["row_id", "text"])
    if len(frame) != 211127 or frame.row_id.astype(str).duplicated().any():
        raise ValueError("Wrong source cohort or duplicated vehicle identities")
    if not np.array_equal(frame.row_id.astype(str).to_numpy(), ids):
        raise ValueError("Original source row order/identities changed")
    if (sum(phase == "train"), sum(phase == "validation"), sum(phase == "test")) != (147777, 31668, 31682):
        raise ValueError("Published chronological partition sizes differ")
    for split in ("validation", "test"):
        if id_hash(ids[phase == split]) != source["holdout_row_sha256"][split]:
            raise ValueError("Original " + split + " row fingerprint changed")
    return frame, ids, source


def load_frozen(name):
    import torch
    from huggingface_hub import model_info
    from sentence_transformers import SentenceTransformer

    if name not in SPECS:
        raise ValueError("Unregistered pretrained embedding candidate")
    model_id, max_length, expected_dim = SPECS[name]
    sha = model_info(model_id, revision="main").sha
    if not sha or len(sha) != 40:
        raise ValueError("Could not pin the exact Hugging Face model commit")
    torch.set_num_threads(2)
    encoder = SentenceTransformer(model_id, revision=sha,
                                  trust_remote_code=False, device="cpu")
    encoder.max_seq_length = max_length
    encoder.eval()
    dimension = encoder.get_sentence_embedding_dimension()
    if dimension != expected_dim:
        raise ValueError("Changed encoder dimension for " + name)
    return encoder, sha


def embed(encoder, texts, batch=16):
    result = encoder.encode(
        texts, batch_size=batch, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False)
    a = np.asarray(result, dtype=np.float32)
    if a.ndim != 2 or len(a) != len(texts) or not np.isfinite(a).all():
        raise ValueError("Invalid pretrained embeddings")
    if not np.allclose(np.linalg.norm(a, axis=1), 1., atol=0.003):
        raise ValueError("Expected L2-normalized frozen embeddings")
    return a


def consistency(encoder, texts):
    """Prove that one-at-a-time Streamlit inference reproduces training batches."""
    singles = np.concatenate([embed(encoder, [text], batch=1) for text in texts], 0)
    batched = embed(encoder, texts, batch=16)
    delta = np.max(np.abs(singles - batched), axis=1)
    dot = np.sum(singles * batched, axis=1)
    if delta.max() > .005 or dot.min() < .999:
        raise ValueError(
            f"Serving/training encoder batch parity failed: max abs={delta.max()}, min cos={dot.min()}")
    return {"samples": len(texts), "max_absolute_difference": float(delta.max()),
            "minimum_cosine_similarity": float(dot.min()),
            "train_batch_size": 16, "live_batch_size": 1,
            "relative_to_price_target": "none"}


def run(name, parquet, features, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame, ids, provenance = canonical(parquet, features)
    encoder, revision = load_frozen(name)
    documents = frame.text.fillna("").astype(str).tolist()
    rng = np.random.default_rng(20261106)
    sample = rng.choice(len(documents), size=24, replace=False)
    check = consistency(encoder, [clean(documents[i]) for i in sample])
    expected_dim = SPECS[name][2]
    matrix = np.lib.format.open_memmap(
        output / "embeddings.npy", mode="w+",
        dtype=np.float16, shape=(len(documents), expected_dim))
    for start in range(0, len(documents), 1024):
        block = [clean(text) for text in documents[start:start + 1024]]
        vec = embed(encoder, block, batch=16)
        if vec.shape[1] != expected_dim:
            raise ValueError("Encoder dimension changed during inference")
        matrix[start:start + len(block)] = vec.astype(np.float16)
        if (start // 1024) % 8 == 0:
            print(json.dumps({"candidate": name, "encoded": start + len(block),
                              "total": len(documents)}), flush=True)
    matrix.flush()
    del matrix
    with open(output / "embeddings.npy", "rb") as file:
        fingerprint = hashlib.file_digest(file, "sha256").hexdigest()
    meta = {
        "status": "complete", "market": "ask",
        "variant": name, "model_id": SPECS[name][0],
        "model_revision": revision, "max_tokens": SPECS[name][1],
        "dimensionality": expected_dim, "normalization": "L2",
        "saved_precision": "float16", "training_precision": "float32",
        "encoder_framework": "sentence-transformers",
        "price_derived_features": False, "text_price_heads": 0,
        "supervised_encoder_finetuning": False,
        "price_mentions": "redacted using scripts.common.clean",
        "input_source": "Original immutable 2021 Craigslist asking descriptions",
        "row_count": len(ids), "canonical_cohort_sha256": provenance["cohort_sha256"],
        "holdout_row_sha256": provenance["holdout_row_sha256"],
        "batch_consistency": check,
        "embedding_file_sha256": fingerprint,
        "published_test_labels_used": False,
    }
    (output / "manifest.json").write_text(json.dumps(meta, indent=2))
    print("FROZEN PRICE-FREE EMBEDDINGS COMPLETED", json.dumps(meta), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", choices=tuple(SPECS), required=True)
    p.add_argument("--parquet", default="training-cache/neural-ask/canonical-ask.parquet")
    p.add_argument("--features", default="training-cache/neural-ask/features.npz")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    run(args.candidate, args.parquet, args.features, args.output)
