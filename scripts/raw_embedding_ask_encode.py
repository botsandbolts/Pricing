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



CHUNKS = 10


def boundaries(total, chunk_index, chunks=CHUNKS):
    """Contiguous stable partition, no omitted/duplicated original rows."""
    if total <= 0 or chunks != CHUNKS or not 0 <= chunk_index < chunks:
        raise ValueError("Expected one of ten valid original-row chunks")
    return total * chunk_index // chunks, total * (chunk_index + 1) // chunks


def run_chunk(name, parquet, features, output, chunk_index):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame, ids, provenance = canonical(parquet, features)
    start, end = boundaries(len(ids), chunk_index)
    encoder, revision = load_frozen(name)
    documents = frame.text.iloc[start:end].fillna("").astype(str).tolist()
    rng = np.random.default_rng(20261106 + chunk_index)
    sampled = rng.choice(len(documents), size=min(24, len(documents)), replace=False)
    check = consistency(encoder, [clean(documents[i]) for i in sampled])
    expected_dim = SPECS[name][2]
    matrix = np.lib.format.open_memmap(
        output / "embeddings.npy", mode="w+",
        dtype=np.float16, shape=(len(documents), expected_dim))
    for offset in range(0, len(documents), 512):
        block = [clean(text) for text in documents[offset:offset + 512]]
        vec = embed(encoder, block, batch=16)
        if vec.shape != (len(block), expected_dim):
            raise ValueError("Encoder embedding shape changed during inference")
        matrix[offset:offset + len(block)] = vec.astype(np.float16)
        if (offset // 512) % 8 == 0:
            print(json.dumps({"candidate": name, "chunk": chunk_index,
                              "rows_done": min(offset + len(block), len(documents)),
                              "chunk_rows": len(documents)}), flush=True)
    matrix.flush()
    del matrix
    with open(output / "embeddings.npy", "rb") as file:
        fingerprint = hashlib.file_digest(file, "sha256").hexdigest()
    meta = {
        "status": "complete_chunk", "market": "ask",
        "variant": name, "model_id": SPECS[name][0],
        "model_revision": revision, "max_tokens": SPECS[name][1],
        "dimensionality": expected_dim, "normalization": "L2",
        "saved_precision": "float16", "training_precision": "float32",
        "encoder_framework": "sentence-transformers",
        "price_derived_features": False, "text_price_heads": 0,
        "supervised_encoder_finetuning": False,
        "price_mentions": "redacted using scripts.common.clean",
        "input_source": "Original immutable 2021 Craigslist asking descriptions",
        "cohort_rows": len(ids), "chunk_index": chunk_index,
        "chunk_count": CHUNKS, "start": start, "end": end,
        "row_count": end - start,
        "chunk_row_sha256": id_hash(ids[start:end]),
        "canonical_cohort_sha256": provenance["cohort_sha256"],
        "holdout_row_sha256": provenance["holdout_row_sha256"],
        "batch_consistency": check,
        "embedding_file_sha256": fingerprint,
        "published_test_labels_used": False,
    }
    (output / "chunk.json").write_text(json.dumps(meta, indent=2))
    print("PRICE-FREE EMBEDDING CHUNK COMPLETED", json.dumps(meta), flush=True)


def assemble(name, features, parts, output):
    """Merge ten exact-ID chunks without using labels, embeddings-by-price, or inference."""
    features, parts, output = Path(features), Path(parts), Path(output)
    source = json.loads(features.with_suffix(".json").read_text())
    with np.load(features, allow_pickle=False) as original:
        ids = original["ids"].astype(str)
        phase = original["phase"].astype(str)
    if len(ids) != 211127 or sum(phase == "validation") != 31668 or sum(phase == "test") != 31682:
        raise ValueError("The original chronological holdouts were changed")
    output.mkdir(parents=True, exist_ok=True)
    dim = SPECS[name][2]
    whole = np.lib.format.open_memmap(
        output / "embeddings.npy", mode="w+",
        dtype=np.float16, shape=(len(ids), dim))
    revision = None
    audits = []
    for chunk_index in range(CHUNKS):
        folder = parts / f"raw-ask-embed-chunk-{name}-{chunk_index}"
        record = json.loads((folder / "chunk.json").read_text())
        begin, end = boundaries(len(ids), chunk_index)
        if (record.get("status"), record.get("variant"),
            record.get("chunk_index"), record.get("chunk_count"),
            record.get("start"), record.get("end"),
            record.get("row_count")) != (
                "complete_chunk", name, chunk_index, CHUNKS,
                begin, end, end-begin):
            raise ValueError(f"Wrong or duplicate embedding chunk {chunk_index}")
        if ((record.get("model_id"), record.get("max_tokens"),
             record.get("dimensionality")) != SPECS[name] or
            record.get("canonical_cohort_sha256") != source["cohort_sha256"] or
            record.get("holdout_row_sha256") != source["holdout_row_sha256"] or
            record.get("chunk_row_sha256") != id_hash(ids[begin:end])):
            raise ValueError(f"Original row identity/encoder mismatch in chunk {chunk_index}")
        if (record.get("text_price_heads") != 0 or
            record.get("price_derived_features") is not False or
            record.get("published_test_labels_used") is not False):
            raise ValueError("A chunk contains target-price features or leaked test labels")
        if revision is not None and record["model_revision"] != revision:
            raise ValueError("One encoder had multiple upstream model revisions across chunks")
        revision = record["model_revision"]
        check=record["batch_consistency"]
        if (check["max_absolute_difference"] > .005 or
            check["minimum_cosine_similarity"] < .999):
            raise ValueError(f"Batch-one serving parity failed for chunk {chunk_index}")
        filename = folder / "embeddings.npy"
        with open(filename, "rb") as file:
            fingerprint = hashlib.file_digest(file, "sha256").hexdigest()
        if fingerprint != record["embedding_file_sha256"]:
            raise ValueError(f"Checksum mismatch in embedding chunk {chunk_index}")
        piece = np.load(filename, mmap_mode="r")
        if piece.shape != (end-begin, dim) or piece.dtype != np.float16:
            raise ValueError(f"Wrong embedding matrix geometry in chunk {chunk_index}")
        if not np.isfinite(piece).all():
            raise ValueError(f"Nonfinite embeddings in chunk {chunk_index}")
        whole[begin:end] = piece
        audits.append(record)
        print("ASSEMBLED", name, chunk_index+1, "of", CHUNKS, flush=True)
    whole.flush()
    del whole
    with open(output / "embeddings.npy", "rb") as file:
        combined_hash = hashlib.file_digest(file, "sha256").hexdigest()
    meta = {
        "status":"complete", "market":"ask",
        "variant":name,"model_id":SPECS[name][0],
        "model_revision":revision,"max_tokens":SPECS[name][1],
        "dimensionality":dim, "normalization":"L2",
        "saved_precision":"float16","training_precision":"float32",
        "encoder_framework":"sentence-transformers",
        "price_derived_features":False,"text_price_heads":0,
        "supervised_encoder_finetuning":False,
        "price_mentions":"redacted using scripts.common.clean",
        "input_source":"Original immutable 2021 Craigslist asking descriptions",
        "row_count":len(ids),
        "canonical_cohort_sha256":source["cohort_sha256"],
        "holdout_row_sha256":source["holdout_row_sha256"],
        "batch_consistency":{
            "samples":sum(a["batch_consistency"]["samples"] for a in audits),
            "max_absolute_difference":max(a["batch_consistency"]["max_absolute_difference"] for a in audits),
            "minimum_cosine_similarity":min(a["batch_consistency"]["minimum_cosine_similarity"] for a in audits),
            "train_batch_size":16,"live_batch_size":1,
            "relative_to_price_target":"none"},
        "embedding_file_sha256":combined_hash,
        "chunk_count":CHUNKS,
        "chunk_sha256":[a["embedding_file_sha256"] for a in audits],
        "chunk_original_row_sha256":[a["chunk_row_sha256"] for a in audits],
        "published_test_labels_used":False,
    }
    (output/"manifest.json").write_text(json.dumps(meta,indent=2))
    print("ALL TEN CHUNKS VERIFIED AND REASSEMBLED",json.dumps(meta),flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", choices=tuple(SPECS), required=True)
    p.add_argument("--parquet", default="training-cache/neural-ask/canonical-ask.parquet")
    p.add_argument("--features", default="training-cache/neural-ask/features.npz")
    p.add_argument("--output", required=True)
    p.add_argument("--chunk-index",type=int,default=None)
    p.add_argument("--chunks",type=int,default=CHUNKS)
    p.add_argument("--assemble",action="store_true")
    p.add_argument("--parts",default="training-cache/raw-ask-chunks")
    args=p.parse_args()
    if args.chunks != CHUNKS:
        p.error("Reproducible experiment requires exactly ten chunks")
    if args.assemble:
        if args.chunk_index is not None:
            p.error("Assembly does not accept a chunk index")
        assemble(args.candidate,args.features,args.parts,args.output)
    elif args.chunk_index is not None:
        run_chunk(args.candidate,args.parquet,args.features,args.output,args.chunk_index)
    else:
        p.error("Specify either --chunk-index or --assemble")
