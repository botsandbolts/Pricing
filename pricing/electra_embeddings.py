"""Frozen ELECTRA feature pooling and strict 25-shard identity checks.

No torch dependency here: the same pooling and shard checks are unit-tested on CPU.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

MODEL_ID = "google/electra-small-discriminator"
SHARDS = 25
DIMENSIONS = 256
MAX_TOKENS = 512
POOLING = "attention-masked mean of last hidden state; L2 normalization"


def masked_mean(hidden, mask):
    hidden = np.asarray(hidden, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2]:
        raise ValueError("Hidden-state and attention-mask shapes differ")
    if not np.isfinite(hidden).all() or not np.isin(mask, [0, 1]).all():
        raise ValueError("Invalid hidden states or attention mask")
    counts = mask.sum(axis=1, keepdims=True)
    if (counts <= 0).any():
        raise ValueError("Empty attention mask")
    values = (hidden * mask[..., None]).sum(axis=1) / counts
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("Invalid pooled vector")
    return (values / norms).astype(np.float32)


def shard_positions(offset, length, shard, total=SHARDS):
    if not 0 <= shard < total or offset < 0 or length < 0:
        raise ValueError("Invalid shard or row interval")
    return (np.arange(offset, offset + length) % total) == shard


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def merge_shards(root, expected):
    root = Path(root)
    ids, vectors = [], []
    scanned_counts = set()
    receipts = []
    if expected.get("shards") != SHARDS or expected.get("model") != MODEL_ID:
        raise ValueError("Unexpected experiment manifest")
    for shard in range(SHARDS):
        receipt = json.loads((root / f"{shard:02d}.json").read_text())
        for key in ("model", "revision", "source_sha256", "pooling", "dimensions", "max_tokens", "shards"):
            if receipt.get(key) != expected.get(key):
                raise ValueError(f"Shard {shard}: mismatched {key}")
        if receipt["shard"] != shard or not receipt.get("frozen"):
            raise ValueError("Wrong shard identity or trainable encoder")
        scanned = receipt["source_rows_scanned"]
        scanned_counts.add(scanned)
        with np.load(root / f"{shard:02d}.npz", allow_pickle=False) as part:
            row_ids = part["ids"].astype(str)
            values = part["embeddings"]
        count = len(range(shard, scanned, SHARDS))
        if len(row_ids) != count or receipt["rows"] != count:
            raise ValueError("Shard row count differs from source partition")
        if values.shape != (count, DIMENSIONS) or not np.isfinite(values).all():
            raise ValueError("Invalid ELECTRA vector matrix")
        if not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-4):
            raise ValueError("ELECTRA vectors are not normalized")
        ids.append(row_ids)
        vectors.append(values)
        receipts.append(receipt)
    if len(scanned_counts) != 1:
        raise ValueError("Different scanned source lengths")
    ids = np.concatenate(ids)
    if len(set(ids.tolist())) != len(ids) or len(ids) != next(iter(scanned_counts)):
        raise ValueError("Duplicate or missing source identities")
    return ids, np.concatenate(vectors), receipts
