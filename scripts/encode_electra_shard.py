"""Pin and smoke-test ELECTRA, then encode 25 disjoint source partitions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pricing.advert_text import scrub
from pricing.electra_embeddings import (
    MODEL_ID, SHARDS, DIMENSIONS, MAX_TOKENS, POOLING,
    masked_mean, shard_positions, file_sha256,
)

SOURCE = Path("training-cache/full-cohort.jsonl")
MANIFEST = Path("training-cache/electra-manifest.json")


def load_encoder(revision):
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
    model = AutoModel.from_pretrained(MODEL_ID, revision=revision).to("cpu").eval()
    return tokenizer, model


def encode_texts(texts, tokenizer, model, batch_size=32):
    import torch
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start:start + batch_size], padding=True, truncation=True,
                max_length=MAX_TOKENS, return_tensors="pt",
            )
            hidden = model(**batch).last_hidden_state.float().cpu().numpy()
            vectors.append(masked_mean(hidden, batch["attention_mask"].cpu().numpy()))
    return np.concatenate(vectors) if vectors else np.empty((0, DIMENSIONS), dtype=np.float32)


def prepare():
    from huggingface_hub import HfApi
    expected = json.loads(Path(
        "models/xgboost-sparse-minilm/ask/minilm/provenance.json"
    ).read_text())["cohort_sha256"]
    actual = file_sha256(SOURCE)
    if actual != expected:
        raise ValueError("Saved source cohort checksum mismatch")
    revision = HfApi().model_info(MODEL_ID).sha
    if not revision or len(revision) != 40:
        raise ValueError("No immutable ELECTRA revision resolved")
    tokenizer, model = load_encoder(revision)
    if model.config.hidden_size != DIMENSIONS:
        raise ValueError("Unexpected ELECTRA hidden size")
    sentence = "A red sedan with leather seats."
    alone = encode_texts([sentence], tokenizer, model)
    padded = encode_texts([sentence, "A longer car description. " * 25], tokenizer, model)
    if alone.shape != (1, DIMENSIONS) or not np.allclose(alone[0], padded[0], atol=2e-5):
        raise ValueError("Padding-invariance smoke test failed")
    manifest = {
        "model": MODEL_ID, "revision": revision, "source_sha256": actual,
        "pooling": POOLING, "dimensions": DIMENSIONS, "max_tokens": MAX_TOKENS,
        "shards": SHARDS, "frozen": True,
        "encoder_parameters": sum(p.numel() for p in model.parameters()),
        "smoke_test": "real pretrained encoder forward; padding invariance passed",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


def run(shard):
    shard_positions(0, 0, shard)  # fail before downloads if invalid
    manifest = json.loads(MANIFEST.read_text())
    if file_sha256(SOURCE) != manifest["source_sha256"]:
        raise ValueError("Source changed after preparation")
    tokenizer, model = load_encoder(manifest["revision"])
    ids, vectors = [], []
    scanned = changed = 0
    for chunk in pd.read_json(SOURCE, lines=True, chunksize=1000, dtype={"row_id": str}):
        selected = chunk.loc[shard_positions(scanned, len(chunk), shard)]
        scanned += len(chunk)
        if selected.empty:
            continue
        if selected.row_id.isna().any():
            raise ValueError("Missing source row identity")
        raw = selected.text.fillna("").astype(str).tolist()
        texts = [scrub(t) for t in raw]
        changed += sum(a != b for a, b in zip(raw, texts))
        vectors.append(encode_texts(texts, tokenizer, model))
        ids.extend(selected.row_id.astype(str).tolist())
        print(f"shard {shard:02d}: {len(ids)} descriptions encoded", flush=True)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Empty shard or duplicate source row IDs")
    out = Path("electra-shards")
    out.mkdir(exist_ok=True)
    np.savez_compressed(
        out / f"{shard:02d}.npz", ids=np.asarray(ids),
        embeddings=np.concatenate(vectors).astype(np.float32),
    )
    (out / f"{shard:02d}.json").write_text(json.dumps({
        **manifest, "shard": shard, "rows": len(ids),
        "source_rows_scanned": scanned, "money_redacted_descriptions": changed,
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--shard", type=int)
    args = parser.parse_args()
    prepare() if args.prepare else run(args.shard)
