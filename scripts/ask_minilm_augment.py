"""Frozen MiniLM + true two-out-of-bag price feature for all 18 asking neural arms."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from pricing.minilm_signal import fit_head, price as predict_head

REV = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def aligned_vectors(ids, root):
    manifest = json.loads((root / "provenance.json").read_text())
    if (manifest.get("revision") != REV or
        manifest.get("source") != "sentence-transformers/all-MiniLM-L6-v2" or
        not manifest.get("frozen") or manifest.get("supervised_backbone_training")):
        raise ValueError("Wrong or supervised-finetuned MiniLM backbone")
    reference = pd.Index(json.loads((root / "row-ids.json").read_text()))
    if not reference.is_unique:
        raise ValueError("Duplicate MiniLM embedding identities")
    positions = reference.get_indexer(pd.Index(ids.astype(str)))
    if (positions < 0).any():
        raise ValueError("MiniLM embeddings missing canonical row IDs")
    matrix = np.load(root / "embeddings.npy", mmap_mode="r")
    if matrix.shape != (len(reference), 384):
        raise ValueError("Expected original 384-dimensional MiniLM embeddings")
    vectors = np.asarray(matrix[positions], dtype=np.float32)
    if not np.isfinite(vectors).all():
        raise ValueError("Nonfinite pretrained embeddings")
    return vectors, manifest


def two_oob(vectors, target, fit, seed=42):
    """Three one-fold-trained Ridge heads, TWO genuine OOB predictions per FIT row."""
    fit = np.asarray(fit, dtype=int)
    if len(fit) < 9 or len(np.unique(fit)) != len(fit):
        raise ValueError("Invalid FIT group")
    if not np.isfinite(target[fit]).all() or (target[fit] <= 0).any():
        raise ValueError("Invalid FIT target")
    fold = np.full(len(target), -1, dtype=np.int8)
    outputs, heads = [], []
    for i, (_, held_offsets) in enumerate(KFold(n_splits=3, shuffle=True, random_state=seed).split(fit)):
        single_fold_rows = fit[held_offsets]
        fold[single_fold_rows] = i
        head = fit_head(vectors[single_fold_rows], target[single_fold_rows])
        heads.append(head)
        outputs.append(predict_head(head, vectors).astype(np.float64))
    all_three = np.stack(outputs)
    signal = all_three.mean(axis=0)
    for j in range(3):
        own = fit[fold[fit] == j]
        other = [i for i in range(3) if i != j]
        signal[own] = (all_three[other[0], own] + all_three[other[1], own]) / 2
    if not np.isfinite(signal).all() or (signal <= 0).any():
        raise ValueError("Invalid OOB MiniLM signal")
    return signal, fold, heads


def encode_fallback(parquet, output, encoder):
    """Only if previous cached embeddings disappeared: pretrained ONNX inference, no fit."""
    from pricing.minilm_signal import Encoder
    from scripts.common import clean
    frame = pd.read_parquet(parquet, columns=["row_id", "text"])
    if len(frame) != 211127 or frame.row_id.duplicated().any():
        raise ValueError("Unexpected original asking cohort in fallback")
    e = Encoder(encoder)
    blocks = []
    for i in range(0, len(frame), 1024):
        blocks.append(e.encode([clean(t) for t in frame.text.iloc[i:i+1024].fillna("").astype(str)]))
        print("MiniLM frozen encoding", min(i+1024, len(frame)), "/", len(frame), flush=True)
    vec = np.concatenate(blocks).astype(np.float32)
    if vec.shape != (211127, 384):
        raise ValueError("Wrong pretrained MiniLM embedding shape")
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "embeddings.npy", vec)
    (output / "row-ids.json").write_text(json.dumps(frame.row_id.astype(str).tolist()))
    manifest = json.loads((encoder / "provenance.json").read_text())
    manifest.update(frozen=True, supervised_backbone_training=False,
                    reconstructed_from_pinned_onnx=True, original_rows=len(frame))
    (output / "provenance.json").write_text(json.dumps(manifest, indent=2))


def prepare(features, embeddings, output):
    manifest = json.loads(features.with_suffix(".json").read_text())
    if (manifest["full_train"], manifest["lemon_train"], manifest["validation"], manifest["test"]) != (147777,110832,31668,31682):
        raise ValueError("Original archived asking-data cohort mismatch")
    with np.load(features, allow_pickle=False) as src:
        d = {k:src[k] for k in ("num","cat","txt","y","phase","ids","lemon")}
    ids, phase, lemon = d["ids"].astype(str), d["phase"].astype(str), d["lemon"].astype(bool)
    for name in ("validation", "test"):
        digest = hashlib.sha256("\n".join(sorted(ids[phase == name])).encode()).hexdigest()
        if digest != manifest["holdout_row_sha256"][name]:
            raise ValueError("Published heldout row fingerprints changed")
    x, info = aligned_vectors(ids, embeddings)
    for cohort in ("full", "lemon"):
        train = np.flatnonzero((phase == "train") & (lemon if cohort == "lemon" else True))
        stop = max(512, int(len(train)*.1))
        fit, early = train[:-stop], train[-stop:]
        signal, folds, heads = two_oob(x, d["y"].astype(float), fit)
        log_signal = np.log(signal)
        med = float(np.median(log_signal[fit]))
        p25,p75 = np.percentile(log_signal[fit],[25,75])
        iqr = max(float(p75-p25),.1)
        added = np.clip((log_signal-med)/iqr,-8,8).astype(np.float32)
        num = np.column_stack((d["num"], added)).astype(np.float32)
        directory = output/cohort
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory/"features.npz",num=num,cat=d["cat"],txt=d["txt"],
                            y=d["y"],phase=d["phase"],ids=d["ids"],lemon=d["lemon"])
        record = dict(manifest)
        record.update(variant="minilm-2-oob-price",market="ask",num_features=num.shape[1],
                      minilm_encoder_revision=REV,minilm_embeddings_dimensions=384,
                      minilm_head="3 Ridge(alpha=100) price heads fitted to disjoint FIT-only folds",
                      minilm_fit_feature="mean of two heads untrained on its FIT fold",
                      minilm_heldout_feature="mean of 3 FIT-only heads",
                      minilm_internal_stopper_labels_used=False,
                      minilm_validation_test_labels_used=False,
                      minilm_oob_sha256=hashlib.sha256(signal.tobytes()).hexdigest(),
                      minilm_fit_rows=len(fit),minilm_stopper_rows=len(early),
                      minilm_original_text_svd_retained=True,
                      minilm_price_log_median=med,minilm_price_log_iqr=iqr)
        (directory/"features.json").write_text(json.dumps(record,indent=2))
        (directory/"heads.json").write_text(json.dumps({"heads":heads,"policy":record["minilm_fit_feature"]}))
        pd.DataFrame({"row_id":ids,"phase":phase,"minilm_price_usd":signal,"fit_fold":folds}).to_csv(
            directory/"oob-audit.csv.gz",index=False)
        print(json.dumps({"cohort":cohort,"fit":len(fit),"early":len(early),
                          "oob_fit_rows":int((folds>=0).sum()),
                          "validation":int(sum(phase=="validation")),
                          "test":int(sum(phase=="test"))}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--features",type=Path,default=Path("training-cache/neural-ask/features.npz"))
    p.add_argument("--embeddings",type=Path,default=Path("training-cache/minilm-embeddings"))
    p.add_argument("--output",type=Path,default=Path("training-cache/ask-minilm"))
    p.add_argument("--encode-fallback",action="store_true")
    p.add_argument("--parquet",type=Path,default=Path("training-cache/neural-ask/canonical-ask.parquet"))
    p.add_argument("--encoder",type=Path,default=Path("models/xgboost-sparse-minilm/ask/minilm"))
    a=p.parse_args()
    if a.encode_fallback: encode_fallback(a.parquet,a.embeddings,a.encoder)
    else:prepare(a.features,a.embeddings,a.output)
