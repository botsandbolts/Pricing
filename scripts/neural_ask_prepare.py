"""Build one auditable ask-only neural benchmark from the ORIGINAL held-out cohort.

No hierarchy, no price-derived features, no filtering of validation/test. The
Lemon treatment is the frozen 25%-trimmed TRAINING membership from the audited
XGBoost release, matched by exact row_id. All preprocessing fits on full TRAIN
only, never on original validation/test; training text is represented without
using its price target (TF-IDF + truncated SVD).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

CATS = ("manufacturer", "model", "type", "transmission", "state",
        "fuel", "drive", "title_status", "cylinders", "paint_color", "condition")
TOP = {"manufacturer": 150, "model": 1600, "type": 45, "transmission": 24,
       "state": 75, "fuel": 28, "drive": 24, "title_status": 24,
       "cylinders": 30, "paint_color": 45, "condition": 60}
NUMS = ("year", "age_2021", "log_miles", "miles_per_year",
        "log_description_chars", "log_description_words", "description_punctuation")


def phase_from_source(frame):
    if frame.row_id.astype(str).duplicated().any():
        raise ValueError("Duplicate canonical row IDs")
    if set(frame.phase) != {"train", "validation", "test"}:
        raise ValueError("Original train/validation/test phases are required")
    # The canonical preparation is chronological, with train then validation then test.
    if frame.phase.ne(frame.phase.shift()).sum() != 3:
        raise ValueError("Non-contiguous original chronological partitions")
    return frame.phase.astype(str).to_numpy()


def lemon_mask(frame, lemon_csv):
    """An identity-based train-only trim; NEVER trim validation/test."""
    evidence = pd.read_csv(lemon_csv, usecols=["row_id", "split"],
                           dtype={"row_id": str, "split": str})
    if evidence.row_id.duplicated().any():
        raise ValueError("Duplicate frozen Lemon identities")
    target = frame[["row_id", "phase"]].copy()
    target.row_id = target.row_id.astype(str)
    mapping = {"calibration": "train", "validation": "validation", "test": "test"}
    evidence["phase"] = evidence.split.map(mapping)
    if evidence.phase.isna().any():
        raise ValueError("Unexpected Lemon split name")
    merged = evidence.merge(target, on="row_id", how="left", validate="one_to_one",
                            suffixes=("_frozen", "_canonical"), indicator=True)
    if not merged._merge.eq("both").all() or not merged.phase_frozen.eq(merged.phase_canonical).all():
        raise ValueError("Frozen Lemon members are not identical canonical rows/splits")
    allowed = set(evidence.row_id)
    holdout = set(target.loc[target.phase.ne("train"), "row_id"])
    if not holdout.issubset(allowed) or len(allowed & holdout) != len(holdout):
        raise ValueError("The Lemon experiment must retain ALL original holdouts")
    keep = target.row_id.isin(allowed).to_numpy()
    if keep[target.phase.ne("train").to_numpy()].all() is not True:
        raise ValueError("Attempted heldout filtering")
    return keep


def numeric_features(frame):
    year = pd.to_numeric(frame.year, errors="coerce").to_numpy(float)
    miles = pd.to_numeric(frame.odometer, errors="coerce").to_numpy(float)
    safe_miles = np.where(np.isfinite(miles), np.maximum(miles, 0), np.nan)
    age = np.where(np.isfinite(year), np.maximum(2021 - year, 0), np.nan)
    descriptions = frame.text.fillna("").astype(str)
    count = descriptions.str.len().to_numpy(float)
    words = descriptions.str.count(r"\S+").to_numpy(float)
    punctuation = descriptions.str.count(r"[!?]").to_numpy(float)
    return np.column_stack((year, age, np.log1p(safe_miles),
                            safe_miles / np.maximum(age, 1), np.log1p(count),
                            np.log1p(words), np.log1p(punctuation)))


def encode_features(frame, train, svd_components=24, atlas=30000, seed=42):
    numeric = numeric_features(frame)
    med = np.nanmedian(numeric[train], axis=0)
    med[~np.isfinite(med)] = 0
    numeric = np.where(np.isfinite(numeric), numeric, med)
    q25, q75 = np.percentile(numeric[train], [25, 75], axis=0)
    scale = np.maximum(q75 - q25, 1)
    numeric = np.clip((numeric - med) / scale, -8, 8).astype(np.float32)
    cats = np.zeros((len(frame), len(CATS)), dtype=np.int32)
    sizes, dictionaries = [], {}
    for col_i, name in enumerate(CATS):
        values = (frame[name] if name in frame else pd.Series("", index=frame.index))
        values = values.fillna("unknown").astype(str).str.lower().str.strip()
        counts = values.iloc[train].value_counts()
        retained = [v for v in counts.head(TOP[name]).index
                    if counts[v] >= 10 and v and v != "unknown"]
        mapping = {value: ix + 2 for ix, value in enumerate(retained)}
        cats[:, col_i] = values.map(mapping).fillna(1).to_numpy(dtype=np.int32)
        sizes.append(len(mapping) + 2)
        dictionaries[name] = {"levels": len(mapping), "unknown": 1, "padding": 0}

    descriptions = frame.text.fillna("").astype(str).tolist()
    rng = np.random.default_rng(seed)
    fitted = rng.choice(np.flatnonzero(train), min(atlas, sum(train)), replace=False)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=6, max_df=.97,
                                 max_features=50000, sublinear_tf=True,
                                 dtype=np.float32, strip_accents="unicode")
    atlas_matrix = vectorizer.fit_transform([descriptions[i] for i in fitted])
    dimensions = min(svd_components, min(atlas_matrix.shape) - 1)
    if dimensions < 2:
        raise ValueError("Not enough train-only text vocabulary for unsupervised SVD")
    reducer = TruncatedSVD(n_components=dimensions, n_iter=5, random_state=seed)
    reducer.fit(atlas_matrix)
    text = np.zeros((len(frame), dimensions), dtype=np.float32)
    for start in range(0, len(frame), 4096):
        batch = vectorizer.transform(descriptions[start:start+4096])
        text[start:start+4096] = reducer.transform(batch)
    tmean = text[train].mean(axis=0)
    tstd = np.maximum(text[train].std(axis=0), .02)
    text = np.clip((text - tmean) / tstd, -8, 8).astype(np.float32)
    return numeric, cats, text, sizes, dictionaries


def prepare(cohort, lemon_csv, output):
    frame = pd.concat(pd.read_json(cohort, lines=True, chunksize=10000), ignore_index=True)
    phase = phase_from_source(frame)
    price = pd.to_numeric(frame.price, errors="coerce").to_numpy(float)
    if not np.isfinite(price).all() or (price <= 0).any():
        raise ValueError("Invalid canonical asking prices")
    train = phase == "train"
    keep = lemon_mask(frame, lemon_csv)
    if int(train.sum()) != 147777 or sum(phase == "validation") != 31668 or sum(phase == "test") != 31682:
        raise ValueError("Canonical cohort changed; do not compare unmatched tests")
    if int(np.sum(keep & train)) != 110832:
        raise ValueError("Frozen Lemon training membership changed")
    numeric, categories, text, card, dictionaries = encode_features(frame, train)
    ids = frame.row_id.astype(str).to_numpy(dtype="U80")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, num=numeric, cat=categories, txt=text,
                        y=price.astype(np.float32), phase=np.asarray(phase, dtype="U10"),
                        ids=ids, lemon=keep.astype(np.bool_))
    hashes = {}
    for name in ("validation", "test"):
        rows = np.flatnonzero(phase == name)
        hashes[name] = hashlib.sha256("\n".join(sorted(ids[rows])).encode()).hexdigest()
    manifest = {"dataset": "Craigslist 2021 original chronological asking cohort",
                "cohort_sha256": hashlib.sha256(Path(cohort).read_bytes()).hexdigest(),
                "lemon_source": "matched-bayesian-redesign-data/ask-minilm-cleaned.csv.gz",
                "cohort_rows": len(frame), "full_train": int(train.sum()),
                "lemon_train": int(np.sum(train & keep)),
                "validation": int(sum(phase == "validation")),
                "test": int(sum(phase == "test")), "holdout_row_sha256": hashes,
                "text": "Train-only unsupervised TF-IDF 1-2grams -> 24-component SVD (30k train atlas)",
                "no_hierarchy": True, "price_derived_inputs": False,
                "numerics": list(NUMS), "categories": dictionaries,
                "cardinalities": card, "num_features": numeric.shape[1],
                "text_features": text.shape[1], "split_policy": "Original chronological 70/15/15; immutable",
                "outlier_policy": "All canonical full-train rows vs frozen train-only 25% Lemon exclusion; full original holdouts"}
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("categories",)}, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", default="training-cache/full-cohort.jsonl")
    p.add_argument("--lemon-csv", default="evidence/ask-minilm-cleaned.csv.gz")
    p.add_argument("--output", default="training-cache/neural-ask/features.npz")
    args = p.parse_args()
    prepare(args.cohort, args.lemon_csv, args.output)
