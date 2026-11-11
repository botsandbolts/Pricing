"""Historical U.S. Bid feature refresh: train-only average mileage by make/model.

The observed auction sale remains the label and the observed historical
odometer remains the odometer. Neither 2026 mileage nor 2026 bid prices
can be inferred from 2014/15 auction records.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED = {"train": 384198, "validation": 82380, "test": 83603}
from scripts.neural_bid_prepare import sha_ids

EXTRA = (
    "expected_annual_miles_make_model_train_only",
    "odometer_to_expected_mileage_ratio_at_auction",
    "signed_excess_mileage_vs_make_model_at_auction",
)

def _normalize(values):
    return (values.fillna("unknown").astype(str).str.lower()
            .str.strip().replace({"": "unknown", "nan": "unknown"}))

def add_model_mileage(frame: pd.DataFrame, phase: np.ndarray, existing: np.ndarray):
    """Return 8 numerical columns; no price, validation, or test labels are used."""
    required = {"row_id", "year", "date", "odometer", "manufacturer", "model"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing original pre-sale facts: {required-set(frame.columns)}")
    dates = pd.to_datetime(frame["date"], utc=True, errors="coerce", format="mixed")
    model_year = pd.to_numeric(frame["year"], errors="coerce").to_numpy(float)
    age = (dates.dt.year.to_numpy(float) +
           (dates.dt.dayofyear.to_numpy(float)-1)/365.25 - model_year)
    miles = pd.to_numeric(frame["odometer"], errors="coerce").to_numpy(float)
    annual = miles / np.maximum(age, 0.5)
    train = np.asarray(phase, dtype=str) == "train"
    eligible = (train & np.isfinite(annual) & np.isfinite(age) &
                (age >= 1) & (age <= 30) &
                (annual >= 1000) & (annual <= 50000) &
                np.isfinite(miles) & (miles >= 0) & (miles <= 800000))
    if int(eligible.sum()) < 10000:
        raise ValueError("Insufficient original TRAIN mileage records")
    make = _normalize(frame["manufacturer"])
    model = _normalize(frame["model"])
    rates = pd.DataFrame({"make": make, "model": model, "annual": annual})
    clean = rates.loc[eligible]
    grouped = clean.groupby(["make", "model"])["annual"].agg(["median", "size"])
    model_rate = grouped["median"].where(grouped["size"] >= 100)
    ix = pd.MultiIndex.from_arrays([make, model])
    per_model = model_rate.reindex(ix).to_numpy(dtype=float)
    by_make = clean.groupby("make")["annual"].agg(["median", "size"])
    make_rate = by_make["median"].where(by_make["size"] >= 300)
    per_make = make.map(make_rate).to_numpy(dtype=float)
    fallback = float(np.median(annual[eligible]))
    expected_annual = np.where(np.isfinite(per_model), per_model,
                               np.where(np.isfinite(per_make), per_make, fallback))
    expected_miles = expected_annual * np.maximum(age, 0.5)
    valid = np.isfinite(miles) & (miles >= 0) & np.isfinite(age) & (age >= 0)
    additional = np.stack([
        np.log1p(expected_annual),
        np.log1p(np.maximum(miles, 0) / np.maximum(expected_miles, 1000)),
        np.sign(miles-expected_miles) *
            np.log1p(np.abs(miles-expected_miles)/10000),
    ], axis=1)
    additional[~valid, 1:] = np.nan
    med = np.nanmedian(additional[train], axis=0)
    med[~np.isfinite(med)] = 0
    additional = np.where(np.isfinite(additional), additional, med)
    q25, q75 = np.percentile(additional[train], [25, 75], axis=0)
    scale = np.maximum(q75-q25, 0.05)
    scaled = np.clip((additional-med)/scale, -8, 8).astype(np.float32)
    if existing.shape != (len(frame), 5):
        raise ValueError("Original 5 numeric auction inputs changed unexpectedly")
    result = np.column_stack([existing.astype(np.float32), scaled])
    if result.shape != (len(frame), 8) or not np.isfinite(result).all():
        raise ValueError("Nonfinite expanded model inputs")
    audit = {
        "training_records_for_expected_mileage": int(eligible.sum()),
        "overall_train_median_annual_miles": round(fallback, 3),
        "make_model_min_fit_count": 100,
        "manufacturer_min_fit_count": 300,
        "make_model_fallback_rows": int(np.isnan(per_model).sum()),
        "manufacturer_fallback_rows": int((np.isnan(per_model)&np.isnan(per_make)).sum()),
        "extra_input_names": list(EXTRA),
        "extra_median_train_only": med.tolist(),
        "extra_iqr_train_only": scale.tolist(),
        "observed_mileage_is_unchanged": True,
        "no_hypothetical_nov2026_odometer": True,
        "no_2026_price_targets": True,
        "training_only_model_mileage_profiles": True,
    }
    return result, audit

def prepare(source: Path, output: Path):
    source, output = Path(source), Path(output)
    meta = json.loads(source.with_suffix(".json").read_text())
    if (meta.get("market") != "bid" or meta.get("full_train") != EXPECTED["train"] or
        meta.get("no_price_derived_inputs") is not True):
        raise ValueError("Expected authentic original U.S. wholesale Bid cohort")
    parquet = source.with_name("canonical-bid.parquet")
    frame = pd.read_parquet(parquet)
    with np.load(source, allow_pickle=False) as archive:
        z = {k: archive[k] for k in ("num", "cat", "y", "ids", "phase", "lemon")}
    if len(frame) != len(z["y"]) or len(frame) != 550181:
        raise ValueError("Canonical row count changed")
    if not np.array_equal(frame["row_id"].astype(str).to_numpy(),
                          z["ids"].astype(str)):
        raise ValueError("Original auction row alignment changed")
    phase = z["phase"].astype(str)
    for split, count in EXPECTED.items():
        if (phase == split).sum() != count:
            raise ValueError("Original split membership changed: " + split)
    for split in ("validation", "test"):
        if sha_ids(z["ids"][phase == split]) != meta["holdout_row_sha256"][split]:
            raise ValueError("Frozen VIN-disjoint holdout changed: " + split)
    enhanced, audit = add_model_mileage(frame, phase, z["num"])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, num=enhanced, cat=z["cat"], y=z["y"],
                        phase=z["phase"], ids=z["ids"], lemon=z["lemon"])
    shutil.copy2(parquet, output.with_name("canonical-bid.parquet"))
    meta = dict(meta)
    meta.update({
        "numeric_features": list(meta["numeric_features"])+list(EXTRA),
        "num_features": 8,
        "model_specific_mileage_expected_train_only": True,
        "mileage_expected_train_only": True,
        "temporal_basis": "observed historical 2014/15 auction date",
        "target_currency_basis": "original 2015-index-normalized USD; not current dollars",
        "dataset_is_not_current": True,
        "added_mileage_feature_audit": audit,
    })
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2)+"\n")
    output.with_name("mileage_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print("US_BID_MILEAGE_PREPARED", json.dumps(audit), flush=True)

def smoke():
    n = 12000
    indexes = np.arange(n)
    make_model = np.where(indexes % 2 == 0, "Focus", "Fiesta")
    age = 5
    odometer = np.where(indexes % 2 == 0, 50000., 150000.)
    frame = pd.DataFrame({
        "row_id": indexes.astype(str), "year": np.full(n, 2010),
        "date": ["2015-01-01"] * n,
        "odometer": odometer, "manufacturer": ["Ford"] * n,
        "model": make_model, "observed_price": np.full(n, 5000.),
    })
    phase = np.where(indexes < 10500, "train", "validation")
    existing = np.zeros((n, 5), dtype=np.float32)
    original, audit = add_model_mileage(frame, phase, existing)
    assert original.shape == (n, 8)
    assert audit["training_only_model_mileage_profiles"]
    assert original[0, 5] != original[1, 5], "Model-specific mileage was ignored"
    changed = frame.copy()
    changed.loc[phase == "validation", "odometer"] = 800000.
    changed["observed_price"] *= 5
    repeated, audit2 = add_model_mileage(changed, phase, existing)
    np.testing.assert_array_equal(original[phase == "train"],
                                  repeated[phase == "train"])
    assert audit["overall_train_median_annual_miles"] == audit2["overall_train_median_annual_miles"]
    print("US_BID_MILEAGE_SMOKE_OK", flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--source", type=Path, default=Path("training-cache/neural-bid/features.npz"))
    p.add_argument("--output", type=Path, default=Path("training-cache/us-bid-ray-current/features.npz"))
    args = p.parse_args()
    smoke() if args.smoke else prepare(args.source, args.output)
