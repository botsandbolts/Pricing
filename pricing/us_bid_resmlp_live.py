"""Verified 11 November 2026 U.S. historical wholesale ResMLP+Huber Bid.

Fail closed on absent/wrong weights or preprocessing. The 8 numerical inputs
include 3 observed-date mileage deviations fitted on TRAIN auctions only.
No guessed 2026 odometer values and no current-market claim.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import math

import numpy as np

FOLDER = "models/us-bid-resmlp-20261111"
FILES = ("weights.pt", "manifest.json", "verification.json", "mileage_profiles.json")
PREPROCESS = "models/neural-experimental-bid/preprocessing.json"
RUN_ID = 42998336606
EXPECTED = {"CVaR95_absolute_USD": 8719.286553909142,
            "MAE_USD": 1757.9273969371043,
            "RMSE_USD": 2869.77338296956}
COHORT_SHA = "16290e5e7b7a2913edb47d2b7e34ffa6d4922bdd4269b729f705a53a4c135045"

def _sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

def available(root):
    root = Path(root)
    folder = root / FOLDER
    if not all((folder / n).is_file() for n in FILES) or not (root / PREPROCESS).is_file():
        return False
    try:
        m = json.loads((folder / "manifest.json").read_text())
        v = json.loads((folder / "verification.json").read_text())
        p = json.loads((folder / "mileage_profiles.json").read_text())
        prep = json.loads((root / PREPROCESS).read_text())
        scores = m["original_validation"]
        return bool(
            m.get("status") == "verified_for_live_historical_bid"
            and m.get("source_workflow_run") == RUN_ID
            and m.get("source_shard") == 4
            and m.get("market") == "bid"
            and m.get("training_full_non_lemon") is True
            and m.get("test_scored") is False
            and m.get("cohort_sha256") == COHORT_SHA
            and m["configuration"]["family"] == "resmlp"
            and m["configuration"]["loss"] == "huber_usd"
            and m["checkpoint_sha256"] == _sha(folder / "weights.pt")
            and m["preprocessing_sha256"] == _sha(root / PREPROCESS)
            and m["mileage_profile_sha256"] == _sha(folder / "mileage_profiles.json")
            and all(math.isclose(float(scores[k]), value, abs_tol=.001)
                    for k, value in EXPECTED.items())
            and int(scores["rows"]) == 82380
            and prep["cardinalities"] == m["cardinalities"]
            and p["source_run_id"] == RUN_ID
            and p["training_only"] is True
            and p["extra_feature_count"] == 3
            and v.get("status") == "passed"
            and v.get("validation_prediction_parity_passed") is True
            and v.get("training_numeric_feature_parity_passed") is True
            and v.get("no_test_scoring") is True
            and v.get("serving_schema") == "8_num_10_cat"
        )
    except (OSError, KeyError, ValueError, TypeError, OverflowError):
        return False

def encode_extra_mileage(fields, profile):
    """Mirror scripts.us_bid_ray_current_features.add_model_mileage at inference."""
    from pricing.neural_bid_blend_live import _norm, raw_numeric
    raw = raw_numeric(fields)
    age = float(raw[1])
    miles = float(fields["odometer"])
    maker = _norm(fields.get("manufacturer"))
    model = _norm(fields.get("model"))
    per_model = profile["by_make_model"].get(maker, {}).get(model)
    per_make = profile["by_make"].get(maker)
    rate = float(per_model if per_model is not None else
                 per_make if per_make is not None else profile["pooled"])
    expected = rate * max(age, .5)
    numbers = np.array([
        np.log1p(rate),
        np.log1p(max(miles, 0.) / max(expected, 1000.)),
        np.sign(miles - expected) * np.log1p(abs(miles-expected) / 10000.),
    ], dtype=np.float64)
    if not (np.isfinite(numbers).all() and 1000 <= rate <= 50000):
        raise ValueError("Invalid reference annual mileage rate")
    center = np.array(profile["median"], dtype=np.float64)
    span = np.array(profile["iqr"], dtype=np.float64)
    if center.shape != (3,) or span.shape != (3,) or np.any(span <= 0):
        raise ValueError("Mismatched trained mileage-scaler dimension")
    return np.clip((numbers - center) / span, -8, 8).astype(np.float32)

class USBidResMLP:
    def __init__(self, root, *, _parity_test=False):
        import torch
        from scripts.us_bid_ray_15x3h import BidNetwork, bound_config
        from pricing.neural_bid_blend_live import CATS
        root = Path(root)
        if not _parity_test and not available(root):
            raise FileNotFoundError("New U.S. Bid checkpoint unavailable or failed release parity gate")
        folder = root / FOLDER
        m = json.loads((folder / "manifest.json").read_text())
        v = json.loads((folder / "verification.json").read_text())
        if not _parity_test and v.get("status") != "passed":
            raise ValueError("ResMLP release verification missing")
        self.manifest = m
        self.profile = json.loads((folder / "mileage_profiles.json").read_text())
        preprocessing = json.loads((root / PREPROCESS).read_text())
        self.median = np.asarray(preprocessing["numeric_median"], dtype=np.float64)
        self.scale = np.asarray(preprocessing["numeric_scale"], dtype=np.float64)
        self.maps = preprocessing["category_maps"]
        self.cards = [int(c) for c in preprocessing["cardinalities"]]
        if (self.median.shape != (5,) or self.scale.shape != (5,)
                or np.any(self.scale <= 0)
                or not np.isfinite(self.median).all()
                or not np.isfinite(self.scale).all()
                or tuple(preprocessing["categorical_features"]) != CATS
                or self.cards != m["cardinalities"]):
            raise ValueError("Frozen historic wholesale input dictionary differs from training")
        bundle = torch.load(folder / "weights.pt", map_location="cpu", weights_only=True)
        config = bound_config(dict(bundle["config"]))
        if (config != m["configuration"]
                or bundle["cards"] != self.cards
                or bundle["source_sha256"] != m["cohort_sha256"]
                or config["family"] != "resmlp" or config["loss"] != "huber_usd"):
            raise ValueError("ResMLP checkpoint differs from validated search")
        self.network = BidNetwork(self.cards,
                    float(bundle["state_dict"]["core.fm.base"]), config)
        self.network.load_state_dict(bundle["state_dict"], strict=True)
        self.network.eval()

    def features(self, fields):
        from pricing.neural_bid_blend_live import CATS, _norm, raw_numeric
        raw = raw_numeric(fields)
        original = np.clip(
            (np.where(np.isfinite(raw), raw, self.median)-self.median) / self.scale,
            -8, 8).astype(np.float32)
        enriched = encode_extra_mileage(fields, self.profile)
        categories = np.asarray([
            self.maps[name].get(_norm(fields.get(name)), 1) for name in CATS
        ], dtype=np.int64)
        if (original.shape != (5,) or enriched.shape != (3,)
                or categories.shape != (10,)
                or any(not (1 <= int(code) < card)
                       for code, card in zip(categories, self.cards))):
            raise ValueError("ResMLP bid schema out of bounds")
        return np.concatenate((original, enriched))[None,:], categories[None,:]

    def predict(self, fields):
        import torch
        from pricing.neural_bid_blend_live import CATS, _auction_date, checked_price
        num, cat = self.features(fields)
        with torch.inference_mode():
            log_price = float(self.network(torch.from_numpy(num),
                               torch.from_numpy(cat))[0])
        price = checked_price(float(np.exp(np.clip(log_price,
                        np.log(100), np.log(1e6)))), "U.S. Bid ResMLP")
        unknown = [name for name, code in zip(CATS, cat[0]) if code == 1]
        return {
            "market": "bid", "price": price,
            "model": "U.S. Bid ResMLP · Huber USD · historical wholesale",
            "neural_weight": 1., "resmlp_price": price,
            "historical_price_basis": "2015-index-normalized wholesale USD",
            "original_auction_date": _auction_date(fields["auction_date"]).isoformat(),
            "is_completed_auction_sale_estimate": True,
            "live_2026_validated": False, "verified": True,
            "unrecognized_categories": sorted(set(unknown)),
            "validation": EXPECTED,
        }

def load(root):
    import streamlit as st
    folder = Path(root) / FOLDER
    fingerprint = tuple((n, (folder/n).stat().st_mtime_ns,
                         (folder/n).stat().st_size) for n in FILES)
    @st.cache_resource(max_entries=1, show_spinner=False)
    def cached(root_path, signature):
        return USBidResMLP(root_path)
    return cached(str(root), fingerprint)
