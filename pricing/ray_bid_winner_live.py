"""Fail-closed, unblended non-Lemon Ray Tune wholesale Bid winner.
Only the 9-Nov-2026 full original auction cohort is admissible. The
historical 2015-indexed sale estimate is not a live dealer bid.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

FOLDER = "models/ray-bid-9am-winner"
FILES = ("weights.pt", "manifest.json", "verification.json")
PREPROCESS = "models/neural-experimental-bid/preprocessing.json"


def _sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def available(root):
    root = Path(root)
    folder = root / FOLDER
    if not all((folder / name).is_file() for name in FILES):
        return False
    if not (root / PREPROCESS).is_file():
        return False
    try:
        m = json.loads((folder / "manifest.json").read_text())
        v = json.loads((folder / "verification.json").read_text())
        return bool(
            m.get("status") == "release_candidate_verified"
            and m.get("market") == "bid"
            and m.get("source_workflow_run") == 42570947698
            and m.get("original_training_no_lemon") is True
            and m.get("selection_metric") == "TRAIN internal CVaR95_absolute_USD"
            and m.get("configuration", {}).get("family") == "deepfm"
            and m["configuration"].get("loss") == "cvar95_weighted_huber"
            and m.get("original_test_scored") is False
            and m.get("checkpoint_sha256") == _sha(folder / "weights.pt")
            and m.get("preprocessing_sha256") == _sha(root / PREPROCESS)
            and v.get("status") == "passed"
            and v.get("inner_train_partition_verified") is True
            and v.get("original_validation_rows") == 82380
            and v.get("checkpoint_forward_finite") is True
            and v.get("model_not_blended_with_lemon") is True
            and v.get("original_test_scored") is False
        )
    except (OSError, KeyError, TypeError, ValueError):
        return False


class RayWinnerWholesale:
    def __init__(self, root, *, _parity_test=False):
        import torch
        from scripts.ray_bid_9am import BidNetwork, bound_config
        from pricing.neural_bid_blend_live import CATS
        root = Path(root)
        if not _parity_test and not available(root):
            raise FileNotFoundError("Ray Bid winner checkpoint or provenance gate unavailable")
        folder = root / FOLDER
        self.manifest = json.loads((folder / "manifest.json").read_text())
        prep = json.loads((root / PREPROCESS).read_text())
        self.median = np.asarray(prep["numeric_median"], dtype=np.float64)
        self.scale = np.asarray(prep["numeric_scale"], dtype=np.float64)
        self.maps = prep["category_maps"]
        self.cards = [int(x) for x in prep["cardinalities"]]
        if (len(self.median) != 5 or len(self.scale) != 5
                or not np.isfinite(self.median).all()
                or not np.isfinite(self.scale).all()
                or (self.scale <= 0).any()
                or tuple(prep["categorical_features"]) != CATS):
            raise ValueError("Wholesale frozen preprocessing schema changed")
        package = torch.load(folder / "weights.pt", map_location="cpu", weights_only=True)
        cfg = bound_config(dict(package["config"]))
        if (cfg != self.manifest["configuration"]
                or package.get("cards") != self.cards
                or package.get("source_sha256") != self.manifest["cohort_sha256"]):
            raise ValueError("Winning Bid model and training identity do not match")
        self.model = BidNetwork(self.cards,
            float(package["state_dict"]["core.fm.base"]), cfg)
        self.model.load_state_dict(package["state_dict"], strict=True)
        self.model.eval()

    def predict(self, fields):
        import torch
        from pricing.neural_bid_blend_live import CATS, _norm, raw_numeric, checked_price, _auction_date
        raw = raw_numeric(fields)
        num = np.clip((np.where(np.isfinite(raw), raw, self.median)
                    - self.median) / self.scale, -8, 8).astype(np.float32).reshape(1,5)
        categories = np.asarray(
            [self.maps[name].get(_norm(fields.get(name)), 1) for name in CATS],
            dtype=np.int64).reshape(1,10)
        if any(not (1 <= int(i) < card)
               for i, card in zip(categories[0], self.cards)):
            raise ValueError("Wholesale category code is outside frozen training vocabulary")
        with torch.inference_mode():
            logprice = float(self.model(torch.from_numpy(num),
                           torch.from_numpy(categories))[0])
        price = checked_price(float(np.exp(np.clip(logprice,
            np.log(100.0), np.log(1e6)))), "Ray CVaR95 Huber DeepFM")
        unknown = [name for name, code in zip(CATS, categories[0]) if code == 1]
        return {
            "market":"bid", "price":price,
            "model":"Non-Lemon Ray DeepFM / CVaR95-weighted Huber",
            "deepfm_price":price, "neural_weight":1.0,
            "historical_price_basis":"2015-index-normalized wholesale USD",
            "original_auction_date":_auction_date(fields["auction_date"]).isoformat(),
            "is_completed_auction_sale_estimate":True,
            "live_2026_validated":False, "verified":True,
            "unrecognized_categories":sorted(set(unknown))
        }


def load(root):
    import streamlit as st
    folder = Path(root) / FOLDER
    fp = tuple((name, (folder/name).stat().st_mtime_ns,
                (folder/name).stat().st_size) for name in FILES)
    @st.cache_resource(max_entries=1, show_spinner=False)
    def factory(root_path, fingerprint):
        return RayWinnerWholesale(root_path)
    return factory(str(root), fp)
