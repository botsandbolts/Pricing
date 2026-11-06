"""Live, parity-verified DeepCross AFT + MiniLM 2021 asking-price inference.

Only versioned local files. PyTorch native state_dict + trusted compressed joblib
preprocessors; nothing is retrieved from a database or remote runtime. The
historical benchmark and its 2021 target are not current sale-price guarantees.
"""
import json,re
from pathlib import Path
import numpy as np
from pricing.minilm_signal import Encoder,price as head_price
from scripts.common import clean

REVISION="1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
FOLDER="models/neural-experimental-ask"
FILES=("weights.pt","manifest.json","preprocessing.json","tfidf.joblib",
       "svd.joblib","minilm-heads.json","verification.json")
NUMERIC=("year","age_2021","log_miles","miles_per_year",
         "log_description_chars","log_description_words","description_punctuation")
CATS=("manufacturer","model","type","transmission","state","fuel",
      "drive","title_status","cylinders","paint_color","condition")


def available(root):
    p=Path(root)/FOLDER
    return all((p/name).is_file() for name in FILES)


def numerical(fields,text):
    try:
        year=int(fields["year"]);mileage=float(fields["odometer"])
    except (ValueError,KeyError,TypeError) as exc:
        raise ValueError("Model year and mileage are required.") from exc
    if not 1900<=year<=2021:
        raise ValueError("This historical study supports model years through 2021.")
    if not np.isfinite(mileage) or not 0<=mileage<=2_000_000:
        raise ValueError("Mileage must be between 0 and 2,000,000.")
    age=max(2021-year,0)
    return np.array((year,age,np.log1p(mileage),mileage/max(age,1),
                     np.log1p(len(text)),
                     np.log1p(len(re.findall(r"\S+",text))),
                     np.log1p(len(re.findall(r"[!?]",text)))),dtype=np.float64)


def categoricals(fields,maps):
    return np.array([[maps[name].get(str(fields.get(name) or "unknown").lower().strip(),1)
                      for name in CATS]],dtype=np.int64)


class DeepCrossAsk:
    def __init__(self,root):
        import joblib
        import torch
        from scripts.deepcross_aft_tune import TunedDeepCross,config
        root=Path(root);folder=root/FOLDER
        if not available(root):
            raise FileNotFoundError("Verified DeepCross serving bundle not installed.")
        self.manifest=json.loads((folder/"manifest.json").read_text())
        self.verification=json.loads((folder/"verification.json").read_text())
        if not self.verification.get("verified"):
            raise ValueError("The experimental model did not pass holdout parity verification.")
        if ((self.manifest.get("variant"),self.manifest.get("feature_variant"),
             self.manifest.get("minilm_encoder_revision"))!=
            ("dropout12","minilm",REVISION)):
            raise ValueError("Wrong pretrained model identity or version.")
        self.state=json.loads((folder/"preprocessing.json").read_text())
        if tuple(self.state["numeric_names"])!=NUMERIC or tuple(self.state["categorical_names"])!=CATS:
            raise ValueError("Serving features differ from training features.")
        self.tfidf=joblib.load(folder/"tfidf.joblib")
        self.svd=joblib.load(folder/"svd.joblib")
        self.heads=json.loads((folder/"minilm-heads.json").read_text())["heads"]
        if len(self.heads)!=3:
            raise ValueError("Expected three original FIT-only MiniLM heads.")
        self.median=np.asarray(self.state["numeric_median"],dtype=float)
        self.scale=np.asarray(self.state["numeric_scale"],dtype=float)
        self.tmean=np.asarray(self.state["text_mean"],dtype=float)
        self.tstd=np.asarray(self.state["text_std"],dtype=float)
        if self.median.shape!=(7,) or self.scale.shape!=(7,) or self.tmean.shape!=(24,) or self.tstd.shape!=(24,):
            raise ValueError("Training numeric/text geometry mismatch.")
        self.encoder=Encoder(root/"models/xgboost-sparse-minilm/ask/minilm")
        # Training embeddings were computed in 32-text ONNX batches. Those
        # batches almost always reached the pinned 128-token truncation limit.
        # A single-text serving batch otherwise has shorter dynamic padding;
        # ONNX numerical pooling may then differ. Use an explicit 128-token
        # tensor to reproduce the train-time padded input geometry.
        self.encoder.tokenizer.enable_padding(length=128,pad_id=0,pad_token='[PAD]')
        ckpt=torch.load(folder/"weights.pt",map_location="cpu",weights_only=True)
        if (ckpt.get("variant"),ckpt.get("feature_variant"),ckpt.get("settings"))!=(
            "dropout12","minilm",config("dropout12")):
            raise ValueError("Checkpoint is not the validation-selected DeepCross.")
        if (ckpt["numeric_dimensions"],ckpt["text_dimensions"])!=(8,24):
            raise ValueError("Neural weights expect different input dimensions.")
        cards=self.manifest["cardinalities"]
        if cards!=ckpt["cardinalities"] or cards!=self.state["categorical_cardinalities"]:
            raise ValueError("Categorical embedding dimensions differ from training.")
        self.network=TunedDeepCross(cards,8,24,config("dropout12"))
        self.network.load_state_dict(ckpt["state_dict"],strict=True)
        self.network.eval()

    def prepare_text(self,text):
        """Raw description for TF-IDF; cleaned/redacted description for MiniLM."""
        text=str(text or "")[:8000]
        embedding=self.encoder.encode([clean(text)])
        guesses=np.array([head_price(h,embedding)[0] for h in self.heads],float)
        estimate=float(guesses.mean())
        m=self.manifest
        scalar=float(np.clip((np.log(estimate)-m["minilm_price_log_median"])/
                             m["minilm_price_log_iqr"],-8,8))
        raw_svd=self.svd.transform(self.tfidf.transform([text]))[0]
        svd=np.clip((raw_svd-self.tmean)/self.tstd,-8,8).astype(np.float32).reshape(1,-1)
        if not np.isfinite(scalar) or not np.isfinite(svd).all():
            raise ValueError("Text representation is not finite.")
        return dict(raw=text,svd=svd,minilm_scalar=scalar,price_text_usd=estimate)

    def encode_inputs(self,fields,prepared):
        raw=numerical(fields,prepared["raw"])
        scaled=np.clip((raw-self.median)/self.scale,-8,8)
        scaled=np.r_[scaled,prepared["minilm_scalar"]].astype(np.float32).reshape(1,-1)
        return scaled,categoricals(fields,self.state["category_maps"]),prepared["svd"]

    def predict_prepared(self,fields,prepared):
        import torch
        n,c,t=self.encode_inputs(fields,prepared)
        with torch.inference_mode():
            raw=self.network(torch.from_numpy(n),torch.from_numpy(c),
                             torch.from_numpy(t))[:,0].numpy()
        price=float(np.exp(np.clip(raw[0],np.log(100),np.log(1_000_000))))
        if not np.isfinite(price) or price<=0:
            raise ValueError("Invalid neural predicted asking price.")
        return dict(price=price,predicted_text_price=prepared["price_text_usd"],
                    model="DeepCross AFT MiniLM dropout12",
                    market="ask",historical_anchor="2021-05-01",
                    current_market_accuracy_validated=False)

    def predict(self,text,fields,explain=False):
        return self.predict_prepared(fields,self.prepare_text(text))


def load(root):
    """Lazy and cached; does not cache user descriptions or vehicle details."""
    import streamlit as st
    p=Path(root)/FOLDER
    version=tuple((name,(p/name).stat().st_mtime_ns,(p/name).stat().st_size) for name in FILES)
    return _cached(str(root),version)


def _cached(root,version):
    # Decorator applied once at import: only model weights and public preprocessors.
    return _cached_resource(root,version)


def _cached_resource(root,version):
    # Lazy import avoids Streamlit/PyTorch loading unless the experimental
    # model is actually selected. Streamlit manages cache_resource below.
    return _get_cached()(root,version)


def _get_cached():
    import streamlit as st
    @st.cache_resource(show_spinner=False,max_entries=2)
    def cached(root,version):
        return DeepCrossAsk(Path(root))
    return cached
