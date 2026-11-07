"""Verified frozen BGE encoder -> unchanged PyTorch DeepCross/AFT.

The ORIGINAL trained neural price heads, categories and normalization remain
byte-for-byte identical. The frozen pretrained BGE can use a separately
parity-checked INT8 ONNX serving backend, reducing resident encoder memory.
If there is no independently approved INT8 release, use the original
SentenceTransformer; never use an unverified lightweight substitute.
"""
import json
import re
from pathlib import Path
import numpy as np

FOLDER="models/neural-experimental-ask-bge"
FILES=("weights.pt","preprocessing.json","manifest.json","verification.json")
NUMS=("year","age_2021","log_miles","miles_per_year",
      "log_description_chars","log_description_words","description_punctuation")
CATS=("manufacturer","model","type","transmission","state","fuel",
      "drive","title_status","cylinders","paint_color","condition")


def available(root):
    folder=Path(root)/FOLDER
    if not all((folder/name).is_file() for name in FILES):
        return False
    try:
        m=json.loads((folder/"manifest.json").read_text())
        v=json.loads((folder/"verification.json").read_text())
        return bool(
            m.get("candidate")=="bge-256"
            and m.get("price_model")=="EmbeddingOnlyDeepCross AFT"
            and m.get("encoder_model")=="BAAI/bge-small-en-v1.5"
            and m.get("text_price_heads")==0
            and m.get("native_checkpoints_only") is True
            and m.get("no_quantization") is True
            and v.get("verified") is True and v.get("status")=="passed"
            and v.get("native_encoder_parity") is True
            and v.get("full_network_parity_passed") is True
            and v.get("original_holdout_samples_checked",0)>=40)
    except (OSError,ValueError,TypeError,KeyError):
        return False


def numerical(fields,text):
    try:
        year=int(fields["year"]); miles=float(fields["odometer"])
    except (ValueError,TypeError,KeyError) as error:
        raise ValueError("Vehicle model year and mileage are required") from error
    if not 1900<=year<=2021:
        raise ValueError("Historical asking model trained on model years up to 2021")
    if not np.isfinite(miles) or not 0<=miles<=2e6:
        raise ValueError("Invalid odometer mileage; expected original US miles")
    age=max(2021-year,0)
    return np.asarray([year,age,np.log1p(miles),miles/max(age,1),
        np.log1p(len(text)),np.log1p(len(re.findall(r"\S+",text))),
        np.log1p(len(re.findall(r"[!?]",text)))],dtype=np.float64)


def categoricals(fields,maps):
    return np.asarray([[maps[name].get(str(fields.get(name) or "unknown")
        .lower().strip(),1) for name in CATS]],dtype=np.int64)


class BGEDeepCrossAsk:
    def __init__(self,root,*,_parity_test=False):
        import torch
        from scripts.raw_embedding_ask_train import EmbeddingOnlyDeepCross
        from scripts.raw_embedding_ask_encode import SPECS
        root=Path(root)
        if not _parity_test and not available(root):
            raise FileNotFoundError("Unverified native BGE + complete DeepCross/AFT release")
        folder=root/FOLDER
        self.manifest=json.loads((folder/"manifest.json").read_text())
        self.state=json.loads((folder/"preprocessing.json").read_text())
        s=self.state
        if (tuple(s["numeric_names"])!=NUMS or
            tuple(s["categorical_names"])!=CATS or
            s.get("text_price_heads")!=0):
            raise ValueError("Native asking features or price head unexpectedly changed")
        self.median=np.asarray(s["numeric_median"],dtype=np.float64)
        self.scale=np.asarray(s["numeric_scale"],dtype=np.float64)
        self.maps=s["category_maps"]
        self.cards=list(s["cardinalities"])
        if (self.median.shape!=(7,) or self.scale.shape!=(7,) or
            not np.isfinite(self.median).all() or
            not np.isfinite(self.scale).all() or
            not (self.scale>0).all() or set(self.maps)!=set(CATS) or
            len(self.cards)!=len(CATS)):
            raise ValueError("Original train-only numerical/categorical preprocessing differs")
        candidate="bge-256"
        model_id,max_tokens,dimension=SPECS[candidate]
        if (model_id,max_tokens,dimension) != (
            self.manifest["encoder_model"],self.manifest["encoder_max_tokens"],384):
            raise ValueError("Unexpected native pretrained sentence encoder")
        ckpt=torch.load(folder/"weights.pt",map_location="cpu",weights_only=True)
        if (ckpt.get("candidate"),ckpt.get("seed"),ckpt.get("model_id"),
            ckpt.get("model_revision"),ckpt.get("max_tokens"),
            ckpt.get("cardinalities"),ckpt.get("n_num"),
            ckpt.get("embedding_dim"),ckpt.get("pretrained_text_price_heads")) != (
            candidate,20261106,model_id,self.manifest["encoder_revision"],
            256,self.cards,7,384,0):
            raise ValueError("Native asking checkpoint is not the original trained model")
        self.price_network=EmbeddingOnlyDeepCross(self.cards,7,dimension)
        self.price_network.load_state_dict(ckpt["state_dict"],strict=True)
        self.price_network.eval()
        # The small pricing head is never quantized: Captum requires its
        # gradients and the verified checkpoint remains completely unchanged.
        # The ONLY compressed component is the frozen pretrained BGE encoder.
        # Release-gated INT8 contains original pinned tokenizer and a checked
        # 384D CLS + L2 contract, not a replacement language model.
        from pricing.quantized_bge_runtime import preferred,VerifiedInt8BGE
        if preferred(root,model_id,self.manifest["encoder_revision"]):
            self.encoder=VerifiedInt8BGE(
                root,model_id,self.manifest["encoder_revision"])
            self.encoder_backend="onnx-int8-avx2"
        else:
            from sentence_transformers import SentenceTransformer
            self.encoder=SentenceTransformer(
                model_id,revision=self.manifest["encoder_revision"],
                device="cpu",trust_remote_code=False)
            self.encoder_backend="native-pytorch"
        self.encoder.max_seq_length=256
        self.encoder.eval()
        if self.encoder.get_sentence_embedding_dimension()!=384:
            raise ValueError("Pinned BGE encoder output dimension has changed")

    def prepare_text(self,description):
        from scripts.common import clean
        raw=str(description or "")[:8000]
        cleaned=clean(raw)
        # The frozen artifact was saved float16; reproduce ONLY storage
        # rounding, not a new quantized encoder nor a new learned model.
        with __import__("torch").inference_mode():
            vector=self.encoder.encode(
                [cleaned],batch_size=1,convert_to_numpy=True,
                normalize_embeddings=True,show_progress_bar=False)
        vector=np.asarray(vector,dtype=np.float32)
        if (vector.shape!=(1,384) or not np.isfinite(vector).all() or
            not np.allclose(np.linalg.norm(vector,axis=1),1,atol=.01)):
            raise ValueError("Native BGE encoder output failed dimension/normalization")
        return {"embedding":vector.astype(np.float16).astype(np.float32),"raw":raw}

    def encode_inputs(self,fields,prepared):
        raw=numerical(fields,prepared["raw"])
        n=np.clip((np.where(np.isfinite(raw),raw,self.median)-
                   self.median)/self.scale,-8,8)
        c=categoricals(fields,self.maps)
        if any(not(1<=int(code)<card) for code,card in zip(c[0],self.cards)):
            raise ValueError("Invalid original category code")
        return n.astype(np.float32).reshape(1,7),c,prepared["embedding"]

    def predict_prepared(self,fields,prepared):
        import torch
        num,cat,text=self.encode_inputs(fields,prepared)
        with torch.inference_mode():
            value=self.price_network(torch.from_numpy(num),
                torch.from_numpy(cat.astype(np.int64)),
                torch.from_numpy(text))[:,0].numpy()
        price=float(np.exp(np.clip(value[0],np.log(100),np.log(1e6))))
        if not np.isfinite(price) or price<=0:
            raise ValueError("Full asking network produced an invalid historical price")
        return {"market":"ask","price":price,
                "model":"BGE-small-256 + complete DeepCross/AFT",
                "historical_anchor":"2021-05-01",
                "text_price_heads":0,
                "price_predictions_from":"full native DeepCross neural network",
                "current_market_accuracy_validated":False}

    def predict(self,description,fields,explain=False):
        return self.predict_prepared(fields,self.prepare_text(description))


def load(root):
    import streamlit as st
    folder=Path(root)/FOLDER
    fingerprint=tuple((name,(folder/name).stat().st_size,
        (folder/name).stat().st_mtime_ns) for name in FILES)
    # An optional verified INT8 encoder release changes cache identity. The
    # old float32 encoder is not a second active model on a new deployment.
    from pricing.quantized_bge_runtime import manifest_path
    extra=manifest_path(root)
    fingerprint+=(("bge-runtime",extra.stat().st_size,extra.stat().st_mtime_ns)
                  if extra.is_file() else ("bge-runtime",0,0),)
    # No network on page import; actual native BGE loads only when requested.
    return _cached(str(root),fingerprint)


def _cached(root,signature):
    import streamlit as st
    @st.cache_resource(show_spinner=False,max_entries=1)
    def create(original_path,release_fingerprint):
        return BGEDeepCrossAsk(original_path)
    return create(root,signature)
