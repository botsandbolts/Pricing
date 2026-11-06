"""BGE-small 256-token embeddings -> entire DeepCross/AFT ask price model.

No Ridge, no text-price scalar, no TF-IDF or SVD. The pinned pretrained BGE
encoder emits one L2-normalized sentence vector; a learned projection and
the complete DeepCross+AFT neural network output the historical dollar price.
Refuse inference unless packaged ONNX+network parity passed on held-out data.
"""
import json
from pathlib import Path
import numpy as np

FOLDER="models/neural-experimental-ask-bge"
FILES=("encoder-int8.onnx","tokenizer.json","weights.pt","preprocessing.json",
       "manifest.json","verification.json")


def available(root):
    folder=Path(root)/FOLDER
    if not all((folder/item).is_file() for item in FILES):
        return False
    try:
        m=json.loads((folder/"manifest.json").read_text())
        v=json.loads((folder/"verification.json").read_text())
        return (
            m.get("candidate")=="bge-256" and
            m.get("text_price_heads")==0 and
            m.get("price_model")=="EmbeddingOnlyDeepCross AFT" and
            v.get("status")=="passed" and v.get("verified") is True and
            v.get("original_holdout_samples_checked",0)>=40 and
            v.get("full_network_parity_passed") is True)
    except (OSError,ValueError,TypeError,KeyError):
        return False


class BGEDeepCrossAsk:
    def __init__(self,root):
        import torch
        import onnxruntime as ort
        from tokenizers import Tokenizer
        from scripts.raw_embedding_ask_train import EmbeddingOnlyDeepCross
        from pricing.neural_experimental_live import numerical,categoricals
        self._numerical=numerical
        self._categoricals=categoricals
        root=Path(root)
        if not available(root):
            raise FileNotFoundError("Full BGE neural asking model has not passed release parity.")
        folder=root/FOLDER
        self.manifest=json.loads((folder/"manifest.json").read_text())
        self.state=json.loads((folder/"preprocessing.json").read_text())
        self.numeric_median=np.asarray(self.state["numeric_median"],dtype=np.float64)
        self.numeric_scale=np.asarray(self.state["numeric_scale"],dtype=np.float64)
        if self.numeric_median.shape!=(7,) or self.numeric_scale.shape!=(7,) or (self.numeric_scale<=0).any():
            raise ValueError("Invalid seven-feature ask numeric scaler")
        self.tok=Tokenizer.from_file(str(folder/"tokenizer.json"))
        self.tok.enable_truncation(max_length=256)
        self.tok.enable_padding(length=256,pad_id=0,pad_token="[PAD]")
        so=ort.SessionOptions()
        so.intra_op_num_threads=2;so.inter_op_num_threads=1
        self.encoder=ort.InferenceSession(str(folder/"encoder-int8.onnx"),
                                          sess_options=so,providers=["CPUExecutionProvider"])
        self.network=EmbeddingOnlyDeepCross(self.state["cardinalities"],7,384)
        checkpoint=torch.load(folder/"weights.pt",map_location="cpu",weights_only=True)
        if ((checkpoint.get("candidate"),checkpoint.get("seed"),
             checkpoint.get("model_revision"),checkpoint.get("n_num"),
             checkpoint.get("embedding_dim"),checkpoint.get("pretrained_text_price_heads"))!=(
             "bge-256",20261106,self.manifest["encoder_revision"],7,384,0)):
            raise ValueError("Wrong frozen BGE + DeepCross research checkpoint")
        self.network.load_state_dict(checkpoint["state_dict"],strict=True)
        self.network.eval()

    def prepare_text(self,description):
        from scripts.common import clean
        clean_description=clean(str(description or "")[:8000])
        tok=self.tok.encode(clean_description)
        values=self.encoder.run(None,{
          "input_ids":np.asarray([tok.ids],dtype=np.int64),
          "attention_mask":np.asarray([tok.attention_mask],dtype=np.int64)})[0]
        vector=np.asarray(values,dtype=np.float32)
        if vector.shape!=(1,384) or not np.isfinite(vector).all() or not np.allclose(
                np.linalg.norm(vector,axis=1),1.,atol=.01):
            raise ValueError("Pretrained BGE sentence embedding is not finite or normalized")
        # Frozen training artifact was float16. Reproduce this rounding at serving.
        return {"embedding":vector.astype(np.float16).astype(np.float32),
                "raw":str(description or "")[:8000]}

    def encode_inputs(self,fields,prepared):
        raw=self._numerical(fields,prepared["raw"])
        n=np.clip((np.where(np.isfinite(raw),raw,self.numeric_median)-
                   self.numeric_median)/self.numeric_scale,-8,8)
        c=self._categoricals(fields,self.state["category_maps"])
        return n.astype(np.float32).reshape(1,7),c,prepared["embedding"]

    def predict_prepared(self,fields,prepared):
        import torch
        num,cat,text=self.encode_inputs(fields,prepared)
        with torch.inference_mode():
            output=self.network(torch.from_numpy(num),
                                torch.from_numpy(cat.astype(np.int64)),
                                torch.from_numpy(text))[:,0].numpy()
        price=float(np.exp(np.clip(output[0],np.log(100),np.log(1000000))))
        if not np.isfinite(price) or price<=0:
            raise ValueError("Full DeepCross network returned an invalid asking price")
        return {"price":price,"market":"ask",
                "model":"BGE-small 256 + EmbeddingOnlyDeepCross+AFT",
                "historical_anchor":"2021-05-01",
                "text_price_heads":0,
                "price_predictions_from":"full DeepCross neural network",
                "current_market_accuracy_validated":False}

    def predict(self,text,fields,explain=False):
        return self.predict_prepared(fields,self.prepare_text(text))


def load(root):
    import streamlit as st
    folder=Path(root)/FOLDER
    signature=tuple((name,(folder/name).stat().st_size,
                    (folder/name).stat().st_mtime_ns) for name in FILES)
    return _cached(str(root),signature)


def _cached(root,signature):
    import streamlit as st
    @st.cache_resource(show_spinner=False,max_entries=1)
    def build(path,key):
        return BGEDeepCrossAsk(path)
    return build(root,signature)
