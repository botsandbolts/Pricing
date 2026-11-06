"""Recover EXACT original train-only neural feature encoders for portable inference.

We intentionally reuse neural_ask_prepare.encode_features unchanged and capture
the fitted sklearn TF-IDF and SVD objects via controlled constructor patches.
Independent numeric/category/SVD calculations must then match frozen tensors.
"""
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD as ActualSVD
from sklearn.feature_extraction.text import TfidfVectorizer as ActualTFIDF
from scripts import neural_ask_prepare as source


def rebuild(frame, train):
    captured={}
    def tfidf(*args,**kwargs):
        result=ActualTFIDF(*args,**kwargs)
        captured["tfidf"]=result
        return result
    def svd(*args,**kwargs):
        result=ActualSVD(*args,**kwargs)
        captured["svd"]=result
        return result
    with patch.object(source,"TfidfVectorizer",tfidf),patch.object(source,"TruncatedSVD",svd):
        numeric,category,text,cards,definitions=source.encode_features(frame,train)
    if set(captured)!={"tfidf","svd"}:
        raise ValueError("Failed to capture the exact fitted text encoders")
    raw=source.numeric_features(frame)
    med=np.nanmedian(raw[train],axis=0)
    med[~np.isfinite(med)]=0
    imputed=np.where(np.isfinite(raw),raw,med)
    q25,q75=np.percentile(imputed[train],[25,75],axis=0)
    scale=np.maximum(q75-q25,1)
    reproduced=np.clip((imputed-med)/scale,-8,8).astype(np.float32)
    if not np.array_equal(reproduced,numeric):
        raise ValueError("Robust numerical normalization differs from training")
    maps={}
    for j,name in enumerate(source.CATS):
        vals=(frame[name] if name in frame else pd.Series("",index=frame.index))
        vals=vals.fillna("unknown").astype(str).str.lower().str.strip()
        counts=vals.iloc[np.flatnonzero(train)].value_counts()
        retained=[v for v in counts.head(source.TOP[name]).index
                  if counts[v]>=10 and v and v!="unknown"]
        maps[name]={v:i+2 for i,v in enumerate(retained)}
        recoded=vals.map(maps[name]).fillna(1).to_numpy(np.int32)
        if not np.array_equal(recoded,category[:,j]):
            raise ValueError("Category codes diverged for "+name)
    tmean=text[train].mean(axis=0)
    tstd=np.maximum(text[train].std(axis=0),.02)
    # Original text in encode_features is already normalized. Recover unscaled
    # SVD mean/SD by replaying the fitted encoder over TRAIN rows, in the same
    # summation order and float32 buffers as the first training run.
    raw_svd=np.zeros_like(text,dtype=np.float32)
    docs=frame.text.fillna("").astype(str).tolist()
    for i in range(0,len(frame),4096):
        raw_svd[i:i+4096]=captured["svd"].transform(captured["tfidf"].transform(docs[i:i+4096]))
    avg=raw_svd[train].mean(axis=0)
    std=np.maximum(raw_svd[train].std(axis=0),.02)
    rebuilt=np.clip((raw_svd-avg)/std,-8,8).astype(np.float32)
    if not np.allclose(rebuilt,text,atol=.00001,rtol=0):
        raise ValueError("Original SVD normalization cannot be reproduced")
    state=dict(numeric_names=list(source.NUMS),numeric_median=med.tolist(),
               numeric_scale=scale.tolist(),categorical_names=list(source.CATS),
               category_maps=maps,text_mean=avg.tolist(),text_std=std.tolist(),
               text_dimensions=text.shape[1],numeric_dimensions=numeric.shape[1],
               categorical_cardinalities=cards,atlas_seed=42)
    return numeric,category,text,cards,state,captured["tfidf"],captured["svd"]
