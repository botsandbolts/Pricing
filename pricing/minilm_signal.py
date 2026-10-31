"""Small ONNX MiniLM encoder and cross-fitted description-price heads."""
import json
from pathlib import Path
import numpy as np

class Encoder:
    def __init__(self,path):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        path=Path(path)
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(path/'encoder-int8.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        self.tokenizer=Tokenizer.from_file(str(path/'tokenizer.json'))
        self.tokenizer.enable_truncation(max_length=128)
        self.tokenizer.enable_padding(pad_id=0,pad_token='[PAD]')
    def encode(self,texts,batch_size=32):
        output=[]
        for start in range(0,len(texts),batch_size):
            encoded=self.tokenizer.encode_batch([str(t)[:8000] for t in texts[start:start+batch_size]])
            feed={'input_ids':np.array([e.ids for e in encoded],dtype=np.int64),
                  'attention_mask':np.array([e.attention_mask for e in encoded],dtype=np.int64)}
            output.append(self.session.run(None,feed)[0])
        return np.concatenate(output).astype(np.float32)

def fit_head(embeddings,prices):
    from sklearn.linear_model import Ridge
    m=Ridge(alpha=100.)
    m.fit(embeddings,np.log(np.asarray(prices,float)))
    return {'coef':m.coef_.astype(float).tolist(),'intercept':float(m.intercept_)}

def price(head,embeddings):
    raw=np.asarray(embeddings)@np.asarray(head['coef'])+head['intercept']
    return np.exp(np.clip(raw,0,np.log(1e7)))

def crossfit(embeddings,prices,groups):
    from sklearn.model_selection import GroupKFold
    oof=np.full(len(prices),np.nan)
    for fit,held in GroupKFold(3).split(embeddings,groups=groups):
        oof[held]=price(fit_head(embeddings[fit],prices[fit]),embeddings[held])
    if not np.isfinite(oof).all():raise ValueError('Incomplete MiniLM head OOF')
    return oof,fit_head(embeddings,prices)


def mean_price(heads,embeddings):
    """Arithmetic mean in dollars, not exponentiated mean log price."""
    if len(heads)!=2:raise ValueError('Exactly two MiniLM price heads are required')
    return np.mean([price(h,embeddings) for h in heads],axis=0)


def two_price_crossfit(embeddings,prices,groups,seeds=(42,1729)):
    """Two group-held-out predictions per row, with two-fold splits repeated twice.

    First repetition's complementary heads also form the inference pair. No
    outcome of a predicted group enters either of its two training predictions.
    Call only on the outer XGBoost training partition for nested cross-fitting.
    """
    groups=np.asarray(groups).astype(str);unique=np.unique(groups)
    if len(unique)<4 or len(seeds)!=2:raise ValueError('Need four groups and two distinct repetitions')
    values=np.full((len(prices),2),np.nan);serving=[]
    for repeat,seed in enumerate(seeds):
        shuffled=np.random.default_rng(seed).permutation(unique)
        halves=np.array_split(shuffled,2)
        for held_groups in halves:
            held=np.isin(groups,held_groups);fit=~held
            head=fit_head(np.asarray(embeddings)[fit],np.asarray(prices)[fit])
            values[held,repeat]=price(head,np.asarray(embeddings)[held])
            if repeat==0:serving.append(head)
    if not np.isfinite(values).all() or (values<=0).any():raise ValueError('Incomplete two-price cross-fitting')
    return values.mean(axis=1),serving,values
