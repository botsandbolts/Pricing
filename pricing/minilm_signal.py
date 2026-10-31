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
