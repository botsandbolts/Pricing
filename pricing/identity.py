"""Small frozen MiniLM encoder + supervised neural make/model price head.

Only make/model names enter this model. Cross-fitted regression heads produce
training scalars; a final head trains on all training rows for inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoModel,AutoTokenizer
from sklearn.model_selection import KFold
from scripts.common import clean,metrics

IDENTITY_MODEL='sentence-transformers/all-MiniLM-L6-v2'
IDENTITY_REVISION='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
MAX_TOKENS=64


def names(frame):
    def value(v):return 'unknown' if pd.isna(v) else clean(str(v)).lower().strip()[:256] or 'unknown'
    return [f'Make: {value(make)}. Model: {value(model)}.' for make,model in zip(frame.manufacturer,frame.model)]


@torch.inference_mode()
def encode(encoder,tokenizer,texts,device,batch_size=64):
    encoder.eval();rows=[]
    for start in range(0,len(texts),batch_size):
        tokens=tokenizer(texts[start:start+batch_size],padding=True,truncation=True,max_length=MAX_TOKENS,return_tensors='pt').to(device)
        h=encoder(**tokens).last_hidden_state
        mask=tokens['attention_mask'].unsqueeze(-1).to(h.dtype)
        pooled=nn.functional.normalize((h*mask).sum(1)/mask.sum(1).clamp_min(1),dim=-1)
        rows.append(pooled.cpu().numpy())
    return np.concatenate(rows)


def new_head(dim):return nn.Sequential(nn.Linear(dim,64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,1))


def fit_head(x,y,epochs=60,seed=42):
    torch.manual_seed(seed)
    log=np.log1p(np.asarray(y,dtype=float));mean=float(log.mean());std=float(max(log.std(),1e-6))
    x=torch.as_tensor(x,dtype=torch.float32);target=torch.tensor((log-mean)/std,dtype=torch.float32)
    head=new_head(x.shape[1]);opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
    rng=np.random.default_rng(seed)
    for epoch in range(epochs):
        head.train()
        order=rng.permutation(len(x))
        for at in range(0,len(x),256):
            idx=order[at:at+256];opt.zero_grad(set_to_none=True)
            loss=nn.functional.mse_loss(head(x[idx]).squeeze(-1),target[idx]);loss.backward();opt.step()
    return head.eval(),mean,std


@torch.inference_mode()
def head_prices(head,x,mean,std):
    z=head(torch.as_tensor(x,dtype=torch.float32)).squeeze(-1).numpy()*std+mean
    return np.expm1(np.clip(z,0,np.log1p(1e7)))


def build_features(data,data_path,output,encoder,tokenizer,*,source=IDENTITY_MODEL,revision=IDENTITY_REVISION,
                   device='cpu',folds=3,epochs=60,seed=42):
    from pricing.train import validate_cohort
    validate_cohort(data)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    config={'source':source,'revision':revision,'folds':folds,'epochs':epochs,'seed':seed,
            'encoder_frozen':True,'max_tokens':MAX_TOKENS,'head':'384→64→1 (dimension inferred from backbone)'}
    fingerprint=hashlib.sha256(Path(data_path).read_bytes()+json.dumps(config,sort_keys=True).encode()+Path(__file__).read_bytes()).hexdigest()
    meta=out/'identity-config.json'
    if meta.exists() and json.loads(meta.read_text())['fingerprint']!=fingerprint:
        raise ValueError('Make/model model configuration changed; choose a fresh output directory')
    if (out/'identity-features.npz').exists():
        print('Make/model price cache already complete',flush=True);return
    frame=data.reset_index(drop=True);text=names(frame)
    unique,indices=np.unique(text,return_inverse=True)
    print(f'Encoding {len(unique):,} unique make/model names with {source}',flush=True)
    vectors=encode(encoder,tokenizer,unique.tolist(),device)[indices]
    phases={p:np.flatnonzero(frame.phase==p) for p in ['train','validation','test']}
    tr=phases['train'];oof=np.full(len(tr),np.nan);fold_ids=np.full(len(tr),-1)
    for fold,(fit_idx,held_idx) in enumerate(KFold(folds,shuffle=True,random_state=seed).split(tr)):
        head,mean,std=fit_head(vectors[tr[fit_idx]],frame.price.iloc[tr[fit_idx]],epochs,seed)
        oof[held_idx]=head_prices(head,vectors[tr[held_idx]],mean,std);fold_ids[held_idx]=fold
        print(f'Make/model price fold {fold+1}/{folds} complete',flush=True)
    head,mean,std=fit_head(vectors[tr],frame.price.iloc[tr],epochs,seed)
    vp=head_prices(head,vectors[phases['validation']],mean,std)
    tp=head_prices(head,vectors[phases['test']],mean,std)
    model_dir=out/'identity-model';model_dir.mkdir(exist_ok=True)
    encoder.save_pretrained(model_dir/'encoder',safe_serialization=True);tokenizer.save_pretrained(model_dir/'tokenizer')
    torch.save({'head':head.state_dict(),'mean':mean,'std':std,'dimensions':vectors.shape[1]},model_dir/'regression.pt')
    meta.write_text(json.dumps({'fingerprint':fingerprint,'data_sha256':hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),**config},indent=2))
    np.savez(out/'identity-features.npz',oof=oof,validation=vp,test=tp,
        train_ids=frame.row_id.iloc[tr].to_numpy(str),validation_ids=frame.row_id.iloc[phases['validation']].to_numpy(str),
        test_ids=frame.row_id.iloc[phases['test']].to_numpy(str),fingerprint=fingerprint)
    pd.DataFrame({'row_id':frame.row_id.iloc[tr].to_numpy(),'fold':fold_ids,'predicted_make_model_price':oof}).to_csv(out/'identity-oof-audit.csv',index=False)
    scores={'status':'evaluated','scope':'make/model-only frozen MiniLM + supervised neural head, not full E5 stack',
            'rows':{p:len(idx) for p,idx in phases.items()},'test':metrics(frame.price.iloc[phases['test']],tp),'config':config}
    (out/'identity-evaluation.json').write_text(json.dumps(scores,indent=2));print(json.dumps(scores,indent=2),flush=True)


def read_features(output,data_path,phases):
    out=Path(output);meta=json.loads((out/'identity-config.json').read_text())
    if meta['data_sha256']!=hashlib.sha256(Path(data_path).read_bytes()).hexdigest():
        raise ValueError('Make/model features belong to different data')
    f=np.load(out/'identity-features.npz')
    if str(f['fingerprint'])!=meta['fingerprint']:raise ValueError('Make/model feature fingerprint mismatch')
    for phase,key in [('train','train_ids'),('validation','validation_ids'),('test','test_ids')]:
        if not np.array_equal(f[key],phases[phase].row_id.to_numpy(str)):raise ValueError('Make/model feature row mismatch')
    return f


class IdentityPrice:
    def __init__(self,path,device='cpu'):
        path=Path(path);self.device=device
        self.encoder=AutoModel.from_pretrained(path/'encoder').to(device).eval()
        self.tokenizer=AutoTokenizer.from_pretrained(path/'tokenizer')
        state=torch.load(path/'regression.pt',map_location='cpu',weights_only=True)
        self.head=new_head(state['dimensions']);self.head.load_state_dict(state['head']);self.head.eval()
        self.mean,self.std=state['mean'],state['std']
    def predict(self,make,model):
        text=names(pd.DataFrame({'manufacturer':[make],'model':[model]}))
        vector=encode(self.encoder,self.tokenizer,text,self.device)
        return float(head_prices(self.head,vector,self.mean,self.std)[0])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='training-cache/ten-percent.jsonl');p.add_argument('--output',default='artifacts/e5-base-two-prices-10')
    p.add_argument('--device',default='cpu');p.add_argument('--epochs',type=int,default=60);p.add_argument('--folds',type=int,default=3)
    a=p.parse_args();torch.set_num_threads(4)
    encoder=AutoModel.from_pretrained(IDENTITY_MODEL,revision=IDENTITY_REVISION).to(a.device).eval()
    encoder.requires_grad_(False)
    tokenizer=AutoTokenizer.from_pretrained(IDENTITY_MODEL,revision=IDENTITY_REVISION)
    build_features(pd.read_json(a.data,lines=True),a.data,a.output,encoder,tokenizer,device=a.device,folds=a.folds,epochs=a.epochs)
