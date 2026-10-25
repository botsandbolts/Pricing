#!/usr/bin/env python3
"""Text-only car price benchmark. No structured predictors enter the models."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='2'
import argparse, json, re, hashlib, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

MONEY = re.compile(r'(?i)(?:[$€£]\s*\d[\d,.]*|\b\d[\d,.]*\s*(?:usd|dollars?|euros?|per month|/mo|monthly)\b|\b(?:price|asking|payment|down payment)\s*[:=]?\s*\$?\d[\d,.]*)')
def clean(s):
 s=re.sub(r'<[^>]+>',' ',str(s));s=re.sub(r'https?://\S+|\b\S+@\S+\b',' ',s)
 return re.sub(r'\s+',' ',MONEY.sub(' price_removed ',s)).strip().lower()
def metrics(y,p):
 p=np.maximum(0,p)
 return dict(mae_usd=float(mean_absolute_error(y,p)),rmse_usd=float(np.sqrt(mean_squared_error(y,p))),median_ape_pct=float(np.median(abs(y-p)/y)*100),rmsle=float(np.sqrt(np.mean((np.log1p(y)-np.log1p(p))**2))),r2=float(r2_score(y,p)))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',default='results');ap.add_argument('--max-rows',type=int,default=30000);ap.add_argument('--encoder',default='sentence-transformers/all-MiniLM-L6-v2');ap.add_argument('--tfidf-only',action='store_true');a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 source=a.data
 if str(source).endswith('.zip'):
  z=zipfile.ZipFile(source);source=z.open(next(n for n in z.namelist() if n.endswith('.csv')))
 # Reservoir-like uniform priority sample across the whole file, rather than first-N rows.
 rng=np.random.default_rng(42);parts=[];nraw=0
 for chunk in pd.read_csv(source,chunksize=10000,low_memory=False,usecols=lambda c: c in {"price","description","VIN","posting_date"}):
  nraw+=len(chunk);print(f'Scanned {nraw:,} rows',flush=True);chunk['price']=pd.to_numeric(chunk['price'],errors='coerce')
  chunk=chunk.loc[chunk.price.between(1000,150000)&chunk.description.notna()].copy()
  chunk=chunk[chunk.description.astype(str).str.len()>=80]
  chunk['_priority']=rng.random(len(chunk));parts.append(chunk)
  if a.max_rows: parts=[pd.concat(parts).nsmallest(a.max_rows,'_priority')]
 d=pd.concat(parts).sort_values('_priority');before=len(d)
 d['text']=d.description.map(clean);d=d[d.text.str.len()>=80].copy()
 d=d.drop_duplicates('text')
 if 'VIN' in d:
  v=d.VIN.fillna('').astype(str).str.upper().str.strip();valid=v.str.fullmatch(r'[A-HJ-NPR-Z0-9]{17}')
  d=d.loc[~(valid & v.duplicated())]
 d['date']=pd.to_datetime(d.posting_date,utc=True,errors='coerce');d=d.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
 if len(d)<100: raise ValueError('Need at least 100 usable rows')
 n=len(d);i=int(n*.7);j=int(n*.85);splits=[slice(0,i),slice(i,j),slice(j,n)]
 texts=d.text.tolist();y=d.price.to_numpy(float);target=np.log1p(y)
 report={'raw_rows_scanned':nraw,'sample_before_dedup':before,'usable_rows':n,'split':'chronological 70/15/15 after exact-text and valid-VIN deduplication','target':'asking price USD; compare raw versus log1p training','price_range':[1000,150000],'seed':42,'text_only':True,'encoder':a.encoder,'sentence_max_tokens':128,'sentence_windows':1,'date_ranges':{k:[str(d.date.iloc[s.start]),str(d.date.iloc[s.stop-1])] for k,s in zip(['train','validation','test'],splits)},'models':{}}
 vector=TfidfVectorizer(ngram_range=(1,2),min_df=3,max_df=.98,max_features=80000,sublinear_tf=True,dtype=np.float32)
 x=vector.fit_transform(texts[:i]);xv=vector.transform(texts[i:j]);xt=vector.transform(texts[j:]);features={'tfidf':(x,xv,xt)}
 if not a.tfidf_only:
  import torch
  from sentence_transformers import SentenceTransformer
  torch.set_num_threads(2);enc=SentenceTransformer(a.encoder,device='cpu');enc.max_seq_length=128
  # CPU baseline: first 128 wordpieces only; TF-IDF retains the full text.
  chunks=[];owners=[]
  for idx,t in enumerate(texts):
   chunks.append(t);owners.append(idx)
  e=enc.encode(chunks,batch_size=64,show_progress_bar=True,normalize_embeddings=True)
  emb=np.zeros((n,e.shape[1]),dtype=np.float32);np.add.at(emb,owners,e);emb/=np.bincount(owners)[:,None];emb/=np.maximum(np.linalg.norm(emb,axis=1,keepdims=True),1e-9)
  np.save(out/'embeddings.npy',emb);features['sentence_ridge']=(emb[:i],emb[i:j],emb[j:])
  features['combined'] = tuple(sparse.hstack([sparse.csr_matrix(u),sparse.csr_matrix(v)],format='csr') for u,v in zip(features['tfidf'],features['sentence_ridge']))
 predictions=pd.DataFrame({'actual_price':y[j:],'posting_date':d.date.iloc[j:].astype(str).to_numpy()})
 for name,(tr,va,te) in list(features.items()):
  for target_kind in ['log','raw']:
   trials=[]
   fit_y=target[:i] if target_kind=='log' else y[:i]
   inverse=np.expm1 if target_kind=='log' else np.asarray
   for alpha in [.1,1,10,100]:
    model=Ridge(alpha=alpha,solver='lsqr');model.fit(tr,fit_y);pred=np.maximum(0,inverse(model.predict(va)));trials.append((mean_absolute_error(y[i:j],pred),alpha,model))
   score,alpha,model=min(trials,key=lambda t:t[0]);pred=np.maximum(0,inverse(model.predict(te)))
   key=name+'_'+target_kind
   report['models'][key]={'alpha':alpha,'target':target_kind,'validation_mae_usd':float(score),'test':metrics(y[j:],pred),'validation_trials':[{ 'alpha':al,'mae_usd':float(sc)} for sc,al,_ in trials]}
   predictions[key]=pred;joblib.dump(model,out/f'{key}.joblib')
 baseline=np.repeat(np.median(y[:i]),n-j);report['models']['train_median_baseline']={'test':metrics(y[j:],baseline)}
 report['selected_by_validation']=min((k for k,v in report['models'].items() if 'validation_mae_usd' in v),key=lambda k:report['models'][k]['validation_mae_usd'])
 joblib.dump(vector,out/'tfidf_vectorizer.joblib');predictions.to_csv(out/'test_predictions.csv',index=False)
 (out/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
