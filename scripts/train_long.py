"""Train the browser's long-description model using the exact browser ONNX encoder."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='8'
from pathlib import Path
import json,zipfile,sys,hashlib
import numpy as np,pandas as pd,joblib,onnxruntime as ort
from scipy import sparse
from sklearn.linear_model import Ridge
from transformers import AutoTokenizer
sys.path.insert(0,str(Path(__file__).parent))
from baseline import clean,metrics
ROOT=Path(__file__).resolve().parents[1];DIST=ROOT/'dist';CACHE=ROOT/'training-cache';CACHE.mkdir(exist_ok=True)
source=Path(sys.argv[1]);old=Path(sys.argv[2]) if len(sys.argv)>2 else None;rng=np.random.default_rng(42);parts=[]
with zipfile.ZipFile(source) as z:
 with z.open('vehicles.csv') as f:
  for c in pd.read_csv(f,chunksize=10000,usecols=['price','description','VIN','posting_date'],low_memory=False):
   c.price=pd.to_numeric(c.price,errors='coerce');c=c[c.price.between(1000,150000)&c.description.notna()].copy();c=c[c.description.astype(str).str.len()>=80];c['_priority']=rng.random(len(c));parts=[pd.concat(parts+[c]).nsmallest(12000,'_priority')]
d=parts[0].sort_values('_priority');d['text']=d.description.map(clean);d=d[d.text.str.len()>=80].copy();d=d.drop_duplicates('text');v=d.VIN.fillna('').astype(str).str.upper().str.strip();d=d.loc[~(v.str.fullmatch('[A-HJ-NPR-Z0-9]{17}')&v.duplicated())];d['date']=pd.to_datetime(d.posting_date,utc=True,errors='coerce');d=d.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
n=len(d);i=int(.7*n);j=int(.85*n);texts=d.text.tolist();y=d.price.to_numpy(float);assert n==10892
# Deterministic row/split reproduction check against the earlier benchmark.
if old is not None:
 past=pd.read_csv(old/'test_predictions.csv');assert np.allclose(past.actual_price,y[j:])
 vec=joblib.load(old/'tfidf_vectorizer.joblib')
else:
 from sklearn.feature_extraction.text import TfidfVectorizer
 vec=TfidfVectorizer(ngram_range=(1,2),min_df=3,max_df=.98,max_features=80000,sublinear_tf=True,dtype=np.float32).fit(texts[:i])
x=vec.transform(texts)
encdir=Path(os.environ.get('PRICING_ENCODER_DIR',str(DIST/'encoder')));tok=AutoTokenizer.from_pretrained(str(encdir),local_files_only=True)
options=ort.SessionOptions();options.intra_op_num_threads=8;options.inter_op_num_threads=1
session=ort.InferenceSession(str(encdir/'onnx/model_quantized.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
ids=tok([t[:8000] for t in texts],add_special_tokens=False,truncation=False,verbose=False)['input_ids'];windows=[];owners=[]
for row,tokenids in enumerate(ids):
 for start in range(0,len(tokenids),126):windows.append([0]+tokenids[start:start+126]+[2]);owners.append(row)
emb=np.zeros((n,768),np.float32)
for offset in range(0,len(windows),32):
 batch=windows[offset:offset+32];L=max(map(len,batch));a=np.ones((len(batch),L),np.int64);mask=np.zeros_like(a)
 for k,t in enumerate(batch):a[k,:len(t)]=t;mask[k,:len(t)]=1
 h=session.run(None,{'input_ids':a,'attention_mask':mask})[0];pooled=(h*mask[:,:,None]).sum(1)/mask.sum(1)[:,None];pooled/=np.linalg.norm(pooled,axis=1,keepdims=True)
 np.add.at(emb,owners[offset:offset+32],pooled)
 if offset%1024==0:print(f'Encoded {offset:,}/{len(windows):,} chunks',flush=True)
emb/=np.bincount(owners)[:,None];emb/=np.maximum(np.linalg.norm(emb,axis=1,keepdims=True),1e-9);np.save(CACHE/'xlmr_embeddings.npy',emb)
features={'tfidf':x,'sentence':sparse.csr_matrix(emb),'combined':sparse.hstack([x,sparse.csr_matrix(emb)],format='csr')};results={};models={}
for name,feat in features.items():
 choices=[]
 for alpha in [.1,1,10,100]:
  m=Ridge(alpha=alpha,solver='lsqr').fit(feat[:i],np.log1p(y[:i]));pred=np.maximum(0,np.expm1(m.predict(feat[i:j])));choices.append((float(np.mean(abs(y[i:j]-pred))),alpha,m))
 val,alpha,m=min(choices,key=lambda a:a[0]);p=np.maximum(0,np.expm1(m.predict(feat[j:])));results[name]={'validation_mae_usd':val,'alpha':alpha,'test':metrics(y[j:],p)};models[name]=m
 joblib.dump(m,CACHE/f'{name}.joblib')
# Validation-only residual band; diagnostic, not a guaranteed individual confidence interval.
m=models['combined'];vp=m.predict(features['combined'][i:j]);res=np.abs(np.log1p(y[i:j])-vp);q=float(np.quantile(res,.8,method='higher'));testpred=m.predict(features['combined'][j:]);coverage=float(np.mean(np.abs(np.log1p(y[j:])-testpred)<=q))
terms=vec.get_feature_names_out().tolist()
meta={'trained_rows':i,'validation_rows':j-i,'test_rows':n-j,'usable_rows':n,'encoder':'XLM-RoBERTa sentence transformer (paraphrase-multilingual-mpnet-base-v2, int8 ONNX)','max_characters':8000,'chunk_tokens':126,'embedding_dimensions':768,'encoder_model_id':'Xenova/paraphrase-multilingual-mpnet-base-v2','encoder_revision':'e5d116277351513fd260955ece953ecddde7046e','encoder_sha256':'280b5fe103cc79d891d672f47826067835b7feed8b0b1865e34ed38f21719b49','encoder_bytes':278647663,'evaluation':results,'band80_log_residual':q,'band80_test_coverage':coverage,'period':'April–May 2021','price_min':1000,'price_max':150000,'target':'log1p USD asking price','form_validation':'Held-out metrics evaluate original seller descriptions, not generated form summaries.'}
export={'meta':meta,'terms':terms,'idf':vec.idf_.tolist(),'models':{k:{'coef':v.coef_.tolist(),'intercept':float(v.intercept_)} for k,v in models.items()}}
export['idf']=[float(f'{v:.7g}') for v in export['idf']]
for item in export['models'].values():
 item['coef']=[float(f'{v:.7g}') for v in item['coef']];item['intercept']=float(f"{item['intercept']:.9g}")
(DIST/'pricing-model.json').write_text(json.dumps(export,separators=(',',':')));(DIST/'evaluation.json').write_text(json.dumps(meta,indent=2))
# Tokenizer parity includes multilingual text, accents and punctuation.
fixtures=[]
for idx in [0,i,j,n-1]:
 t=texts[idx][:1800];a=vec.transform([t]);tokenids=tok(t,add_special_tokens=False)['input_ids'];fixtures.append({'text':t,'ids':tokenids,'tfidf_log':float(models['tfidf'].predict(a)[0])})
for t in ['Gepflegter Gebrauchtwagen mit vollständigem Serviceheft. Unfallfrei, 95000 Kilometer.','רכב שמור עם היסטוריית טיפולים מלאה.','Voiture en bon état, entretien régulier.','2015 Toyota Corolla. 95000 miles. Clean title.']:
 a=vec.transform([t]);fixtures.append({'text':t,'ids':tok(t,add_special_tokens=False)['input_ids'],'tfidf_log':float(models['tfidf'].predict(a)[0])})
(DIST/'parity-fixtures.json').write_text(json.dumps(fixtures))
print(json.dumps(meta,indent=2),flush=True)
