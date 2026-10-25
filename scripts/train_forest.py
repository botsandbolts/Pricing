"""Encode description-only E5 features, then fit Isolation Forest + Extra Trees."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='8'
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import argparse,json,time,hashlib,subprocess,sys
import numpy as np,pandas as pd,onnxruntime as ort
from transformers import AutoTokenizer
p=argparse.ArgumentParser();p.add_argument('--models',required=True);a=p.parse_args();root=Path(__file__).resolve().parents[1];cache=root/'training-cache';models=Path(a.models);data=pd.read_json(cache/'dense-cohort.jsonl',lines=True)
def encoder(path,dim,texts,cache_name,prefix='',max_length=512):
 tok=AutoTokenizer.from_pretrained(str(path),local_files_only=True);prefix_ids=tok(prefix,add_special_tokens=False)['input_ids'];budget=max_length-2-len(prefix_ids);cfg=json.loads((path/'config.json').read_text());bos=cfg.get('bos_token_id',101);eos=cfg.get('eos_token_id',102);pad=cfg.get('pad_token_id',0)
 assert cfg.get('model_type')=='xlm-roberta'
 ids=tok(texts,add_special_tokens=False,truncation=False,verbose=False)['input_ids'];windows=[];owners=[]
 for row,seq in enumerate(ids):
  positions=range(0,max(1,len(seq)),budget)
  for start in positions:windows.append([bos]+prefix_ids+seq[start:start+budget]+[eos]);owners.append(row)
 opts=ort.SessionOptions();opts.intra_op_num_threads=8;opts.inter_op_num_threads=1;session=ort.InferenceSession(str(path/'onnx/model_quantized.onnx'),sess_options=opts,providers=['CPUExecutionProvider']);input_names={i.name for i in session.get_inputs()}
 order=np.argsort([len(w) for w in windows],kind='stable');emb=np.zeros((len(texts),dim),np.float32);checkpoint=cache/(cache_name+'.partial.npz');start_index=0
 fingerprint=hashlib.sha256(json.dumps({'texts':texts,'dim':dim,'prefix':prefix,'max_length':max_length},ensure_ascii=False).encode()).hexdigest()
 if checkpoint.exists():
  c=np.load(checkpoint);assert 'fingerprint' in c and str(c['fingerprint'])==fingerprint,'Checkpoint belongs to another corpus/configuration';assert int(c['total'])==len(windows);emb=c['emb'];start_index=int(c['next_index']);print(cache_name,'resume',start_index,flush=True)
 started=time.monotonic()
 for at in range(start_index,len(order),8):
  indices=order[at:at+8];batch=[windows[k] for k in indices];L=max(map(len,batch));x=np.full((len(batch),L),pad,np.int64);mask=np.zeros_like(x)
  for j,seq in enumerate(batch):x[j,:len(seq)]=seq;mask[j,:len(seq)]=1
  feed={'input_ids':x,'attention_mask':mask}
  if 'token_type_ids' in input_names:feed['token_type_ids']=np.zeros_like(x)
  h=session.run(None,feed)[0];assert h.shape[-1]==dim;fm=mask.astype(np.float32);pooled=(h*fm[:,:,None]).sum(1)/fm.sum(1)[:,None];pooled/=np.maximum(np.linalg.norm(pooled,axis=1,keepdims=True),1e-9);np.add.at(emb,np.array(owners)[indices],pooled)
  if at%128==0:
   print(f'{cache_name}: {at:,}/{len(order):,} chunks; {time.monotonic()-started:.0f}s',flush=True)
   np.savez(checkpoint,emb=emb,next_index=at+len(indices),total=len(windows),fingerprint=fingerprint)
 emb/=np.bincount(owners,minlength=len(texts))[:,None];emb/=np.maximum(np.linalg.norm(emb,axis=1,keepdims=True),1e-9);np.save(cache/(cache_name+'.npy'),emb);checkpoint.unlink(missing_ok=True)
 metadata={'dimensions':dim,'max_sequence_length':max_length,'chunk_text_tokens':budget,'prefix':prefix,'prefix_ids':prefix_ids,'bos':bos,'eos':eos,'pad':pad}
 return emb,metadata

assert json.loads((models/'weight-verification.json').read_text())['learned_weights_match']
path=models/'onnx'
emb,meta=encoder(path,1024,[s[:8000] for s in data.text],'e5-descriptions','query: ',512)
lines=(path/'.cache/huggingface/download/onnx/model_quantized.onnx.metadata').read_text().splitlines()
meta.update(onnx_model_id='Xenova/multilingual-e5-large',source_model_id='embaas/sentence-transformers-multilingual-e5-large',revision=lines[0],sha256=lines[1],bytes=(path/'onnx/model_quantized.onnx').stat().st_size)
(cache/'encoder-meta.json').write_text(json.dumps(meta,indent=2))
subprocess.run([sys.executable,str(root/'scripts/fit_forest.py')],check=True)
