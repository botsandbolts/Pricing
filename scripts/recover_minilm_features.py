"""Reconstruct held-out MiniLM inference from saved Kaggle OOF exports.

No prices enter inference. Original row order and both OOF assignments are checked
by pricing.suite_text before the downstream model is allowed to fit.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from pricing.text import load_text,predict_text


def run(a):
 root=Path(a.features)
 d=pd.concat([c[['row_id','phase','text']] for c in pd.read_json(a.cohort,lines=True,chunksize=5000)],ignore_index=True)
 held=d.loc[d.phase!='train'].copy()
 path=root/'held-predictions.csv'
 done=pd.read_csv(path,dtype={'row_id':str}) if path.exists() else pd.DataFrame(columns=['row_id','predicted_text_price'])
 if done.row_id.tolist()!=held.row_id.iloc[:len(done)].tolist():raise ValueError('Held prediction cache row order mismatch')
 model,tok=load_text(root/'assessment/text-model',a.device);start=time.time()
 for n in range(len(done),len(held),256):
  chunk=held.iloc[n:n+256]
  prices=predict_text(model,tok,chunk.text.fillna('').tolist(),a.device,batch_size=a.batch_size)
  pd.DataFrame({'row_id':chunk.row_id.to_numpy(),'predicted_text_price':prices}).to_csv(path,mode='a',header=not path.exists(),index=False)
  print('HELD',n+len(chunk),'/',len(held),'seconds',round(time.time()-start),flush=True)
 meta={'cohort_sha256':hashlib.file_digest(open(a.cohort,'rb'),'sha256').hexdigest(),'assessment_text_provenance':json.loads((root/'assessment-text-provenance.json').read_text()),'held_rows':len(held),'training_oof_rows':int((d.phase=='train').sum()),'note':'Original Kaggle training OOF and full OOF exports; held inference regenerated from saved training-only MiniLM head without using outcomes.'}
 (root/'provenance.json').write_text(json.dumps(meta,indent=2))
 from pricing.suite_text import validated_features
 f=validated_features(a.cohort,root);print('VERIFIED',len(f),'assessment and',len(f.attrs['full_oof']),'full OOF features',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cohort',required=True);p.add_argument('--features',required=True);p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=16);run(p.parse_args())
