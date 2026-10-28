"""Validate separate assessment and deployment text features before any refit."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def validated_features(cohort,path,*,full_only=False):
    path=Path(path);meta=json.loads((path/'provenance.json').read_text())
    digest=hashlib.file_digest(open(cohort,'rb'),'sha256').hexdigest()
    if meta['cohort_sha256']!=digest:raise ValueError('Text predictions belong to different cohort')
    rows=pd.concat([c[['row_id','phase']] for c in pd.read_json(cohort,lines=True,chunksize=5000)],ignore_index=True)
    train=rows.loc[rows.phase=='train','row_id'].tolist();held=rows.loc[rows.phase!='train','row_id'].tolist()
    p=meta['assessment_text_provenance']
    if p['fitting_rows']!=len(train):raise ValueError('Assessment text head was not trained on training partition only')
    def read(name,ids,folds):
        d=pd.read_csv(path/name,dtype={'row_id':str})
        if d.row_id.tolist()!=ids or d.row_id.duplicated().any():raise ValueError('Text prediction identity/order mismatch: '+name)
        prices=d.predicted_text_price.to_numpy(float)
        if not np.isfinite(prices).all() or (prices<0).any():raise ValueError('Invalid text price')
        if folds:
            expected=np.empty(len(ids),int)
            for number,(_,idx) in enumerate(KFold(3,shuffle=True,random_state=42).split(ids)):expected[idx]=number
            if not np.array_equal(expected,d.fold.to_numpy()):raise ValueError('OOF fold assignment mismatch')
        return pd.Series(prices,index=ids)
    full=read('full-oof.csv',rows.row_id.tolist(),True)
    fp=json.loads((path/'full-text-provenance.json').read_text())
    if fp['fitting_rows']!=len(rows) or fp['fingerprint']!=p['fingerprint'] or fp['revision']!=p['revision']:
        raise ValueError('Full and assessment text model provenance differ')
    if full_only:return full
    tr=read('assessment-oof.csv',train,True);ht=read('held-predictions.csv',held,False)
    result=pd.concat([tr,ht]);result.attrs['full_oof']=full
    return result
