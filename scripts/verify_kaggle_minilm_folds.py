"""Recover supplied original MiniLM assessment OOF archives with strict identity checks."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    d=pd.concat([c[['row_id','price','phase']] for c in pd.read_json(a.cohort,lines=True,chunksize=5000)],ignore_index=True)
    tr=d.loc[d.phase.eq('train')].reset_index(drop=True)
    if len(tr)!=148290 or tr.row_id.duplicated().any():raise ValueError('Unexpected original training cohort')
    meta=json.loads(Path('models/minilm-assessment/text-model/provenance.json').read_text())
    stamp=hashlib.sha256(json.dumps({'ids':tr.row_id.astype(str).tolist(),'targets':tr.price.to_numpy(float).tolist(),'folds':3,'seed':42,'config':meta['fingerprint']},sort_keys=True).encode()).hexdigest()
    values=np.full(len(tr),np.nan);folds=np.full(len(tr),-1)
    for fold,(_,held) in enumerate(KFold(3,shuffle=True,random_state=42).split(tr)):
        with np.load(Path('data/kaggle-minilm-assessment')/f'fold-{fold}.npz',allow_pickle=False) as z:
            if str(z['fingerprint'])!=stamp:raise ValueError('Saved MiniLM fold fingerprint does not match cohort and installed assessment checkpoint')
            if not np.array_equal(z['held_idx'],held):raise ValueError('MiniLM held-out split differs')
            prices=z['price']
            if len(prices)!=len(held) or not np.isfinite(prices).all() or (prices<=0).any():raise ValueError('Invalid saved prices')
            values[held]=prices;folds[held]=fold
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'row_id':tr.row_id,'fold':folds,'predicted_text_price':values}).to_csv(out/'assessment-oof.csv',index=False)
    report={'status':'verified','training_rows':len(tr),'folds':3,'fold_fingerprint':stamp,'assessment_text_provenance':meta,'cohort_sha256':hashlib.file_digest(open(a.cohort,'rb'),'sha256').hexdigest(),'validation_test_predictions':'not supplied','live_checkpoint':'models/minilm-assessment/text-model','warning':'These original three-fold text features are suitable for training stacking and held-out assessment with matching training-only head. They are not nested within a new five-fold XGBoost partition.'}
    (out/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
