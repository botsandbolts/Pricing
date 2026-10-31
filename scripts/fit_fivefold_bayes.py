"""Calibrate a joint make/model Bayesian residual model from genuine CatBoost OOF rows."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pricing.bayesian_hierarchy import fit, assess, save


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--market',choices=['ask','bid'],required=True)
    p.add_argument('--base-kind',choices=['catboost','xgboost'],default='catboost')
    p.add_argument('--cohort',required=True)
    p.add_argument('--predictions',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    pred=pd.read_csv(a.predictions,dtype={'row_id':str})
    facts=pd.concat((c[['row_id','manufacturer','model','year','date']] for c in pd.read_json(a.cohort,lines=True,chunksize=5000)),ignore_index=True)
    if pred.row_id.duplicated().any() or facts.row_id.duplicated().any():raise ValueError('Duplicate row IDs')
    d=pred.merge(facts,on='row_id',validate='one_to_one')
    if len(d)!=len(pred):raise ValueError('Unmatched predictions')
    d['age']=(pd.to_datetime(d.date,utc=True).dt.year-pd.to_numeric(d.year)).clip(0,150)
    d['base_price']=d.predicted_price
    d['prediction_kind']=np.where(d.split.eq('calibration'),'oof','heldout')
    d['base_fold']=d.fold
    d['base_model_id']=a.base_kind+'-fivefold-'+a.market
    d['price_basis']='historical advertised USD' if a.market=='ask' else 'CPI-reference auction USD'
    calibration=d[d.split.eq('calibration')]
    if calibration.fold.nunique()!=5:raise ValueError('Expected five OOF folds')
    meta,draws=fit(calibration)
    report,_,_=assess(meta,draws,d[d.split.eq('validation')],d[d.split.eq('test')])
    report['market']=a.market
    report['base_test_mae_usd']=float(np.abs(d.loc[d.split.eq('test'),'observed_price']-d.loc[d.split.eq('test'),'base_price']).mean())
    report['model']=a.base_kind+' five-fold OOF + joint Bayesian make/model/age log-residual correction'
    save(Path(a.output),meta,draws,report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
