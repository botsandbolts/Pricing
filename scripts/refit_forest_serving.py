"""Reproduce calibration-only RF refits and verify against published held-out outputs."""
import json,lzma,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from pricing.model_suite import load_cohort,predictors
from marketbridge.structural import IndexPanel,known_pricing_vins
from pricing.forest_residuals import CATEGORICAL
from pricing.suite_storage import split_model


def run(root=Path('.')):
    known=known_pricing_vins(root/'training-cache/full-cohort.jsonl');panel=IndexPanel(root/'models/structural-transfer/used-car-index.csv')
    for side,name in [('ask','full-cohort'),('bid','bid-cohort')]:
        folder=root/'models/forest-original'/side;meta=json.loads((folder/'base-assessment.json').read_text());provenance=json.loads((folder/'provenance.json').read_text())
        d=load_cohort(root/f'training-cache/{name}.jsonl');x,valid,dates=predictors(d,side);ratio,_=panel.ratios(dates)
        if side=='bid':valid&=ratio.notna()&~d.vehicle_group.astype(str).str.removeprefix('vin:').isin(known)
        assert int(valid.sum())==provenance['eligible_rows']
        train=valid&d.phase.eq('train');x=x.loc[train,meta['features']].copy()
        y=np.log(d.loc[train,'price'].to_numpy(float)/(ratio.loc[train].to_numpy(float) if side=='bid' else 1.))
        for c in CATEGORICAL:
            levels=sorted(x[c].unique().tolist());assert levels==meta['category_schema'][c],c
            x[c]=pd.Categorical(x[c],categories=levels)
        params=meta['parameters'].copy();params['n_estimators']=1;params['num_parallel_tree']=200
        print('Fitting live',side,len(x),flush=True);model=XGBRegressor(**params).fit(x,y)
        examples=pd.read_csv(folder/'examples.csv.gz');check=pd.DataFrame({'age':examples.age,'log_miles':np.log1p(examples.odometer),'auction_grade':examples.auction_grade})
        for c in CATEGORICAL:check[c]=pd.Categorical(examples[c].fillna('unknown').astype(str).str.strip().str.lower(),categories=meta['category_schema'][c])
        prediction=np.exp(np.clip(model.predict(check[meta['features']]),0,np.log(1e7)))
        np.testing.assert_allclose(prediction,examples.base_price,rtol=2e-5,atol=.02)
        print(side,'all 1000 published predictions reproduced',flush=True)
        raw=bytes(model.get_booster().save_raw(raw_format='ubj'));path=folder/'serving.ubj.xz'
        path.write_bytes(lzma.compress(raw,preset=6))
        (folder/'serving.json').write_text(json.dumps({'rows':len(x),'verification':'All 1000 published test examples reproduced within 0.002% / $0.02 tolerance. Calibration-only refit, held-out targets excluded.','max_abs_prediction_difference':float(np.max(abs(prediction-examples.base_price))),'raw_bytes':len(raw),'compressed_bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},indent=2))
        print(side,'packed',path.stat().st_size,flush=True)
        if path.stat().st_size>500000:split_model(path,500000)

if __name__=='__main__':run()
