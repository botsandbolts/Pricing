import json,hashlib
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scripts.train_bid_huber import trim_quarter,run
from pricing.selection_imr import ProbitSelection,inverse_mills
from pricing.catboost_aft_pipeline import CatBoostAFTPipeline


def test_trimming_is_exact_per_phase_and_ties_are_stable():
    ids=np.array(['car-'+str(i).zfill(2) for i in range(24)]);phase=np.repeat(['train','validation','test'],8);r=np.tile([-4.,4.,3.,2.,1.,.5,.2,0.],3)
    keep,counts=trim_quarter(ids,phase,r)
    assert all(v['removed_rows']==2 for v in counts.values())
    assert not keep[[0,1,8,9,16,17]].any() and keep.sum()==18
    _,odd=trim_quarter(ids[:21],np.repeat(['train','validation','test'],7),np.ones(21))
    assert all(v['removed_rows']==2 for v in odd.values())
    reverse=np.arange(23,-1,-1);other,_=trim_quarter(ids[reverse],phase[reverse],r[reverse])
    assert dict(zip(ids,keep))==dict(zip(ids[reverse],other))


def test_probit_requires_non_sales_and_ignores_price_columns():
    from scipy.special import ndtr
    rng=np.random.default_rng(42);n=700;age=rng.uniform(0,15,n)
    x=pd.DataFrame({'age':age,'log_miles':rng.normal(10,1,n),'auction_grade':rng.uniform(1,5,n),'manufacturer':'toyota','type':'sedan','transmission':'automatic','state':'ca','fuel':'gas','price':rng.uniform(100,100000,n)})
    with pytest.raises(ValueError,match='sold AND unsold'):ProbitSelection().fit(x,np.ones(n))
    sold=(rng.random(n)<ndtr((age-7)/3)).astype(int);m=ProbitSelection().fit(x,sold)
    mills=m.predict(x)
    assert np.isfinite(mills).all() and np.corrcoef(age,mills)[0,1]<-.8
    np.testing.assert_array_equal(mills,m.predict(x.assign(price=1e99)))
    extremes=inverse_mills(np.array([-100.,-10.,0.,10.,100.]))
    assert np.isfinite(extremes).all() and np.all(np.diff(extremes)<0)


@pytest.mark.parametrize('use_selection',[False,True])
def test_bid_huber_training_serving_and_independent_default(tmp_path,use_selection):
    n=100;phases=['train']*60+['validation']*20+['test']*20
    d=pd.DataFrame([{'row_id':'bid-'+str(i),'phase':phases[i],'date':'2015-01-10' if i<60 else '2015-02-10' if i<80 else '2015-03-10','year':2000+i%15,'odometer':20000+i*500,'condition':3.,'manufacturer':['ford','toyota'][i%2],'model':['focus','corolla'][i%2],'type':'sedan','transmission':'automatic','state':'ca','price':2000+200*i,'vehicle_group':'vin:'+str(i)} for i in range(n)])
    cohort=tmp_path/'bid.jsonl';d.to_json(cohort,orient='records',lines=True)
    prior=tmp_path/'prior';prior.mkdir();d[['row_id','phase']].to_csv(prior/'filter-rows.csv.gz',index=False)
    (prior/'evaluation.json').write_text(json.dumps({'source_sha256':hashlib.sha256(cohort.read_bytes()).hexdigest()}))
    root=tmp_path/'project';out=root/'models/bid-huber-cleaned';index=Path(__file__).resolve().parents[1]/'models/structural-transfer/used-car-index.csv'
    selection_path=None
    if use_selection:
        unsold=d.iloc[:60].copy();unsold['row_id']=['unsold-'+str(i) for i in range(len(unsold))];unsold['vehicle_group']=['unsold-vin-'+str(i) for i in range(len(unsold))];unsold['price']=np.nan
        population=pd.concat([d.assign(sold=1),unsold.assign(sold=0)],ignore_index=True)
        selection_path=tmp_path/'population.jsonl';population.to_json(selection_path,orient='records',lines=True)
    run(SimpleNamespace(output=str(out),bid=str(cohort),prior=str(prior),index=str(index),selection_population=selection_path,cleaner_iterations=10,iterations=10,delta=.2))
    manifest=json.loads((out/'manifest.json').read_text());audit=pd.read_csv(out/'filter-rows.csv.gz')
    assert manifest['pipeline']=='catboost-log-huber' and manifest['parameters']['loss_function']=='Huber:delta=0.2'
    assert manifest['parameters']['leaf_estimation_method']=='Gradient'
    assert manifest['trained_rows']==75 and 'selection_imr' not in manifest['feature_names']
    assert manifest['selection_imr']['status']==('enabled' if use_selection else 'disabled')
    import gzip
    from catboost import CatBoostRegressor
    cleaner=CatBoostRegressor();cleaner.load_model(blob=gzip.decompress((out/'cleaning-assessment.cbm.gz').read_bytes()))
    assert ('selection_imr' in cleaner.feature_names_)==use_selection
    assert set(audit.loc[audit.phase=='train','oof_fold'])=={0,1,2}
    assert set(audit.loc[audit.phase!='train','oof_fold'])=={-1}
    p=CatBoostAFTPipeline(out);car=d.iloc[0].to_dict();r=p.predict('',car,explain=True)
    assert r['price']>0 and r['explanation']['additivity_error']<1e-5
    assert r['model_name']=='CatBoost log-Huber · delta 0.2'
    from pricing.inference import PricePipeline
    assert PricePipeline(out).predict('',car)['price']==r['price']
    assert abs(np.log(r['price'])-(p.model.predict(p.encoded(pd.DataFrame([car])))[0]+manifest['log_target_offset']))<1e-8
    old=root/'models/minilm-catboost-aft'
    for side in ['ask','bid']:
        (old/side).mkdir(parents=True);(old/side/'manifest.json').write_text(json.dumps({'status':'ready'}))
    from pricing.residual_ui import bid_folder,model_root
    assert model_root(root)==old and bid_folder(root)==out


def test_huber_rejects_invalid_delta_before_any_data_access(tmp_path):
    for delta in [0.,-1.,float('nan'),float('inf')]:
        with pytest.raises(ValueError,match='Huber delta'):
            run(SimpleNamespace(delta=delta,output=str(tmp_path/'unused')))


def test_huber_new_default_takes_priority_without_changing_ask(tmp_path):
    from pricing.residual_ui import bid_folder,model_root
    root=tmp_path
    old=root/'models/minilm-catboost-aft'
    for side in ['ask','bid']:
        (old/side).mkdir(parents=True)
        (old/side/'manifest.json').write_text(json.dumps({'status':'ready'}))
    quantile=root/'models/bid-quantile-cleaned';quantile.mkdir()
    (quantile/'manifest.json').write_text(json.dumps({'status':'ready'}))
    huber=root/'models/bid-huber-cleaned';huber.mkdir()
    (huber/'manifest.json').write_text(json.dumps({'status':'training'}))
    assert bid_folder(root)==quantile
    (huber/'manifest.json').write_text(json.dumps({'status':'ready'}))
    assert bid_folder(root)==huber and model_root(root)==old
