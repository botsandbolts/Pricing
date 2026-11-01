import json,sqlite3,zlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pricing.taste_ranking import CATEGORICAL,encode,features,ndcg10,train_rank

def sample():
    rows=[]
    for i in range(400):
        c=i%10
        rows.append(dict(manufacturer='make'+str(c),model='model'+str(c),type='suv' if c%2 else 'sedan',year=2000+c,odometer=10000+c*1000+i,fuel='gas',price=1000+i,cluster=c,user='excluded',description='excluded'))
    return pd.DataFrame(rows)

def test_vehicle_only_native_categories():
    frame=sample();codes,nums,levels=encode(frame);x=features(codes,nums,levels)
    assert all(str(x[c].dtype)=='category' for c in CATEGORICAL)
    assert not {'price','description','cluster','user','qid'}.intersection(x.columns)
    changed=frame.copy();changed.price*=100;changed.user='another';a,b,l=encode(changed)
    assert np.array_equal(a,codes) and np.array_equal(b,nums) and l==levels

def test_ndcg_matches_exponential_gain_definition():
    assert ndcg10([3,2,1,0],[4,3,2,1])==pytest.approx(1)
    assert 0<ndcg10([3,2,1,0],[1,2,3,4])<1
    assert ndcg10([0,0],[2,1]) is None

def test_ranker_learns_without_duplicate_or_user_rows():
    frame=sample();codes,nums,levels=encode(frame);clusters=np.arange(len(frame))%10
    validation=np.random.default_rng(42).random(len(frame))<.2
    result=train_rank(codes,nums,levels,clusters,[1,2,3,4,1,2,3,4,2,3],list(range(10)),validation,rounds=20,prices=frame.price.to_numpy(),adjusted_prices=frame.price.to_numpy()*.8)
    assert result['final_rows']==400 and result['direct_judgments']==10
    assert len(set(result['positions']))==10
    assert not set(result['positions']).intersection(range(10))
    assert 0<=result['ndcg10']<=1
    assert not {'cluster','user','qid'}.intersection(result['feature_importance'])
    assert 'log_advertised_price' in result['feature_importance'] and 'log_date_adjusted_price' in result['feature_importance']
    with pytest.raises(ValueError):train_rank(codes,nums,levels,clusters,[2]*10,list(range(10)),validation,rounds=3)

def test_hdbscan_real_prototypes_and_noise_are_distinguished():
    from scripts.build_taste_garage import density_profiles
    rng=np.random.default_rng(42)
    centres=rng.normal(0,30,(12,5));points=np.concatenate([rng.normal(c,.08,(80,5)) for c in centres])
    clusters,noise,prototypes,attempts,native=density_profiles(points)
    assert len(prototypes)==len(set(prototypes))==10
    assert set(clusters)==set(range(10))
    assert all(not noise[p] for p in prototypes)
    assert native>=10
    assert all(clusters[p]==i for i,p in enumerate(prototypes))

def test_rating_ui_and_result_staleness(monkeypatch,tmp_path):
    import pricing.taste_garage_ui as ui
    from streamlit.testing.v1 import AppTest
    folder=tmp_path/'data/taste-garage';folder.mkdir(parents=True);(folder/'manifest.json').write_text('{}')
    car=dict(year=2015,manufacturer='Ford',model='Focus',price=10000,observed_date='2021-05-01',odometer=50000)
    meta={'rows':400,'density_clusters':12,'noise_fraction':.1,'profiles':[{'prototype':car,'position':i,'name':f'Profile {i}','cars':40} for i in range(10)]}
    monkeypatch.setattr(ui,'catalogue',lambda *a:(meta,{}));monkeypatch.setattr(ui,'price_view',lambda *a:8000.)
    at=AppTest.from_string(f"from pathlib import Path\nfrom pricing.taste_garage_ui import render\nrender(Path({str(tmp_path)!r}))").run(timeout=30)
    assert not at.exception
    assert next(b for b in at.button if b.label=='Find my ten cars →').disabled
    radios=[r for r in at.radio if r.label=='Your reaction'];assert len(radios)==10
    for r in radios:r.set_value(2)
    at.run();assert not at.exception
    assert next(b for b in at.button if b.label=='Find my ten cars →').disabled


def test_adjusted_price_features_allow_native_categories_and_keep_missing_facts():
    from datetime import date
    from pricing.taste_ranking import adjusted_price_features
    frame=sample();codes,nums,levels=encode(frame)
    class Curve:
        def ratio(self,frame,*args,**kwargs):
            frame.manufacturer.fillna('unknown')
            return np.full(len(frame),.5)
    class Clock:
        def value(self,*args,**kwargs):return 200.,{}
        def reference(self,*args,**kwargs):return 100.,'2021-05'
    class Transport:
        curves={'ask':Curve()};clock=Clock()
    days=np.full(len(frame),np.datetime64('2021-05-01').astype(int))
    nums[0,2]=1
    price=frame.price.to_numpy(dtype=float)
    result=adjusted_price_features(codes,nums,levels,price,days,date(2026,11,1),Transport())
    assert np.isnan(result[0])
    assert result[1:]==pytest.approx(price[1:])
    assert price[0]==1000


def test_leave_one_out_excludes_own_target_and_uses_dense_numeric_geometry():
    from pricing.taste_ranking import leave_one_out_geometry
    frame=sample().iloc[:20].copy();frame['manufacturer']=['unique']+['shared']*19
    prices=frame.price.to_numpy(dtype=float)
    geometry,a=leave_one_out_geometry(frame,prices)
    changed=prices.copy();changed[0]=999999
    _,b=leave_one_out_geometry(frame,changed)
    assert a[0]==pytest.approx(b[0]) # neither category sum nor prior includes this row
    assert geometry.shape==(20,len(CATEGORICAL)+3) and np.isfinite(geometry).all()
    assert a[0,0]==pytest.approx(np.log(prices[1:]).mean())
    assert not np.allclose(a[1:],b[1:])
