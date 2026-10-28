import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from marketbridge.depreciation import basis,DepreciationCurve,AGE_KNOTS,MILE_KNOTS
from marketbridge.market_clock import MarketClock,forecast
from marketbridge.time_engine import TimeTransport,age_on


def test_monotone_basis_and_serialized_categorical_curve(tmp_path):
    grid=np.linspace(0,40,500)
    for kind in ['linear','piecewise','spline']:
        b=basis(grid,AGE_KNOTS,kind)
        assert np.isfinite(b).all() and np.min(np.diff(b,axis=0))>=-1e-12
    rng=np.random.default_rng(5);n=1500;age=rng.uniform(0,20,n);make=np.where(np.arange(n)%2,'toyota','ford')
    f=pd.DataFrame({'age':age,'odometer':rng.uniform(1000,200000,n),'manufacturer':make,'model':'example','type':'sedan'})
    y=10-.025*age-.025*age*(make=='ford')-.1*np.log1p(f.odometer/10000)+rng.normal(0,.03,n)
    m=DepreciationCurve('piecewise',10).fit(f,y);file=tmp_path/'curve.json';m.save(file);loaded=DepreciationCurve.load(file)
    np.testing.assert_allclose(m.predict_log(f),loaded.predict_log(f),rtol=0,atol=1e-12)
    test=pd.DataFrame([{'age':a,'odometer':50000,'manufacturer':'toyota','model':'example','type':'sedan'} for a in range(31)])
    assert np.max(np.diff(m.predict_log(test)))<=1e-10
    assert np.max(np.diff(m.predict_log(test.assign(age=10,odometer=np.arange(31)*10000))))<=1e-10
    assert np.isfinite(loaded.predict_log(test.assign(manufacturer='unseen',model='new'))).all()
    ratios=m.ratio(pd.DataFrame([{'age':5,'odometer':50000,'manufacturer':'toyota','model':'example','type':'sedan'},{'age':5,'odometer':50000,'manufacturer':'ford','model':'example','type':'sedan'}]),5,15)
    assert ratios[1]<ratios[0]


def clock_fixture(tmp_path):
    months=pd.period_range('2013-01','2026-09',freq='M');pd.DataFrame({'month':months.astype(str),'value':np.linspace(100,180,len(months))}).to_csv(tmp_path/'market-index.csv',index=False)
    (tmp_path/'market-index-meta.json').write_text(json.dumps({'latest_observation':'2026-09','latest_release_date':'2026-10-14'}));(tmp_path/'market-backtest.json').write_text(json.dumps({'selected':'last_value'}));return MarketClock(tmp_path)


def test_market_publication_cutoff_and_forecast_limit(tmp_path):
    clock=clock_fixture(tmp_path)
    _,before=clock.value('2026-10-01',as_of='2026-10-01');assert before['last_known_month']=='2026-08' and before['horizon']==2
    _,after=clock.value('2026-10-28');assert after['last_known_month']=='2026-09' and after['horizon']==1
    _,history=clock.value('2021-05-01',as_of='2026-10-28');assert history['kind']=='observed_monthly_index'
    with pytest.raises(ValueError,match='six months'):clock.value('2027-07-01')
    for method in ['last_value','seasonal_naive','damped_holt','ar_1_12']:
        pred=forecast(np.exp(np.linspace(4,4.3,100)),6,method);assert pred.shape==(6,) and np.isfinite(pred).all() and (pred>0).all()


def test_calendar_age_and_new_generation_guards():
    t=TimeTransport.__new__(TimeTransport)
    with pytest.raises(ValueError,match='required'):t.validate({'year':None,'odometer':None},'2026-10-28')
    assert age_on(2014,'2026-10-28')>12 and age_on(2014,'2015-01-01')==1
    with pytest.raises(ValueError,match='after 2021'):t.validate({'year':2022,'odometer':5000},'2026-10-28')
    with pytest.raises(ValueError,match='0–30'):t.validate({'year':1990,'odometer':50000},'2026-10-28')
