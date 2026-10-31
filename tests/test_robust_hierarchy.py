import numpy as np
import pandas as pd
import pytest
from pricing import robust_hierarchy as rh

def cohort(n=1200):
    rng=np.random.default_rng(514);age=rng.uniform(1,23,n);miles=rng.uniform(10000,230000,n);base=rng.uniform(4000,35000,n)
    offset=.12+.03*np.maximum(age-10,0)+.18*(miles>140000)
    return pd.DataFrame(dict(row_id=[str(i) for i in range(n)],vehicle_group=[str(i) for i in range(n)],age=age,odometer=miles,base_price=base,
        observed_price=base*np.exp(offset+rng.standard_t(4,n)*.06),manufacturer=np.where(np.arange(n)%2,'ford','toyota'),model=np.where(np.arange(n)%3,'a','b'),
        condition='good',prediction_kind='oof',base_fold=np.arange(n)%5,base_model_id='test',price_basis='usd'))

def test_heldout_recovery_target_independence_and_unknown_groups():
    d=cohort();meta,draws=rh.fit(d.iloc[:900],'student_t',warmup=60,draws=60);held=d.iloc[900:];p=rh.predict(meta,draws,held)
    assert np.abs(held.observed_price-p.corrected_price).mean()<.6*np.abs(held.observed_price-held.base_price).mean()
    changed=held.copy();changed['observed_price']*=10;pd.testing.assert_frame_equal(p,rh.predict(meta,draws,changed))
    unseen=held.iloc[:1].copy();unseen['manufacturer']='new';unseen['model']='new';u=rh.predict(meta,draws,unseen).iloc[0]
    assert u.model_n==0 and u.make_n==0 and np.isfinite(u.corrected_price) and u.predictive_sd_log>0
    from pricing.bayesian_hierarchy import predict
    pd.testing.assert_frame_equal(p,predict(meta,draws,held))

@pytest.mark.parametrize('likelihood',['gaussian','student_t','mixture'])
def test_likelihood_density_and_artifact_roundtrip(tmp_path,likelihood):
    d=cohort(400);meta,arrays=rh.fit(d.iloc[:300],likelihood,warmup=20,draws=20);held=d.iloc[300:]
    density=rh.pointwise_log_predictive(meta,arrays,held);assert len(density)==len(held) and np.isfinite(density).all()
    from pricing.bayesian_hierarchy import save,load,predict,interval
    save(tmp_path,meta,arrays,{'validation_multiplier':1.6});m,a=load(tmp_path);result=predict(m,a,held);low,high=interval(result,1.6)
    assert np.all(low<result.corrected_price) and np.all(high>result.corrected_price)
    pd.testing.assert_frame_equal(rh.predict(meta,arrays,held),result)

def test_selection_and_calibration_keep_vehicle_groups_disjoint():
    d=cohort(100);d['vehicle_group']=[str(i//2) for i in range(len(d))];s,c=rh.split_validation(d)
    assert len(s)+len(c)==len(d) and set(s.vehicle_group).isdisjoint(c.vehicle_group)
    other=d.copy();other['observed_price']=1;assert rh.split_validation(other)[0].row_id.tolist()==s.row_id.tolist()

def test_rejects_in_sample_calibration():
    d=cohort(100);d['prediction_kind']='heldout'
    with pytest.raises(ValueError,match='OOF'):rh.fit(d)

def test_stratified_ranges_widen_for_rare_cars_without_changing_points():
    d=cohort(900);meta,arrays=rh.fit(d.iloc[:500],'student_t',warmup=20,draws=20);v=d.iloc[500:].copy()
    v.loc[v.index[:200],'model']='unseen';v.loc[v.index[:200],'observed_price']*=2
    before=rh.predict(meta,arrays,v);m,multiplier=rh.calibrate_intervals(meta,arrays,v);after=rh.predict(m,arrays,v)
    np.testing.assert_array_equal(before.corrected_price,after.corrected_price);rare=after.model_n.to_numpy()==0
    assert np.median(after.loc[rare,'interval_multiplier'])>np.median(after.loc[~rare,'interval_multiplier'])
    from pricing.bayesian_hierarchy import interval
    _,hi=interval(after,multiplier);np.testing.assert_allclose(hi,after.corrected_price.to_numpy()*np.exp(after.predictive_sd_log.to_numpy()*after.interval_multiplier.to_numpy()))
