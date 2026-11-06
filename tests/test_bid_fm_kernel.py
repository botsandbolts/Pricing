"""Contract and gradient tests for the full-data bid FM/FFM and RFF challenge."""
import numpy as np
import pytest
import torch

from scripts.bid_fm_kernel_train import (
    TRIALS, AuctionFactorization, NUMERIC_COUNT, FIELD_COUNT,
    _scores, y_price, partition, rff_features)
from scripts.bid_fm_kernel_report import ALPHAS,validation_frame


def test_prespecified_trials_and_four_hour_bounded_design():
    assert len(TRIALS)==10
    assert sum(f=="fm" for f,_,_ in TRIALS.values())==4
    assert sum(f=="ffm" for f,_,_ in TRIALS.values())==2
    assert sum(f.startswith("rff") for f,_,_ in TRIALS.values())==4
    assert set(cohort for _,_,cohort in TRIALS.values())=={"full","lemon"}
    assert ALPHAS[0]==0. and ALPHAS[-1]==1.


@pytest.mark.parametrize("field_aware", [False,True])
def test_factorization_network_learns_from_real_features(field_aware):
    torch.manual_seed(42)
    model=AuctionFactorization([8,9,5,7,6,8,5,6,9,5],rank=8,
                               field_aware=field_aware,initial_log_price=9.5)
    assert NUMERIC_COUNT==5 and FIELD_COUNT==15
    n=torch.randn(6,5)
    c=torch.randint(1,5,(6,10))
    out=model(n,c)
    assert out.shape==(6,) and torch.isfinite(out).all()
    loss=torch.nn.functional.smooth_l1_loss(out,torch.ones(6)*9.0,beta=.25)
    loss.backward()
    assert model.num_latent.grad is not None
    assert torch.isfinite(model.num_latent.grad).all()
    assert any(t.weight.grad is not None for t in model.cat_latent)
    assert model.cat_latent[0].weight.grad.shape[1]==8*(15 if field_aware else 1)


def test_continuous_nonlinear_main_effect_and_unknown_categories():
    model=AuctionFactorization([7]*10,rank=8,initial_log_price=9.)
    with torch.no_grad():
        model.num_spline[0,0]=.75
    n=torch.zeros((2,5))
    n[1,0]=3.
    c=torch.ones((2,10),dtype=torch.long)
    y=model(n,c)
    assert torch.isfinite(y).all() and not torch.allclose(y[0],y[1])


def test_score_and_safe_price_geometry():
    assert np.allclose(y_price(np.log([1000.,2000.])),[1000.,2000.])
    s=_scores([1000,2000],[900,2100])
    assert s["mae_usd"]==100. and s["n"]==2
    with pytest.raises(ValueError):
        _scores([100.],[float("nan")])


def test_training_cohorts_preserve_original_unfiltered_holdouts():
    total=550181
    phase=np.array(["train"]*384198+["validation"]*82380+["test"]*83603)
    lemon=np.ones(total,bool)
    lemon[:96050]=False
    data=dict(phase=phase,lemon=lemon)
    fit,inner,val,test=partition(data,"full")
    assert len(fit)+len(inner)==384198
    assert len(val)==82380 and len(test)==83603
    assert np.intersect1d(fit,inner).size==0
    lfit,linner,lval,ltest=partition(data,"lemon")
    assert len(lfit)+len(linner)==288148
    assert np.array_equal(lval,val) and np.array_equal(ltest,test)


def test_val_alignment_rejects_missing_rows_and_changed_target():
    import pandas as pd
    baseline=pd.DataFrame({
        "row_id":["a","b"],"split":["validation","validation"],
        "observed_price":[100.,200.],"predicted_price":[105.,205.]}).set_index("row_id")
    shuffled=pd.DataFrame({
        "row_id":["b","a"],"split":["validation","validation"],
        "observed_price":[200.,100.],"predicted_price":[199.,99.]})
    found=validation_frame(shuffled,baseline)
    assert found.index.tolist()==["a","b"]
    with pytest.raises(ValueError):
        validation_frame(shuffled.iloc[:1],baseline)
    wrong=shuffled.copy();wrong.loc[0,"observed_price"]=9999.
    with pytest.raises(ValueError):
        validation_frame(wrong,baseline)


def test_rff_uses_sparse_onehot_and_raw_numeric_without_price():
    from sklearn.preprocessing import OneHotEncoder,StandardScaler
    from sklearn.kernel_approximation import RBFSampler
    import scipy.sparse as sp
    rng=np.random.default_rng(123)
    nums=rng.normal(size=(30,5)).astype(np.float32)
    cat=rng.integers(1,4,size=(30,10))
    onehot=OneHotEncoder(sparse_output=True,handle_unknown="ignore",dtype=np.float32).fit(cat[:20])
    scaler=StandardScaler().fit(nums[:20])
    x=sp.hstack([sp.csr_matrix(scaler.transform(nums[:20])/np.sqrt(5)),
                 onehot.transform(cat[:20])/np.sqrt(10)],format="csr")
    feature=RBFSampler(gamma=.5,n_components=64,random_state=42).fit(x)
    projected=rff_features(nums,cat,onehot,scaler,feature,np.arange(20,30))
    assert projected.shape==(10,64) and np.isfinite(projected).all()
