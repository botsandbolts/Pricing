import numpy as np
import pytest
from pricing.xgb_imputation import completed_numeric
def test_missing_draws_and_fixed_observations():
    rng=np.random.default_rng(1);context=rng.normal(size=(80,3))
    x=np.column_stack((context[:,0]+rng.normal(size=80),context[:,1]+rng.normal(size=80)))
    x[::5,1]=np.nan
    a=completed_numeric(x,context,2);b=completed_numeric(x,context,3)
    assert np.isfinite(a).all()
    assert np.array_equal(a[np.isfinite(x)],x[np.isfinite(x)])
    assert not np.allclose(a[::5,1],b[::5,1])
def test_all_missing_column_is_rejected():
    with pytest.raises(ValueError):completed_numeric(np.full((20,1),np.nan),np.ones((20,2)),1)
