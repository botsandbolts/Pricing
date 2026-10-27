import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
pytest.importorskip('outliertree')
from scripts.audit_outliertree import stage, run


def test_native_categories_and_row_identity(tmp_path):
    rng=np.random.default_rng(12);n=300
    x=pd.DataFrame({'age':rng.normal(8,.3,n),'log_miles':np.r_[rng.normal(10,.02,n-1),20.],
        'manufacturer':pd.Categorical(['toyota']*150+['ford']*150),
        'model':pd.Categorical(['a']*150+['b']*150),'log_reference_price':rng.normal(9,.05,n)})
    d=pd.DataFrame({'row_id':list(map(str,range(n))),'phase':['train']*n,
        'price':np.exp(x.log_reference_price),'manufacturer':x.manufacturer,'model':x.model})
    r=stage(d,x,tmp_path,.099,2)
    assert set(r['native_categorical_columns'])=={'manufacturer','model'}
    assert r['settings']['pct_outliers']==.099 and r['cars_deleted']==0
    flags=pd.read_csv(tmp_path/'flagged-cars.csv.gz',dtype={'row_id':str})
    assert '299' in flags.row_id.tolist()
    assert json.loads(flags.set_index('row_id').loc['299','suspicious_value'])['column']=='log_miles'


def test_unsupported_percentage_rejected_before_reading_data():
    with pytest.raises(ValueError,match='unsupported'):
        run(SimpleNamespace(pct_outliers=.25))
