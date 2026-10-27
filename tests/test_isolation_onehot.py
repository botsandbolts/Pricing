import json
import numpy as np
import pandas as pd
from scripts.audit_isolation_onehot import fit_audit


def test_native_onehot_all_levels_and_score_identity(tmp_path):
    rng=np.random.default_rng(4);n=300
    x=pd.DataFrame({'age':rng.normal(8,.2,n),'model':['a']*149+['b']*150+['rare'],'log_reference_price':rng.normal(9,.1,n)})
    x.loc[0,'age']=np.nan
    d=pd.DataFrame({'row_id':list(map(str,range(n))),'phase':['train']*n,'price':np.exp(x.log_reference_price),'model':x.model})
    ot=tmp_path/'ot.csv';pd.DataFrame({'row_id':['299']}).to_csv(ot,index=False)
    r=fit_audit(d,x,tmp_path/'fit',ot)
    assert r['one_hot_categories']['model']==3 and r['encoded_columns']==5
    assert r['settings']['contamination']=='auto' and r['offset']==-.5
    rows=pd.read_csv(tmp_path/'fit/scores.csv.gz',dtype={'row_id':str})
    assert rows.row_id.tolist()==d.row_id.tolist()
    assert np.allclose(rows.decision_function,rows.score_samples+.5)
    assert np.array_equal(rows.flagged,rows.decision_function<0)
    assert r['flagged_rows']==int(rows.flagged.sum())
