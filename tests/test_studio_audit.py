import pandas as pd
from scripts.evaluate_studio import make_partitions


def test_new_partitions_only_use_old_holdout_and_keep_same_dates_together():
    frame=pd.DataFrame({'row_id':['train','tuning','a','b','c','d'], 'phase':['train','validation','test','test','test','test'], 'date':['2021-01-01','2021-02-01','2021-03-01','2021-03-02','2021-03-02','2021-03-03']})
    pool,_=make_partitions(frame)
    assert set(pool.row_id)=={'a','b','c','d'}
    assert pool.groupby('date').audit_phase.nunique().max()==1
    assert pool.loc[pool.audit_phase=='validation','date'].max()<pool.loc[pool.audit_phase=='test','date'].min()
