import pandas as pd
from scripts.evaluate_studio import make_partitions


def test_new_partitions_only_use_old_holdout_and_keep_same_dates_together():
    frame=pd.DataFrame({'row_id':['train','tuning','a','b','c','d'], 'phase':['train','validation','test','test','test','test'], 'date':['2021-01-01','2021-02-01','2021-03-01','2021-03-02','2021-03-02','2021-03-03']})
    pool,_=make_partitions(frame)
    assert set(pool.row_id)=={'a','b','c','d'}
    assert pool.groupby('date').audit_phase.nunique().max()==1
    assert pool.loc[pool.audit_phase=='validation','date'].max()<pool.loc[pool.audit_phase=='test','date'].min()


def test_published_audit_has_complete_disjoint_partitions_and_honest_scopes():
    import json
    import numpy as np
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    report=json.loads((root/'reports/studio-audit/studio-audit.json').read_text())
    identities=pd.read_csv(root/'reports/studio-audit/audit-split.csv')
    assert report['status']=='complete'
    assert report['counts']=={'validation':1000,'test':1000}
    assert not identities.row_id.duplicated().any()
    assert identities.groupby('audit_phase').size().to_dict()==report['counts']
    expected=json.loads((root/'models/transfer-asking/manifest.json').read_text())['cohort_sha256']
    assert report['data_sha256']==expected
    assert len(report['models'])==6
    for name,model in report['models'].items():
        if not name.startswith('MiniLM'):
            assert 'IN-SAMPLE' in model['evaluation_scope']
        for phase in ['validation','test']:
            result=model[phase]
            assert np.isfinite(result['mae_usd']) and result['mae_usd']>0
            low,high=result['mae_bootstrap_95pct_usd']
            assert low<result['mae_usd']<high
