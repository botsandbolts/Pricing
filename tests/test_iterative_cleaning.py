import numpy as np
import pytest
from scripts.train_iterative_catboost import trim_stage


def test_five_rounds_are_of_survivors_and_never_resurrect():
    phase=np.repeat(['train','validation','test'],1000)
    ids=[f'{i:05}' for i in range(len(phase))]
    active=np.ones(len(ids),bool)
    for expected in [950,902,856,813,772]:
        before=active.copy()
        active,removed=trim_stage(active,phase,np.arange(len(ids),dtype=float),ids)
        assert not np.any(active & ~before)
        assert np.all((active | removed)==before)
        assert all((active & (phase==s)).sum()==expected for s in set(phase))
    assert active.sum()==2316  # Ceil rounding; asymptotic survivor fraction .95**5.


def test_ties_are_stable_and_held_rows_do_not_change_train_selection():
    phase=np.repeat(['train','validation','test'],40)
    ids=[f'{i:05}' for i in range(len(phase))]
    r=np.ones(len(ids));active=np.ones(len(ids),bool)
    keep,removed=trim_stage(active,phase,r,ids)
    assert np.flatnonzero(removed).tolist()==[0,1,40,41,80,81]
    r[40:]=1e6
    keep2,_=trim_stage(active,phase,r,ids)
    assert np.array_equal(keep[:40],keep2[:40])


def test_incomplete_residuals_fail_closed():
    phase=np.repeat(['train','validation','test'],20)
    r=np.ones(60);r[-1]=np.nan
    with pytest.raises(ValueError,match='residual'):
        trim_stage(np.ones(60,bool),phase,r,[str(i) for i in range(60)])

def test_release_refuses_incomplete_or_missing_model_bundles(tmp_path):
    import json
    from pricing.iterative_release import completed,FOLDERS
    root=tmp_path/'models/iterative-catboost';root.mkdir(parents=True)
    (root/'summary.json').write_text(json.dumps({'status':'complete'}))
    assert not completed(tmp_path)
    for name in FOLDERS:
        f=root/name;f.mkdir()
        (f/'manifest.json').write_text(json.dumps({'status':'ready','outlier_filter':{'stage_count':5},'model_file':'model.gz'}))
        (f/'model.gz').write_bytes(b'fixture')
    assert completed(tmp_path)
    (root/'bid-aft/model.gz').unlink()
    assert not completed(tmp_path)
