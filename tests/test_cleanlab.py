"""Real Cleanlab audits; legacy trained artifact metadata must stay unchanged."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pricing.label_quality import cleanlab_inliers


def data(n=250):
    rng=np.random.default_rng(5)
    x=pd.DataFrame({'year':rng.integers(2000,2020,n),'odometer':rng.integers(10000,200000,n),
                    'manufacturer':np.where(np.arange(n)%2,'toyota','honda')})
    y=np.exp(8+.12*(x.year-2000)-.000003*x.odometer+rng.normal(0,.05,n)).to_numpy(copy=True)
    y[:10]*=15
    return x,y


def test_actual_cleanlab_flags_corrupted_labels_and_resumes_exact_audit(tmp_path):
    x,y=data();ids=[str(i) for i in range(len(y))]
    keep,report=cleanlab_inliers(x,y,tmp_path,ids,threads=1,n_boot=1,trees=40)
    assert len(set(range(10))-set(keep))>=8
    assert report['algorithm']=='Cleanlab' and report['settings']['n_boot']==1
    assert report['retained_rows']+report['removed_rows']==len(y)
    assert 'predicted_text_price' not in report['features']
    again,other=cleanlab_inliers(x,y,tmp_path,ids,threads=1,n_boot=1,trees=40)
    np.testing.assert_array_equal(keep,again);assert report==other
    with pytest.raises(ValueError,match='different data'):
        cleanlab_inliers(x,y+100,tmp_path,ids,threads=1,n_boot=1,trees=40)
    audit=pd.read_csv(tmp_path/'label-audit.csv')
    assert len(audit)==len(y) and audit.label_quality.between(0,1).all()
    assert audit.kept.sum()==len(keep)


@pytest.mark.parametrize('column',['price','sellingprice','text','predicted_text_price','phase'])
def test_target_and_stacking_leakage_are_rejected(tmp_path,column):
    x,y=data(40);x[column]=y
    with pytest.raises(ValueError,match='allowlist'):
        cleanlab_inliers(x,y,tmp_path,np.arange(len(y)))


def test_duplicate_identities_and_invalid_labels_are_rejected(tmp_path):
    x,y=data(40)
    with pytest.raises(ValueError,match='Unique row'):
        cleanlab_inliers(x,y,tmp_path,['same']*len(y))
    y[0]=np.nan
    with pytest.raises(ValueError,match='positive finite'):
        cleanlab_inliers(x,y,tmp_path,np.arange(len(y)))


def test_asking_and_auction_trainers_use_shared_train_only_audit():
    for name in ['pricing','carbids']:
        source=(Path(name)/'train.py').read_text()
        assert 'IsolationForest' not in source
        assert "d.iloc[tr][NUMERIC+CATEGORICAL],d.price.iloc[tr]" in source
        assert "d[NUMERIC+CATEGORICAL],d.price" in source
        assert "'outlier_filter':refit_filter" in source


def test_cleanlab_ensemble_selection_never_loads_pending_bundle(tmp_path):
    from marketbridge.engine import best_asking_bundle
    path=tmp_path/'models/robust-asking-cleanlab';path.mkdir(parents=True)
    for status,promoted in [('pending',True),('ready',False)]:
        (path/'manifest.json').write_text(json.dumps({'status':status,'promoted':promoted}))
        assert best_asking_bundle(tmp_path)==tmp_path/'models/robust-asking'
    (path/'manifest.json').write_text(json.dumps({'status':'ready','promoted':True}))
    assert best_asking_bundle(tmp_path)==path


def test_cleanlab_structural_training_keeps_validation_and_test_intact(tmp_path,monkeypatch):
    from marketbridge import structural
    from tests.test_pipeline import frame
    d=frame(60);d.year=2005
    d.date=pd.date_range('2014-03-01',periods=60,freq='h',tz='UTC')
    d['vehicle_group']=['vehicle:'+str(i) for i in range(60)]
    source=tmp_path/'sales.jsonl';d.to_json(source,orient='records',lines=True,date_format='iso')
    seen=[]
    from pricing import label_quality
    original=label_quality.cleanlab_inliers
    def audit(x,y,path,ids,*args,**kwargs):
        seen.append(list(ids));return original(x,y,path,ids,threads=1,trees=25,n_boot=1)
    monkeypatch.setattr(label_quality,'cleanlab_inliers',audit)
    fit=structural.fit_model
    monkeypatch.setattr(structural,'fit_model',lambda data,rounds,alpha:fit(data,5,alpha))
    root=Path(__file__).resolve().parents[1]
    output=tmp_path/'structural'
    report=structural.train(source,root/'models/structural-transfer/used-car-index.csv',output)
    assert set(map(str,seen[0]))==set(d.row_id[d.phase=='train'])
    assert set(map(str,seen[1]))==set(d.row_id)
    assert report['validation_rows']==9 and report['test_rows']==9
    manifest=json.loads((output/'manifest.json').read_text())
    assert manifest['outlier_filter']['algorithm']=='Cleanlab'
    assert manifest['trained_rows']==manifest['outlier_filter']['retained_rows']


def test_cleanlab_ensemble_training_keeps_validation_and_test_intact(tmp_path,monkeypatch):
    from marketbridge import robust
    from tests.test_pipeline import frame
    d=frame(60);source=tmp_path/'asking.jsonl';d.to_json(source,orient='records',lines=True,date_format='iso')
    seen=[]
    from pricing import label_quality
    original=label_quality.cleanlab_inliers
    def audit(x,y,path,ids,*args,**kwargs):
        seen.append(list(ids));return original(x,y,path,ids,threads=1,trees=25,n_boot=1)
    monkeypatch.setattr(label_quality,'cleanlab_inliers',audit)
    fit=robust.fit_one
    monkeypatch.setattr(robust,'fit_one',lambda x,y,c,rounds=3500,**kwargs:fit(x,y,c,rounds=5,**kwargs))
    monkeypatch.setattr(robust,'CONFIGS',robust.CONFIGS[:3])
    root=Path(__file__).resolve().parents[1];output=tmp_path/'asking'
    report=robust.train(source,root/'models/structural-transfer',output)
    assert set(map(str,seen[0]))==set(d.row_id[d.phase=='train'])
    assert set(map(str,seen[1]))==set(d.row_id)
    assert report['validation_rows']==9 and report['test_rows']==9
    manifest=json.loads((output/'manifest.json').read_text())
    assert manifest['outlier_filter']['algorithm']=='Cleanlab'
    assert manifest['trained_rows']==manifest['outlier_filter']['retained_rows']


def test_cleanlab_preserves_callers_numpy_rng(tmp_path):
    x,y=data(40)
    np.random.seed(987);expected=np.random.random(3)
    np.random.seed(987)
    cleanlab_inliers(x,y,tmp_path,np.arange(len(y)),threads=1,n_boot=0,trees=10)
    np.testing.assert_array_equal(np.random.random(3),expected)


def test_kaggle_archives_include_current_cleanlab_trainers():
    import base64,hashlib,io,re,zipfile
    for file in Path('.').glob('*kaggle.ipynb'):
        for cell in json.loads(file.read_text())['cells']:
            source=''.join(cell['source'])
            match=re.search(r"SOURCE_ZIP\s*=\s*'([^']+)'",source)
            if not match:continue
            payload=base64.b64decode(match.group(1))
            digest=re.search(r"SOURCE_SHA(?:256)?\s*=\s*'([a-f0-9]{64})'",source).group(1)
            assert hashlib.sha256(payload).hexdigest()==digest
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                for name in ['pricing/label_quality.py','pricing/train.py','carbids/train.py','marketbridge/robust.py','marketbridge/structural.py']:
                    if name in archive.namelist():assert archive.read(name)==Path(name).read_bytes()
