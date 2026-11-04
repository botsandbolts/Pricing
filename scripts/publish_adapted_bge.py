"""Select UMAP on validation, assemble a complete serving bundle and smoke-test it."""
import argparse,hashlib,json,os,shutil
from pathlib import Path
import numpy as np

def choose(root):
    entries=[]
    for folder in sorted(Path(root).glob('adapted-config-*')):
        report=json.loads((folder/'ask/bayesian/bayesian-evaluation.json').read_text())
        selected=next(c for c in report['candidates'] if c['likelihood']==report['selected_likelihood'])
        if selected['max_split_rhat']>=1.1:raise ValueError('Unconverged selected hierarchy')
        if not np.isfinite(selected['validation_mae_usd']):raise ValueError('Invalid validation score')
        features=json.loads((folder/'ask/features.json').read_text())
        if features['n_epochs']!=500 or features['metric']!='cosine':raise ValueError('Wrong UMAP settings')
        entries.append({'configuration':folder.name,'dimensions':features['umap_dimensions'],
            'n_neighbors':features['n_neighbors'],'min_dist':features['min_dist'],'n_epochs':500,
            'selected_likelihood':report['selected_likelihood'],
            'validation_mae_usd':selected['validation_mae_usd'],
            'test_mae_usd':report['point_scores']['mae_usd'],'test_coverage':report['test_coverage'],
            'encoder_identity':features['encoder_bundle']['identity']})
    expected={(d,n,m) for d in (32,64) for n in (30,100) for m in (.1,.3)}
    if {(r['dimensions'],r['n_neighbors'],r['min_dist']) for r in entries}!=expected or len(entries)!=8:
        raise ValueError('All eight UMAP configurations are required')
    if len({r['encoder_identity'] for r in entries})!=1:raise ValueError('Different adapted checkpoints')
    # The test columns above are descriptive only: selection uses validation MAE exclusively.
    best=min(entries,key=lambda r:(r['validation_mae_usd'],r['configuration']))
    return best,entries

def build(root='adapted-results',output='release-stage'):
    from pricing.suite_storage import split_model
    from pricing.bayesian_hierarchy import load as load_bayes,predict as posterior_predict
    from pricing.description_features import projector
    from pricing.hierarchy_sparse_live import base_predict
    import xgboost as xgb
    import pandas as pd
    best,entries=choose(root)
    source=Path(root)/best['configuration']/'ask'
    stage=Path(output);stage.mkdir(exist_ok=True)
    target=stage/'ask'
    if target.exists():shutil.rmtree(target)
    target.mkdir()
    required=['calibration-refit.ubj','category-schema.json','minilm-heads.json','features.json',
              'evaluation.json','cleaning.json','runtime-versions.json','calibration-refit-bge-umap.joblib']
    for name in required:shutil.copy2(source/name,target/name)
    for path in source.glob('fold-*.ubj'):shutil.copy2(path,target/path.name)
    shutil.copytree(source/'bayesian',target/'bayesian')
    shutil.copytree('minilm',target/'minilm')
    shutil.copytree('adapted-bge',target/'adapted-bge')
    meta,arrays=load_bayes(target/'bayesian')
    if meta['base_evaluation_sha256']!=hashlib.sha256((target/'evaluation.json').read_bytes()).hexdigest():
        raise ValueError('Posterior belongs to another XGBoost model')
    # Pin serialization libraries to the versions that produced this reducer.
    import importlib.metadata as packages
    versions={name:packages.version(name) for name in ('umap-learn','scikit-learn','joblib','numba')}
    if versions!=json.loads((target/'runtime-versions.json').read_text()):raise ValueError('Training and serving serialization versions differ')
    # Check the actual GUI entry points after binary splitting, with no network encoder fallback.
    for path in sorted(target.rglob('*')):
        if path.is_file() and path.parent!=target/'minilm' and path.stat().st_size>8*1024*1024:split_model(path)
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    features=projector(target)
    texts=['Well maintained red sedan with automatic transmission and service records.',
           'Well maintained red sedan with automatic transmission and service records.']
    values=features.transform(texts).toarray()
    if values.shape!=(2,1+best['dimensions']) or not np.allclose(values[0],values[1]):
        raise ValueError('GUI feature smoke failed')
    if not np.allclose(values[0],features.transform(texts[:1]).toarray()[0]):
        raise ValueError('Repeated GUI description changed features')
    from pricing.suite_storage import read_model_file
    booster=xgb.Booster();booster.load_model(bytearray(read_model_file(target/'calibration-refit.ubj')))
    booster.set_param({'nthread':2})
    schema=json.loads((target/'category-schema.json').read_text())
    fields={c:levels[0] for c,levels in schema.items()}
    fields.update(odometer=75000,auction_grade=None,text=texts[0])
    price,_=base_predict(booster,schema,features,fields,8.,'ask')
    pred=posterior_predict(meta,arrays,pd.DataFrame([{**fields,'base_price':price,'age':8.}]))
    if not np.isfinite(pred.corrected_price.iloc[0]) or pred.corrected_price.iloc[0]<=0:
        raise ValueError('GUI Bayesian smoke failed')
    report={'status':'ready for regular pricing','selected':best,'configurations':entries,
        'selection_rule':'Lowest validation dollar MAE after matching Bayesian refit; test never ranks UMAP settings.',
        'checkpoint':'Independent calibration-only BGE MLM checkpoint 0, preselected before pricing comparison.',
        'pooling':'Normalized final-layer CLS; 256 tokens; explicit advertised money redacted.',
        'umap_fit':'Within each outer training fold; final serving reducer fitted on calibration only.',
        'gui':'The same local adapted checkpoint and saved reducer.transform; no inference-time fit.',
        'smoke':{'description_columns':int(values.shape[1]),'xgboost_price':price,
                 'bayesian_price':float(pred.corrected_price.iloc[0]),'offline':True},
        'runtime_versions':versions,
        'limits':'Eight settings share a validation cohort. Test cars were examined in earlier releases; this is not a new external benchmark. Independent MLM shards are not one jointly trained encoder.'}
    (stage/'comparison.json').write_text(json.dumps(report,indent=2))
    release=json.loads(Path('models/xgboost-hierarchy/release.json').read_text())
    release.update(description_features='minilm-price-adapted-bge-umap',ask_description_features=best,
                   source_run=os.environ.get('GITHUB_RUN_ID'),adapted_bge_checkpoint=0)
    (stage/'release.json').write_text(json.dumps(release,indent=2))
    requirements=Path('requirements.txt').read_text()
    for name,version in versions.items():
        import re
        requirements=re.sub(r'^'+re.escape(name)+r'[<>=!~].*$',name+'=='+version,requirements,flags=re.M) if re.search(r'^'+re.escape(name)+r'[<>=!~]',requirements,re.M) else requirements+'\n'+name+'=='+version+'\n'
    (stage/'requirements.txt').write_text(requirements)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':build()
