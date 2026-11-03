"""Publish a complete matched base/posterior pair only after bounded inference."""
from pathlib import Path
import json,shutil,resource
from pricing.suite_storage import split_model
source=Path('training-cache/lemon-hierarchy');root=Path('models/xgboost-hierarchy')
staging=Path('training-cache/lemon-serving');shutil.rmtree(staging,ignore_errors=True)
for side in ['ask','bid']:
    dest=staging/side;dest.mkdir(parents=True)
    for name in ['calibration-refit.ubj','category-schema.json','evaluation.json','cleaning.json']+(['features.json','minilm-heads.json'] if side=='ask' else []):
        shutil.copy2(source/side/name,dest/name)
    shutil.copytree(source/side/'bayesian',dest/'bayesian')
    if side=='ask':shutil.copytree(root/side/'minilm',dest/'minilm')
    if (dest/'calibration-refit.ubj').stat().st_size>8*1024*1024:split_model(dest/'calibration-refit.ubj')
(staging/'release.json').write_text(json.dumps({'transformer_feature':True,'description_features':'minilm-price','method':'lemon-retained-fivefold-bayesian','residual_source':'fivefold-within-retained-75-percent','model':'Five group-held-out XGBoost models per market on retained Lemon training; matched robust Bayesian log residuals','training_fraction_per_fold':'approximately 80% of retained 75%; grouped vehicles kept intact','validation_test_filtered':False},indent=2))
shutil.rmtree(root);shutil.copytree(staging,root)
from pricing.hierarchy_sparse_live import estimate
from datetime import date
results,_,_=estimate(Path('.'),dict(year=2015,manufacturer='Toyota',model='Corolla',odometer=95000,state='ca',text='Full service history.'),date.today(),True)
for side,r in results.items():
    assert 'error' not in r,(side,r)
    assert 0<r['interval_low']<=r['price']<=r['interval_high']
size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file());rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
assert size<100*1024*1024,size
assert rss<550*1024,rss
print('SERVING VERIFIED',size,rss,flush=True)
