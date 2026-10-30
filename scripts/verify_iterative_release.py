from pathlib import Path
import json,math
import numpy as np
from pricing.catboost_aft_pipeline import CatBoostAFTPipeline
from pricing.iterative_release import completed
root=Path('.');assert completed(root)
checks=[]
for name in ['ask','ask-quantile','bid','bid-aft']:
 folder=root/'models/iterative-catboost'/name
 m=json.loads((folder/'manifest.json').read_text());e=json.loads((folder/'evaluation.json').read_text())
 assert m['source_sha256']==e['source_sha256']
 stages=m['outlier_filter']['stages'];assert len(stages)==5
 for i,s in enumerate(stages):
  for r in s['counts']:
   assert r['removed']==math.ceil(.05*r['before'])
   assert r['retained']==r['before']-r['removed']
   if i:assert r['before']==next(v['retained'] for v in stages[i-1]['counts'] if v['phase']==r['phase'])
 assert sum(r['retained'] for r in stages[-1]['counts'])==m['trained_rows']
 assert e['assessment']['original_unfiltered_test']['rows']==next(r['before'] for r in stages[0]['counts'] if r['phase']=='test')
 model=CatBoostAFTPipeline(folder)
 fields=dict(year=2015,manufacturer='Toyota',model='Corolla SE',odometer=95000,type='sedan',state='ca',condition=None,fuel='gas',transmission='automatic')
 good=model.predict('Full service history. Minor cosmetic dent. Inspection welcome.',fields,explain=True)
 assert 0<good['price']<1e7 and good['explanation']['additivity_error']<1e-5
 if m['side']=='ask':
  bad=model.predict('Engine damaged, does not start, sold as a repair project.',fields)
  assert not np.isclose(good['predicted_text_price'],bad['predicted_text_price'])
 checks.append(dict(model=name,price=good['price'],text_price=good['predicted_text_price'],shap_error=good['explanation']['additivity_error'],trained_rows=m['trained_rows']))
 print('VERIFIED',checks[-1],flush=True)
 del model
(root/'models/iterative-catboost/serving-verification.json').write_text(json.dumps({'status':'passed','checks':checks,'scope':'Real saved weights; schema/provenance, stage counts, full original test sizes, dynamic MiniLM and native SHAP checks. Smoke inputs are not accuracy measurements.'},indent=2))
