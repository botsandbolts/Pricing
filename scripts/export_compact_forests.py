"""Compile the exact saved trees to bitsets; verify every published example."""
import io,json,lzma,tempfile,shutil,hashlib,sys
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from pricing.suite_storage import read_model_file,split_model
from pricing.forest_compact import predict
from pricing.forest_residuals import CATEGORICAL

for side in sys.argv[1:] or ['ask','bid']:
 folder=Path('models/forest-original')/side;meta=json.loads((folder/'base-assessment.json').read_text());model=XGBRegressor()
 with tempfile.NamedTemporaryFile(suffix='.ubj') as f:
  with lzma.open(io.BytesIO(read_model_file(folder/'serving.ubj.xz')),'rb') as src:shutil.copyfileobj(src,f)
  f.flush();model.load_model(f.name)
 config=json.loads(model.get_booster().save_config());base=json.loads(config['learner']['learner_model_param']['base_score']);base=base[0] if isinstance(base,list) else base
 arrays={k:[] for k in ['feature','value','yes','no','missing','cat_offset','cat_length']};bits=bytearray();starts=[0]
 for text in model.get_booster().get_dump(dump_format='json'):
  tree=json.loads(text);flat={}
  def visit(node):
   flat[node['nodeid']]=node
   for child in node.get('children',[]):visit(child)
  visit(tree);offset=starts[-1];count=max(flat)+1
  for i in range(count):
   n=flat[i];leaf='leaf' in n;cat=isinstance(n.get('split_condition'),list)
   arrays['feature'].append(-1 if leaf else meta['features'].index(n['split']))
   arrays['value'].append(n['leaf'] if leaf else 0. if cat else n['split_condition'])
   for k in ['yes','no','missing']:arrays[k].append(offset+n.get(k,0))
   arrays['cat_offset'].append(len(bits));length=0
   if cat:
    categories=n['split_condition'];length=max(categories)//8+1 if categories else 0;packed=bytearray(length)
    for code in categories:packed[code//8]|=1<<(code%8)
    bits.extend(packed)
   arrays['cat_length'].append(length)
  starts.append(offset+count)
 compact={k:np.array(v,dtype=np.float32 if k=='value' else np.int32) for k,v in arrays.items()};compact.update(bits=np.frombuffer(bytes(bits),dtype=np.uint8),starts=np.array(starts,np.int32),base=np.array([base],np.float32))
 examples=pd.read_csv(folder/'examples.csv.gz');x=pd.DataFrame({'age':examples.age,'log_miles':np.log1p(examples.odometer),'auction_grade':examples.auction_grade})
 for c in CATEGORICAL:x[c]=pd.Categorical(examples[c].fillna('unknown').astype(str).str.strip().str.lower(),categories=meta['category_schema'][c])
 prices=np.exp(np.clip(predict(compact,x[meta['features']]),0,np.log(1e7)))
 np.testing.assert_allclose(prices,examples.base_price,rtol=2e-5,atol=.02)
 path=folder/'serving-compact.npz';np.savez_compressed(path,**compact)
 spec={'format':'exact_xgboost_tree_bitsets_v1','trees':200,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,'verified_examples':len(examples),'max_abs_prediction_difference_usd':float(np.max(abs(prices-examples.base_price))),'scope':'Same calibration-only XGBoost trees; compilation changes storage and traversal only, no retraining.'}
 (folder/'serving-compact.json').write_text(json.dumps(spec,indent=2));print(side,spec,flush=True)

 if path.stat().st_size>500000:split_model(path,500000)
