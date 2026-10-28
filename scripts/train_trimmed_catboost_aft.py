"""CatBoost SurvivalAft on the exact user-requested residual-trimmed cohorts."""
import argparse,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor,Pool
from scipy.stats import spearmanr
from pricing.model_suite import load_cohort,predictors,apply_category_pooling
from marketbridge.structural import IndexPanel,known_pricing_vins
from pricing.dataset_sources import ASK,BID

PARAMS=dict(loss_function='SurvivalAft:dist=Normal;scale=1.0',iterations=1500,depth=8,learning_rate=.05,l2_leaf_reg=10.,random_strength=1.,rsm=.8,bootstrap_type='Bayesian',bagging_temperature=1.,random_seed=42,thread_count=4,allow_writing_files=False)


def metrics(y,p):
 r=y-p;lr=np.log(y)-np.log(p)
 return {'rows':len(y),'mae_usd':float(abs(r).mean()),'rmse_usd':float(np.sqrt(np.mean(r*r))),'bias_observed_minus_predicted_usd':float(r.mean()),'log_mae':float(abs(lr).mean()),'log_rmse':float(np.sqrt(np.mean(lr*lr))),'spearman':float(spearmanr(y,p).statistic)}


def pack(model,path):
 temporary=path.with_suffix('.cbm');model.save_model(str(temporary))
 with temporary.open('rb') as f,gzip.open(path,'wb',compresslevel=6) as z:
  import shutil;shutil.copyfileobj(f,z)
 temporary.unlink()


def run(a):
 text=None;text_meta=None
 if a.text_features:
  from pricing.suite_text import validated_features
  text=validated_features(a.ask,a.text_features,full_only=a.prefit_full)
  text_meta=json.loads((Path(a.text_model)/'provenance.json').read_text())
  expected=json.loads((Path(a.text_features)/'full-text-provenance.json').read_text())
  if text_meta!=expected:raise ValueError('Deployment text head does not match OOF provenance')
 root=Path(a.output);root.mkdir(parents=True,exist_ok=True);known=known_pricing_vins(a.ask);panel=IndexPanel(a.index);results={}
 for side,path in [('ask',a.ask),('bid',a.bid)]:
  if a.side and side!=a.side:continue
  out=root/side;out.mkdir(exist_ok=True)
  use_text=side=='ask' and text is not None
  if (out/'manifest.json').exists():
   existing=json.loads((out/'manifest.json').read_text())
   if existing.get('text_feature',False)!=use_text:raise ValueError('Choose a fresh output for a different architecture')
   results[side]=json.loads((out/'evaluation.json').read_text());continue
  prior=Path(a.filters)/side;old=json.loads((prior/'evaluation.json').read_text());sha=hashlib.file_digest(open(path,'rb'),'sha256').hexdigest();assert sha==old['source_sha256']
  d=load_cohort(path)
  if side=='bid':d=d.loc[~d.vehicle_group.astype(str).str.removeprefix('vin:').isin(known)].reset_index(drop=True)
  x,valid,dates=predictors(d,side);ratios,_=panel.ratios(dates)
  if side=='bid':valid&=ratios.notna()
  d=d.loc[valid].copy();x=x.loc[valid].copy();audit=pd.read_csv(prior/'filter-rows.csv.gz',dtype={'row_id':str})
  assert audit.row_id.tolist()==d.row_id.astype(str).tolist() and audit.phase.tolist()==d.phase.tolist()
  keep=audit.retained.to_numpy(bool);assert int(keep.sum())==old['outlier_filter']['retained_rows']
  if use_text:x['predicted_text_price']=text.loc[d.row_id].to_numpy(float)
  x=apply_category_pooling(x,old['category_pooling']);cats=[c for c in x if not pd.api.types.is_numeric_dtype(x[c])]
  for c in cats:x[c]=x[c].fillna('unknown').astype(str)
  factor=ratios.loc[d.index].to_numpy(float) if side=='bid' else np.ones(len(d));y=d.price.to_numpy(float)/factor;phase=d.phase.to_numpy()
  masks={'train':keep&(phase=='train'),'validation':keep&(phase=='validation'),'test':keep&(phase=='test')}
  def pool(mask):return Pool(x.loc[mask],label=np.column_stack([y[mask],y[mask]]),cat_features=cats)
  params={**PARAMS,'iterations':a.iterations}
  stamp={'parameters':params,'source_sha256':sha,'text_feature':use_text,'features':list(x),'trained_rows':int(keep.sum())}
  if use_text:stamp['full_oof_sha256']=hashlib.file_digest(open(Path(a.text_features)/'full-oof.csv','rb'),'sha256').hexdigest()
  if a.prefit_full:
   if not use_text:raise ValueError('Prefit is only for the combined text architecture')
   final=CatBoostRegressor(**params);final.fit(pool(keep),verbose=200);pack(final,out/'deployment-prefit.cbm.gz')
   (out/'deployment-prefit.json').write_text(json.dumps(stamp,indent=2));print(side,'DEPLOYMENT PREFIT CACHED; NOT PUBLISHED UNTIL ASSESSMENT COMPLETES',flush=True);continue
  print(side,'ASSESSMENT START',int(masks['train'].sum()),flush=True)
  if (out/'evaluation.json').exists() and (out/'assessment.cbm.gz').exists():
   evaluation=json.loads((out/'evaluation.json').read_text());assert evaluation['parameters']==params and evaluation['source_sha256']==sha
   assert evaluation.get('text_feature',False)==use_text
   rounds=evaluation['selected_boosting_rounds'];model=None;print(side,'RESUME SAVED ASSESSMENT',rounds,flush=True)
  else:
   model=CatBoostRegressor(**params);model.fit(pool(masks['train']),eval_set=pool(masks['validation']),early_stopping_rounds=125,verbose=200);rounds=model.tree_count_;pack(model,out/'assessment.cbm.gz')
   report={}
   for name,mask in [('filtered_validation',masks['validation']),('filtered_test',masks['test']),('original_unfiltered_test',phase=='test'),('excluded_test',(phase=='test')&~keep)]:
    raw=np.asarray(model.predict(x.loc[mask],prediction_type='RawFormulaVal'),float);p=np.exp(raw)*factor[mask];observed=d.price.to_numpy(float)[mask];report[name]=metrics(observed,p)
    if name=='original_unfiltered_test':pd.DataFrame({'row_id':d.row_id.loc[mask],'observed':observed,'predicted':p,'retained':keep[mask]}).to_csv(out/'original-test-predictions.csv.gz',index=False)
   evaluation={'side':side,'assessment':report,'rows':old['rows'],'parameters':params,'selected_boosting_rounds':rounds,'category_pooling':old['category_pooling'],'outlier_filter':old['outlier_filter'],'source_sha256':sha,'evaluation':'Same fixed filter membership as the requested XGBoost experiment: highest absolute log-residual quartile within each original phase plus OutlierTree flags. Filtered validation/test scores are optimistic because labels selected retained cars. Original-test scores are diagnostic, not a fresh external benchmark. Assessment fits training only; validation selects stopping; deployment refit is separate.','text_feature':use_text}
   (out/'evaluation.json').write_text(json.dumps(evaluation,indent=2));print(side,'ASSESSMENT COMPLETE',json.dumps(report),flush=True)
  print(side,'FINAL REFIT',int(keep.sum()),'trees',rounds,flush=True)
  if use_text:x['predicted_text_price']=text.attrs['full_oof'].loc[d.row_id].to_numpy(float)
  cached=out/'deployment-prefit.cbm.gz';recipe=out/'deployment-prefit.json'
  if cached.exists() and recipe.exists() and json.loads(recipe.read_text())==stamp and rounds==params['iterations']:
   import shutil;shutil.copy2(cached,out/'catboost_aft.cbm.gz');final=None;print(side,'REUSE VERIFIED FULL REFIT',flush=True)
  else:
   final=CatBoostRegressor(**{**params,'iterations':rounds});final.fit(pool(keep),verbose=200);pack(final,out/'catboost_aft.cbm.gz')
  manifest={'status':'ready','pipeline':'catboost-aft','training_regime':'residual-trimmed-catboost-aft','side':side,'feature_names':list(x),'categorical_columns':cats,'text_feature':use_text,'trained_rows':int(keep.sum()),'training_candidates':len(d),'selected_boosting_rounds':rounds,'category_pooling':old['category_pooling'],'parameters':params,'outlier_filter':old['outlier_filter'],'deployment_scope':'all-retained-phases-refit-after-separate-assessment','warning':evaluation['evaluation'],'source_sha256':sha,'library_version':__import__('catboost').__version__,'dataset':ASK if side=='ask' else BID}
  if use_text:
   manifest.update(text_model_path=str(Path(a.text_model).resolve().relative_to(root.resolve().parent)),text_model_fingerprint=text_meta['fingerprint'],text_provenance=text_meta,text_stacking='Assessment: training-only OOF, train-only head for held-out; deployment: separately full-cohort OOF features and matching full-cohort head')
   manifest['training_regime']='residual-trimmed-minilm-catboost-aft'
  (out/'manifest.json').write_text(json.dumps(manifest,indent=2));(out/'schema.json').write_text(json.dumps({c:sorted(x.loc[keep,c].unique().tolist()) for c in cats}));(out/'filter-report.json').write_bytes((prior/'filter-report.json').read_bytes());results[side]=evaluation;print(side,'CATBOOST READY',flush=True)
  del d,x,model,final
 if all((root/side/'manifest.json').exists() for side in ['ask','bid']):
  results={side:json.loads((root/side/'evaluation.json').read_text()) for side in ['ask','bid']}
  (root/'summary.json').write_text(json.dumps({'status':'complete','markets':results},indent=2));print('BOTH CATBOOST AFT DEFAULTS READY',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prefit-full',action='store_true');p.add_argument('--text-features');p.add_argument('--text-model',default='models/minilm-cleanlab/text-model');p.add_argument('--side',choices=['ask','bid']);p.add_argument('--ask',required=True);p.add_argument('--bid',required=True);p.add_argument('--filters',default='models/residual-trimmed-xgb');p.add_argument('--output',default='models/catboost-aft');p.add_argument('--iterations',type=int,default=1500);p.add_argument('--index',default='models/structural-transfer/used-car-index.csv');run(p.parse_args())
