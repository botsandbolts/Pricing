"""Modest, independently assessed ask/auction models and same-car comparisons.

AFT receives positive dollar bounds (it logs internally). Asking labels alone are
not censoring evidence: absent measured bounds, labels are exact asking prices.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pricing.label_quality import cleanlab_inliers
from carbids.common import metrics
from marketbridge.structural import prepare, IndexPanel, known_pricing_vins

CATS=['manufacturer','model','type','transmission','state','fuel','drive','title_status','cylinders','paint_color']
NUMS=['age','log_miles','auction_grade']
NAMES=['lgb_huber','lgb_quantile_50','xgb_aft_normal','ridge_baseline']


def predictors(frame, side):
    prepared=frame.copy()
    if side=='bid' and 'condition' not in prepared and 'auction_condition' in prepared:
        prepared['condition']=prepared['auction_condition']
    shape,keys,valid,dates=prepare(prepared,training=side=='bid')
    x=shape[NUMS].copy()
    for col in CATS:
        values=frame[col] if col in frame else pd.Series(None,index=frame.index,dtype=object)
        x[col]=values.astype('string').str.lower().str.strip().fillna('unknown').astype(object)
    return x,valid,dates


def apply_category_pooling(x,pooling):
    result=x.copy()
    for column,spec in pooling.items():
        values=result[column].astype(object)
        result[column]=values.where(values.isin(spec['retained_levels']),spec['replacement'])
    return result


def encode(x,schema):
    result=x.copy()
    for col,values in schema.items():
        result[col]=pd.Categorical(result[col],categories=values)
    return result


def numeric_codes(x):
    result=x.copy()
    for col in result:
        if isinstance(result[col].dtype,pd.CategoricalDtype):
            result[col]=result[col].cat.codes.astype(float).replace(-1,np.nan)
    return result


def bounds(prices,lower=None,upper=None):
    y=np.asarray(prices,float)
    lo=y.copy() if lower is None else np.asarray(lower,float)
    hi=y.copy() if upper is None else np.asarray(upper,float)
    if lo.shape!=y.shape or hi.shape!=y.shape or not np.isfinite(y).all() or (y<=0).any():
        raise ValueError('Positive finite prices and matching AFT bound arrays required')
    if np.isnan(lo).any() or np.isnan(hi).any() or (lo<0).any() or (hi<=0).any() or (lo>hi).any() or np.isinf(lo).any():
        raise ValueError('Invalid AFT price bounds')
    return lo,hi


def fit(name,x,y,lower=None,upper=None):
    y=np.asarray(y,float)
    if name.startswith('lgb_'):
        kwargs={'objective':'huber'} if name=='lgb_huber' else {'objective':'quantile','alpha':.5}
        # All statistical hyperparameters remain library defaults (100 trees, lr .1).
        model=lgb.LGBMRegressor(**kwargs,n_jobs=4,random_state=42,verbosity=-1)
        model.fit(x,np.log1p(y),categorical_feature=list(x.select_dtypes('category')))
        return model.booster_
    if name=='xgb_aft_normal':
        import xgboost as xgb
        # Native categorical splits; category codes must never become ordered numbers.
        d=xgb.DMatrix(x,enable_categorical=True,nthread=4)
        lo,hi=bounds(y,lower,upper)
        d.set_float_info('label_lower_bound',lo);d.set_float_info('label_upper_bound',hi)
        return xgb.train({'objective':'survival:aft','aft_loss_distribution':'normal',
            'aft_loss_distribution_scale':1.,'tree_method':'hist','seed':42,'nthread':4},d,num_boost_round=100)
    cats=list(x.select_dtypes('category'));nums=[c for c in x if c not in cats]
    preprocess=ColumnTransformer([
        ('numeric',make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler()),nums),
        ('categorical',OneHotEncoder(handle_unknown='ignore'),cats)])
    model=make_pipeline(preprocess,Ridge(alpha=1.,solver='lsqr'))
    model.fit(x,np.log1p(y));return model


def predict(model,name,x):
    if name=='xgb_aft_normal':
        import xgboost as xgb
        p=model.predict(xgb.DMatrix(x,enable_categorical=True,nthread=4))
    else:
        raw=model.predict(x,num_threads=4) if name.startswith('lgb_') else model.predict(x)
        p=np.expm1(np.clip(raw,0,np.log1p(1e7)))
    p=np.asarray(p,float)
    if not np.isfinite(p).all():raise ValueError('Nonfinite model prediction')
    return np.clip(p,0,1e7)


def save(model,name,path):
    if name.startswith('lgb_'):
        with gzip.open(path/f'{name}.txt.gz','wt') as f:f.write(model.model_to_string())
    elif name.startswith('xgb_'):
        with gzip.open(path/f'{name}.ubj.gz','wb') as f:f.write(model.save_raw(raw_format='ubj'))
    else:joblib.dump(model,path/f'{name}.joblib',compress=3)


def comparisons(validation,test,observed=None):
    """All pairs share row identities. Direction is response = intercept + beta*predictor."""
    if validation.columns.tolist()!=test.columns.tolist() or not np.isfinite(validation.to_numpy()).all() or not np.isfinite(test.to_numpy()).all():
        raise ValueError('Finite aligned prediction columns required')
    rho=test.corr(method='spearman');pairs=[]
    for response in test:
        for predictor in test:
            if response==predictor:continue
            xv=validation[[predictor]].to_numpy();xt=test[[predictor]].to_numpy()
            scaler=StandardScaler().fit(xv)
            model=Ridge(alpha=1.).fit(scaler.transform(xv),validation[response])
            beta=float(model.coef_[0]/scaler.scale_[0]);intercept=float(model.intercept_-beta*scaler.mean_[0])
            mapped=model.predict(scaler.transform(xt));reference=test[response].to_numpy()
            pairs.append({'response':response,'predictor':predictor,'beta':beta,'intercept_usd':intercept,
                'bias_predictor_minus_response_usd':float((test[predictor]-test[response]).mean()),
                'calibrated_test_mae_usd':float(np.mean(abs(mapped-reference))),
                'calibrated_test_bias_usd':float(np.mean(mapped-reference))})
    calibrations=[]
    if observed is not None:
        yv,yt=observed
        for col in test:
            xv=validation[[col]].to_numpy();scaler=StandardScaler().fit(xv)
            model=Ridge(alpha=1.).fit(scaler.transform(xv),yv)
            beta=float(model.coef_[0]/scaler.scale_[0]);intercept=float(model.intercept_-beta*scaler.mean_[0])
            p=model.predict(scaler.transform(test[[col]].to_numpy()))
            calibrations.append({'model':col,'beta':beta,'intercept_usd':intercept,
                'raw_test_bias_usd':float(np.mean(test[col].to_numpy()-yt)),
                'calibrated_test_bias_usd':float(np.mean(p-yt)),**metrics(yt,p)})
    # Constant vectors have undefined correlation: serialize null, never invented zero.
    return {'spearman':rho.astype(object).where(rho.notna(),None).to_dict(),
        'ridge_pairs':pairs,'ridge_observed_price_calibration':calibrations,
        'ridge_alpha':1.,'ridge_fit_partition':'validation','ridge_coefficient_scale':'original dollar per dollar',
        'ridge_equation':'response_price = intercept_usd + beta * predictor_price',
        'spearman_partition':'test','test_rows':len(test)}


def load_cohort(path):
    return pd.concat([chunk.drop(columns=['text','price_evidence','source_url'],errors='ignore')
        for chunk in pd.read_json(path,lines=True,chunksize=5000)],ignore_index=True)


def train_side(frame,side,out,panel,text=None):
    out.mkdir(parents=True,exist_ok=True)
    x,valid,dates=predictors(frame,side)
    ratio,_=panel.ratios(dates)
    if side=='bid':valid&=ratio.notna()
    frame=frame.loc[valid].copy();x=x.loc[valid].copy();ratio=ratio.loc[valid]
    # Auction training is indexed to Jan 2015; inference restores the requested market month.
    factor=ratio.to_numpy(float) if side=='bid' else np.ones(len(frame))
    y=frame.price.to_numpy(float)/factor
    if ('aft_lower_price' in frame)!=('aft_upper_price' in frame):raise ValueError('Supply both AFT bound columns')
    lower=frame.aft_lower_price.to_numpy(float)/factor if 'aft_lower_price' in frame else y.copy()
    upper=frame.aft_upper_price.to_numpy(float)/factor if 'aft_upper_price' in frame else y.copy()
    bounds(y,lower,upper)
    if text is not None:
        if side!='ask':raise ValueError('No genuine auction descriptions available')
        x['predicted_text_price']=text.loc[frame.row_id].to_numpy(float)
    tr=np.flatnonzero(frame.phase=='train');va=np.flatnonzero(frame.phase=='validation');te=np.flatnonzero(frame.phase=='test')
    audit=x.drop(columns=['predicted_text_price'],errors='ignore')
    print(side,'assessment audit',len(tr),flush=True)
    keep,assessment_filter=cleanlab_inliers(audit.iloc[tr],y[tr],out/'assessment-cleanlab',frame.row_id.iloc[tr])
    schema={c:sorted(x.iloc[tr][c].dropna().unique().tolist()) for c in CATS}
    encoded=encode(x,schema)
    test_predictions=pd.DataFrame(index=frame.row_id.iloc[te].tolist())
    val_predictions=pd.DataFrame(index=frame.row_id.iloc[va].tolist())
    records={};assessment_models={}
    for name in NAMES:
        print(side,'assessment',name,flush=True)
        model=fit(name,encoded.iloc[tr[keep]],y[tr[keep]],lower[tr[keep]],upper[tr[keep]])
        pv=predict(model,name,encoded.iloc[va])*factor[va];pt=predict(model,name,encoded.iloc[te])*factor[te]
        val_predictions[side+'_'+name]=pv;test_predictions[side+'_'+name]=pt
        records[name]={'validation':metrics(frame.price.iloc[va],pv),'test':metrics(frame.price.iloc[te],pt)}
        assessment_models[name]=model
        assessment_path=out/'assessment';assessment_path.mkdir(exist_ok=True);save(model,name,assessment_path)
        print(json.dumps(records[name]),flush=True)
    (out/'assessment'/'schema.json').write_text(json.dumps(schema))
    comparison=comparisons(val_predictions,test_predictions,(frame.price.iloc[va].to_numpy(),frame.price.iloc[te].to_numpy()))
    report={'side':side,'rows':{'eligible':len(frame),'train':len(tr),'validation':len(va),'test':len(te)},
        'text_feature':text is not None,'models':records,'outlier_filter':assessment_filter,'comparisons':comparison,
        'target':'auction CPI-adjusted reference dollars' if side=='bid' else 'historical asking dollars',
        'aft':{'distribution':'normal','scale':1.,'rounds':100,'censored_rows':int(np.sum(lower!=upper)),
            'bounds':'explicit aft_lower_price/aft_upper_price columns in dollars' if 'aft_lower_price' in frame else 'exact observed prices; no fabricated censoring'},
        'lightgbm':'library default statistical parameters, objective Huber or quantile alpha=0.5, log1p target',
        'evaluation':'Chronological original validation/test remain unfiltered; all metrics use assessment models, never full-data refits.'}
    (out/'evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(side,'full-data audit',len(frame),flush=True)
    keep_all,deployment_filter=cleanlab_inliers(audit,y,out/'deployment-cleanlab',frame.row_id)
    # Deployment text must use full-cohort OOF, not train-only held-out predictions.
    if text is not None:
        full=text.attrs.get('full_oof')
        if full is None:raise ValueError('Full-cohort OOF feature required for deployment')
        x['predicted_text_price']=full.loc[frame.row_id].to_numpy(float)
    final_schema={c:sorted(x[c].dropna().unique().tolist()) for c in CATS};xf=encode(x,final_schema)
    for name in NAMES:
        print(side,'full-data refit',name,flush=True)
        save(fit(name,xf.iloc[keep_all],y[keep_all],lower[keep_all],upper[keep_all]),name,out)
    (out/'schema.json').write_text(json.dumps(final_schema))
    (out/'manifest.json').write_text(json.dumps({'status':'ready','pipeline':'modest-model-suite',
        'side':side,'model_names':NAMES,'feature_names':list(x),'text_feature':text is not None,
        'deployment_scope':'full-eligible-data-refit-after-held-out-assessment','outlier_filter':deployment_filter,
        'trained_rows':len(keep_all),'training_candidates':len(frame),
        'library_versions':{package:__import__('importlib.metadata',fromlist=['version']).version(package) for package in ['lightgbm','xgboost','scikit-learn','cleanlab']},
        'training_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
    return frame,x,assessment_models,schema,val_predictions,test_predictions,report


def run(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'comparison.json').exists():raise ValueError('Use new output directory for completed suite')
    panel=IndexPanel(a.index)
    ask=load_cohort(a.ask);bid=load_cohort(a.bid)
    vins=known_pricing_vins(a.ask)
    overlap=bid.vehicle_group.astype(str).str.removeprefix('vin:').isin(vins)
    bid=bid.loc[~overlap].reset_index(drop=True)
    print('Excluded auction VIN overlaps:',int(overlap.sum()),flush=True)
    text=None
    if a.text_features:
        from pricing.suite_text import validated_features
        text=validated_features(a.ask,a.text_features)
    asks=train_side(ask,'ask',out/'ask',panel,text)
    bids=train_side(bid,'bid',out/'bid',panel)
    # Compare ALL eight models on the SAME asking validation/test cars and valuation dates.
    frame,x,_,_,av,at,_=asks
    bx,valid,dates=predictors(frame,'bid');ratio,_=panel.ratios(dates)
    bx=encode(bx,bids[3]);va=frame.phase.to_numpy()=='validation';te=frame.phase.to_numpy()=='test'
    if not (valid&ratio.notna()).all():raise ValueError('Common-car index support incomplete')
    cv=av.copy();ct=at.copy()
    for name,model in bids[2].items():
        cv['bid_'+name]=predict(model,name,bx.loc[va])*ratio.loc[va].to_numpy()
        ct['bid_'+name]=predict(model,name,bx.loc[te])*ratio.loc[te].to_numpy()
    report=comparisons(cv,ct)
    report.update({'auction_vin_overlap_exclusions':int(overlap.sum()),
        'cohort':'same held-out asking cars; auction estimates transported by vehicle age and lagged CPI',
        'warning':'Ask versus auction correlation is model agreement, not correlation between independently observed sale and asking prices. Auction transport is not retail sale calibration.',
        'source_sha256':{side:hashlib.file_digest(open(path,'rb'),'sha256').hexdigest() for side,path in [('ask',a.ask),('bid',a.bid)]}})
    (out/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    ct.to_csv(out/'common-test-predictions.csv',index_label='row_id')
    print('COMPLETED',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ask',required=True);p.add_argument('--bid',required=True)
    p.add_argument('--index',default='models/structural-transfer/used-car-index.csv')
    p.add_argument('--output',default='models/price-model-suite');p.add_argument('--text-features')
    run(p.parse_args())
