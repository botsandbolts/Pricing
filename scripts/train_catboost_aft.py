"""Chronological CatBoost log-price AFT assessment and unfiltered residual audit."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from catboost import CatBoostRegressor,Pool
from pricing.model_suite import load_cohort,predictors
from marketbridge.structural import IndexPanel,known_pricing_vins
from scripts.audit_outliertree import serial


def metrics(y,p):
    r=y-p
    return {'mae_usd':float(np.abs(r).mean()),'rmse_usd':float(np.sqrt(np.mean(r*r))),'bias_observed_minus_predicted_usd':float(r.mean()),'median_absolute_error_usd':float(np.median(np.abs(r))),'log_rmse':float(np.sqrt(np.mean((np.log(y)-np.log(p))**2))),'log_mae':float(np.abs(np.log(y)-np.log(p)).mean()),'spearman':float(stats.spearmanr(y,p).statistic)}


def residual_summary(r):
    q=[0,.01,.05,.25,.5,.75,.95,.99,1.]
    return {'mean':float(r.mean()),'std':float(r.std(ddof=1)),'skewness':float(stats.skew(r)),'excess_kurtosis':float(stats.kurtosis(r)),'quantiles':dict(zip(map(str,q),map(float,np.quantile(r,q))))}


def residual_groups(d):
    records=[]
    for name,g in d.groupby('group',observed=True):
        records.append({'group':str(name),'rows':len(g),'mean_observed':float(g.observed.mean()),'mean_predicted':float(g.predicted.mean()),**metrics(g.observed.to_numpy(),g.predicted.to_numpy())})
    return records


def assess(d,x,y,factor,out,iterations=1000):
    start=time.monotonic();out.mkdir(parents=True,exist_ok=True);cats=[]
    for c in x:
        if pd.api.types.is_numeric_dtype(x[c]):x[c]=x[c].astype(float)
        else:x[c]=x[c].fillna('unknown').astype(str);cats.append(c)
    masks={p:(d.phase==p).to_numpy() for p in ['train','validation','test']}
    def pool(mask):return Pool(x.loc[mask],label=np.column_stack([y[mask],y[mask]]),cat_features=cats)
    params=dict(loss_function='SurvivalAft:dist=Normal;scale=1.0',iterations=iterations,depth=6,learning_rate=.05,l2_leaf_reg=3.,random_seed=42,thread_count=4,allow_writing_files=False)
    model=CatBoostRegressor(**params)
    print('TRAIN',out,len(d),flush=True)
    model.fit(pool(masks['train']),eval_set=pool(masks['validation']),early_stopping_rounds=100,verbose=100)
    model.save_model(str(out/'assessment.cbm'))
    predictions={};measures={}
    for phase in ['validation','test']:
        mask=masks[phase];raw=np.asarray(model.predict(x.loc[mask],prediction_type='RawFormulaVal'),float).reshape(-1)
        if not np.isfinite(raw).all():raise ValueError('Nonfinite AFT predictions')
        p=np.exp(raw)*factor[mask];observed=d.price.to_numpy(float)[mask]
        if not np.isfinite(p).all() or (p<=0).any():raise ValueError('Invalid price prediction')
        r=observed-p;lr=np.log(observed)-np.log(p)
        rows=d.loc[mask,[c for c in ['row_id','phase','manufacturer','model','year','odometer'] if c in d]].copy()
        rows['observed']=observed;rows['predicted']=p;rows['residual_usd']=r;rows['residual_log']=lr
        predictions[phase]=rows;measures[phase]=metrics(observed,p)
    v=predictions['validation'];t=predictions['test'];lr=t.residual_log.to_numpy();r=t.residual_usd.to_numpy()
    center=float(v.residual_log.median());mad=float(np.median(np.abs(v.residual_log-center)));scale=1.4826*mad
    if scale<=0:raise ValueError('Degenerate validation MAD')
    t['log_residual_robust_z']=(t.residual_log-center)/scale;t['review_candidate']=abs(t.log_residual_robust_z)>3.5
    t['group']=pd.qcut(t.predicted,10,duplicates='drop');deciles=residual_groups(t)
    make=t.assign(group=t.manufacturer.fillna('unknown'));make_groups=[g for g in residual_groups(make) if g['rows']>=30]
    # Smearing factor estimated only on validation, not test.
    smear=float(np.exp(v.residual_log).mean());smear_metrics=metrics(t.observed.to_numpy(),t.predicted.to_numpy()*smear)
    for p,rows in predictions.items():rows.drop(columns=['group'],errors='ignore').to_csv(out/f'{p}-residuals.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    plot_residuals(t,out)
    report={'rows':{p:int(m.sum()) for p,m in masks.items()},'parameters':params,'best_iteration':model.get_best_iteration(),'fitted_trees':model.tree_count_,'metrics':measures,'test_residual_usd':residual_summary(r),'test_residual_log':residual_summary(lr),'test_log_residual_factor_95_percent':list(map(float,np.exp(np.quantile(lr,[.025,.975])))),'heteroscedasticity_spearman_abs_dollar_residual_vs_prediction':float(stats.spearmanr(abs(r),t.predicted).statistic),'heteroscedasticity_spearman_abs_log_residual_vs_log_prediction':float(stats.spearmanr(abs(lr),np.log(t.predicted)).statistic),'predicted_price_deciles':deciles,'make_groups_min_30_test_cars':make_groups,'validation_mad_rule':{'center_log':center,'scale_log':scale,'abs_robust_z_threshold':3.5,'test_review_candidates':int(t.review_candidate.sum()),'test_review_fraction':float(t.review_candidate.mean()),'note':'Review candidates only, not confirmed incorrect prices; no rows removed.'},'validation_smearing_factor':smear,'test_metrics_with_validation_smearing':smear_metrics,'worst_absolute_dollar_residuals':t.loc[t.residual_usd.abs().nlargest(10).index].drop(columns=['group']).to_dict('records'),'elapsed_seconds':time.monotonic()-start,'target':'Exact positive dollar bounds [price, price]; auction prices use CPI-reference dollars. SurvivalAft logs bounds internally. RawFormulaVal is log reference price; exp(raw) restores conditional median dollars, then CPI ratio restores auction date. No double logarithm. No invented censoring.','features':list(x),'text_feature':False,'evaluation':'Original chronological train/validation/test. Validation selects stopping iteration and residual MAD/smearing; test remains unfiltered and unused for fitting or tuning. Native categorical handling, no one-hot. Both previous outlier audits are diagnostic only. No model deployment or full-cohort refit.'}
    (out/'report.json').write_text(json.dumps(serial(report),indent=2,allow_nan=False));print('COMPLETE',out,json.dumps(measures),flush=True)
    return report


def plot_residuals(t,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    axes[0,0].hist(t.residual_usd,bins=100);axes[0,0].set_title('Dollar residuals: observed minus predicted')
    axes[0,1].hist(t.residual_log,bins=100);axes[0,1].set_title('Log-price residuals')
    stats.probplot(t.residual_log,dist='norm',plot=axes[0,2]);axes[0,2].set_title('Normal Q-Q: log residuals')
    axes[1,0].hexbin(t.predicted,t.residual_usd,gridsize=60,bins='log',mincnt=1);axes[1,0].axhline(0,color='red');axes[1,0].set(xlabel='Predicted dollars',ylabel='Dollar residual')
    axes[1,1].hexbin(np.log(t.predicted),t.residual_log,gridsize=60,bins='log',mincnt=1);axes[1,1].axhline(0,color='red');axes[1,1].set(xlabel='Log predicted price',ylabel='Log residual')
    g=t.groupby('group',observed=True)[['observed','predicted']].mean();axes[1,2].plot(g.predicted,g.observed,'o-');lo=min(g.min());hi=max(g.max());axes[1,2].plot([lo,hi],[lo,hi],'--');axes[1,2].set(xlabel='Mean predicted dollars by decile',ylabel='Mean observed dollars',title='Held-out calibration')
    fig.tight_layout();fig.savefig(out/'residual-diagnostics.png',dpi=140);plt.close(fig)


def run(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise ValueError('Choose fresh output')
    known=known_pricing_vins(a.ask);panel=IndexPanel(a.index);prior=json.loads(Path(a.prior).read_text());results={}
    for side,path in [('ask',a.ask),('bid',a.bid)]:
        sha=hashlib.file_digest(open(path,'rb'),'sha256').hexdigest();assert sha==prior['markets'][side]['source_sha256']
        d=load_cohort(path);excluded=0
        if side=='bid':
            overlap=d.vehicle_group.astype(str).str.removeprefix('vin:').isin(known);excluded=int(overlap.sum());d=d.loc[~overlap].reset_index(drop=True)
        x,valid,dates=predictors(d,side);ratios,_=panel.ratios(dates)
        if side=='bid':valid&=ratios.notna()
        d=d.loc[valid].copy();x=x.loc[valid].copy();assert len(d)==prior['markets'][side]['eligible_rows']
        y=d.price.to_numpy(float);factor=ratios.loc[d.index].to_numpy(float) if side=='bid' else np.ones(len(d))
        results[side]=assess(d,x,y/factor,factor,out/side,a.iterations);results[side]['source_sha256']=sha;results[side]['auction_vin_overlap_exclusions']=excluded
        del d,x
    (out/'summary.json').write_text(json.dumps(serial({'status':'complete','markets':results}),indent=2,allow_nan=False))
    print('ALL CATBOOST AFT ASSESSMENTS COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ask',required=True);p.add_argument('--bid',required=True);p.add_argument('--output',required=True);p.add_argument('--iterations',type=int,default=1000);p.add_argument('--index',default='models/structural-transfer/used-car-index.csv');p.add_argument('--prior',default='reports/cleanlab-rerun-20261027/summary.json');run(p.parse_args())
