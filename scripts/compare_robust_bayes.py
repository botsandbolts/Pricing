"""Select on validation MAE, calibrate on different validation cars, then test."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from pricing import robust_hierarchy as robust
from pricing import bayesian_hierarchy as old
from pricing.hierarchical import scores

def prepare(path,side):
    d=pd.read_csv(path,dtype={'row_id':str,'vehicle_group':str})
    if d.row_id.duplicated().any():raise ValueError('Duplicate row IDs')
    if d.groupby('vehicle_group').split.nunique().max()>1:raise ValueError('Vehicle crosses data splits')
    if d.loc[d.split.eq('calibration')].groupby('vehicle_group').fold.nunique().max()>1:raise ValueError('Vehicle crosses OOF folds')
    if not np.allclose(d.residual_log,np.log(d.observed_price/d.predicted_price)):raise ValueError('Mismatched residuals')
    dates=pd.to_datetime(d.date,format='mixed',utc=True)
    d['age']=(dates.dt.year+(dates.dt.dayofyear-1)/365.25-pd.to_numeric(d.year)).clip(0,150)
    if side=='bid':
        d['auction_grade']=pd.to_numeric(d['auction_condition'] if 'auction_condition' in d else d.get('condition'),errors='coerce')
        d['condition']='unknown'
    else:d['auction_grade']=np.nan
    d['base_price']=d.predicted_price;d['base_fold']=d.fold
    d['prediction_kind']=np.where(d.split.eq('calibration'),'oof','heldout')
    d['base_model_id']='xgboost-fivefold-'+side;d['price_basis']='historical advertised USD' if side=='ask' else 'CPI-reference auction USD'
    return d

def paired_bootstrap(d,first,second,repetitions=600):
    diff=np.abs(d.observed_price.to_numpy()-second)-np.abs(d.observed_price.to_numpy()-first)
    g=pd.DataFrame({'group':d.vehicle_group.to_numpy(),'change':diff}).groupby('group').change.agg(['sum','count'])
    sums=g['sum'].to_numpy();counts=g['count'].to_numpy();rng=np.random.default_rng(90210);means=[]
    for _ in range(repetitions):
        idx=rng.integers(0,len(g),len(g));means.append(float(sums[idx].sum()/counts[idx].sum()))
    return dict(mean_mae_saved_usd=float(diff.mean()),group_bootstrap_95_usd=np.quantile(means,[.025,.975]).tolist(),vehicle_groups=len(g))

def run(side,data,output,model_root,warmup,draws):
    d=prepare(data,side);cal=d.loc[d.split.eq('calibration')].copy();validation=d.loc[d.split.eq('validation')].copy();test=d.loc[d.split.eq('test')].copy()
    select,interval_cal=robust.split_validation(validation)
    print(side,'training',len(cal),'selection',len(select),'interval',len(interval_cal),'test',len(test),flush=True)
    previous_meta,previous_arrays=old.load(Path(model_root)/side/'bayesian')
    def previous(f):
        f=f.copy();f['age']=(pd.to_datetime(f.date,format='mixed',utc=True).dt.year-pd.to_numeric(f.year)).clip(0,150)
        f['model']=f.model.fillna('None');f['manufacturer']=f.manufacturer.fillna('None')
        return old.predict(previous_meta,previous_arrays,f)
    old_mae=scores(select.observed_price,previous(select).corrected_price)['mae_usd']
    candidates=[];bundles={}
    for likelihood in ['gaussian','student_t','mixture']:
        meta,arrays=robust.fit(cal,likelihood,warmup=warmup,draws=draws)
        predictions=robust.predict(meta,arrays,select);logp=robust.pointwise_log_predictive(meta,arrays,select)
        entry=dict(likelihood=likelihood,validation_mae_usd=scores(select.observed_price,predictions.corrected_price)['mae_usd'],
            validation_elpd_log_price=float(logp.sum()),validation_mean_log_density=float(logp.mean()),max_split_rhat=float(max(np.max(v) for v in meta['rhat'].values())))
        candidates.append(entry);bundles[likelihood]=(meta,arrays);print(side,json.dumps(entry),flush=True)
    eligible=[c for c in candidates if c['max_split_rhat']<1.1];best=min(eligible,key=lambda c:c['validation_mae_usd']) if eligible else None
    report=dict(market=side,method=robust.METHOD,selection_rows=len(select),interval_calibration_rows=len(interval_cal),
        validation_base_mae_usd=scores(select.observed_price,select.base_price)['mae_usd'],validation_previous_bayes_mae_usd=old_mae,candidates=candidates,
        selection_rule='lowest validation dollar MAE among candidates with monitored split R-hat <1.1; test evaluated only after selection',
        density_comparison='held-out selection log-price predictive ELPD, not PSIS-LOO',
        training_limits='Retained 75% OOF calibration after fixed training-only selection; no further deletion. Selection-conditional OOF, not nested evaluation of trimming. Original validation/test unchanged.',
        base_model_identity=hashlib.sha256((Path(model_root)/side/'evaluation.json').read_bytes()).hexdigest())
    dest=Path(output)/side;dest.mkdir(parents=True,exist_ok=True)
    if best is None:report.update(promoted=False,reason='No candidate passed monitored convergence')
    else:
        likelihood=best['likelihood'];meta,arrays=bundles[likelihood]
        meta,multiplier=robust.calibrate_intervals(meta,arrays,interval_cal)
        pt=robust.predict(meta,arrays,test);low,high=old.interval(pt,multiplier);truth=test.observed_price.to_numpy()
        old_pt=previous(test);versus_old=paired_bootstrap(test,pt.corrected_price.to_numpy(),old_pt.corrected_price.to_numpy())
        versus_base=paired_bootstrap(test,pt.corrected_price.to_numpy(),test.base_price.to_numpy());covered=(truth>=low)&(truth<=high);coverage=float(covered.mean())
        promoted=bool(best['validation_mae_usd']<old_mae and versus_old['group_bootstrap_95_usd'][0]>0 and versus_base['group_bootstrap_95_usd'][0]>0 and .88<=coverage<=.95)
        report.update(selected_likelihood=likelihood,point_scores=scores(truth,pt.corrected_price),previous_bayesian_scores=scores(truth,old_pt.corrected_price),
            base_test_mae_usd=scores(truth,test.base_price)['mae_usd'],test_rows=len(test),nominal_coverage=.9,test_coverage=coverage,
            validation_rows=len(interval_cal),validation_multiplier=multiplier,test_median_interval_width_usd=float(np.median(high-low)),
            test_median_interval_ratio=float(np.median(high/low)),improvement_over_previous=versus_old,improvement_over_regular=versus_base,promoted=promoted,
            conditional_interval_calibration=meta['conditional_interval_calibration'],
            limits='Historical outcomes only. Current-date transport coverage unvalidated. Separate markets, not an observed dealer spread. This unchanged test cohort was examined in earlier releases; not a new external benchmark.',
            promotion_rule='validation improvement; paired group-bootstrap lower 95% MAE savings >0 vs previous and regular; monitored R-hat <1.1; historical coverage 88-95%',
            model='Cross-classified nonlinear heteroscedastic Bayesian '+likelihood+' log-residual correction')
        masks={'rare_or_unseen_models':pt.model_n.to_numpy()<12,'well_represented_models':pt.model_n.to_numpy()>=100,
            'base_under_5000':test.base_price.to_numpy()<5000,'base_over_25000':test.base_price.to_numpy()>25000,
            'over_150000_miles':robust.number(test,'odometer',100000)>150000,'older_than_12_years':test.age.to_numpy()>12}
        report['test_segments']={}
        for name,mask in masks.items():
            if mask.any():report['test_segments'][name]=dict(rows=int(mask.sum()),regular_mae_usd=float(np.mean(abs(truth[mask]-test.base_price.to_numpy()[mask]))),
                previous_mae_usd=float(np.mean(abs(truth[mask]-old_pt.corrected_price.to_numpy()[mask]))),new_mae_usd=float(np.mean(abs(truth[mask]-pt.corrected_price.to_numpy()[mask]))),coverage=float(covered[mask].mean()))
        report['test_interval_strata']={name:dict(rows=int((pt.interval_calibration_group==name).sum()),coverage=float(covered[(pt.interval_calibration_group==name).to_numpy()].mean())) for name in pt.interval_calibration_group.unique()}
        meta['base_evaluation_sha256']=report['base_model_identity'];old.save(dest/'bayesian',meta,arrays,report)
        pd.DataFrame(dict(row_id=test.row_id,observed=truth,regular=test.base_price,previous_bayesian=old_pt.corrected_price.to_numpy(),new_bayesian=pt.corrected_price.to_numpy(),low=low,high=high)).to_csv(dest/'test-predictions.csv.gz',index=False)
    (dest/'comparison.json').write_text(json.dumps(report,indent=2));print('FINAL',side,json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--model-root',default='models/xgboost-hierarchy')
    p.add_argument('--market',choices=['ask','bid']);p.add_argument('--warmup',type=int,default=240);p.add_argument('--draws',type=int,default=240);a=p.parse_args()
    for side in ['ask','bid']:
        if not a.market or a.market==side:run(side,Path(a.data)/(side+'-minilm-cleaned.csv.gz'),a.output,a.model_root,a.warmup,a.draws)
