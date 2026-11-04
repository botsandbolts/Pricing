"""Validation-only shrinkage challenger over the existing Bayesian correction."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from pricing import bayesian_hierarchy as bayes
from pricing import robust_hierarchy as robust

def choose_weight(y,base,delta):
    grid=np.arange(0.,2.01,.25)
    errors=np.array([np.mean(np.abs(y-base*np.exp(w*delta))) for w in grid])
    # Prefer the deployed correction when validation scores tie.
    ties=np.flatnonzero(errors<=errors.min()+1e-9)
    return float(grid[ties[np.argmin(abs(grid[ties]-1))]])
def choose_calibration(y,base,delta):
    """Dollar-MAE optimal bounded multiplier for each candidate strength."""
    candidates=[]
    for weight in np.arange(0.,2.01,.25):
        pred=base*np.exp(np.clip(weight*delta,-5,5))
        ratios=y/pred;order=np.argsort(ratios)
        # Weighted median is the exact minimizer of sum |y - multiplier*pred|.
        cumulative=np.cumsum(pred[order])
        index=np.searchsorted(cumulative,cumulative[-1]/2)
        multiplier=float(np.clip(ratios[order[index]],np.exp(-.1),np.exp(.1)))
        error=float(np.mean(abs(y-pred*multiplier)))
        candidates.append((error,abs(weight-1)+abs(np.log(multiplier)),float(weight),multiplier))
    _,_,weight,multiplier=min(candidates)
    return {'weight':weight,'multiplier':multiplier}

def weights_for(strata,weights):
    return np.array([weights['strata'].get(s,weights['global']) for s in strata])
def quantile(values):
    n=len(values);rank=min(n,int(np.ceil((n+1)*.9)))
    return float(np.partition(values,rank-1)[rank-1])

def run(side):
    folder=Path('training-cache/lemon-hierarchy')/side
    prepared=folder.parent/'inputs'/(side+'-minilm-cleaned.csv.gz')
    d=pd.read_csv(prepared,dtype={'row_id':str,'vehicle_group':str})
    assert d.row_id.is_unique and d.groupby('vehicle_group').split.nunique().max()==1
    assert np.allclose(d.residual_log,np.log(d.observed_price/d.predicted_price))
    date=pd.to_datetime(d.date,format='mixed',utc=True)
    d['age']=(date.dt.year+(date.dt.dayofyear-1)/365.25-pd.to_numeric(d.year)).clip(0,150)
    d['base_price']=d.predicted_price
    d['auction_grade']=pd.to_numeric(d['auction_condition'] if 'auction_condition' in d else d.get('condition'),errors='coerce') if side=='bid' else np.nan
    if side=='bid':d['condition']='unknown'
    meta,arrays=bayes.load(folder/'bayesian')
    assert meta.get('method')==robust.METHOD
    validation=d[d.split.eq('validation')].copy();test=d[d.split.eq('test')].copy()
    select,cal=robust.split_validation(validation)
    ps=bayes.predict(meta,arrays,select)
    strata=robust.interval_strata(select,ps.model_n.to_numpy())
    y=select.observed_price.to_numpy();base=ps.base_price.to_numpy();delta=ps.delta_log.to_numpy()
    weights={'global':choose_calibration(y,base,delta),'strata':{}}
    for s in np.unique(strata):
        mask=strata==s
        if mask.sum()>=500:weights['strata'][s]=choose_calibration(y[mask],base[mask],delta[mask])
    def prediction(frame):
        p=bayes.predict(meta,arrays,frame)
        s=robust.interval_strata(frame,p.model_n.to_numpy())
        settings=[weights['strata'].get(k,weights['global']) for k in s]
        w=np.array([v['weight'] for v in settings]);mult=np.array([v['multiplier'] for v in settings])
        center=p.base_price.to_numpy()*np.exp(np.clip(w*p.delta_log.to_numpy(),-5,5))*mult
        return p,s,center
    pc,sc,center=prediction(cal)
    err=np.abs(np.log(cal.observed_price.to_numpy()/center))/pc.predictive_sd_log.to_numpy()
    scale={'global':quantile(err),'strata':{}}
    for s in np.unique(sc):
        mask=sc==s
        if mask.sum()>=100:scale['strata'][s]=quantile(err[mask])
    pt,st,center=prediction(test);truth=test.observed_price.to_numpy()
    width=np.array([scale['strata'].get(s,scale['global']) for s in st])*pt.predictive_sd_log.to_numpy()
    low=center*np.exp(-width);high=center*np.exp(width)
    current=pt.corrected_price.to_numpy()
    old_mae=float(np.mean(abs(truth-current)));new_mae=float(np.mean(abs(truth-center)))
    original=json.loads((folder/'bayesian/bayesian-evaluation.json').read_text())
    if not np.isclose(old_mae,original['point_scores']['mae_usd'],rtol=1e-8,atol=1e-6):
        raise ValueError('Challenger baseline does not reproduce the installed Bayesian evaluation')
    savings=abs(truth-current)-abs(truth-center)
    groups=pd.DataFrame({'g':test.vehicle_group,'savings':savings}).groupby('g').savings.agg(['sum','count'])
    rng=np.random.default_rng(927);boot=[]
    for _ in range(400):
        ix=rng.integers(0,len(groups),len(groups))
        boot.append(groups['sum'].to_numpy()[ix].sum()/groups['count'].to_numpy()[ix].sum())
    ci=np.quantile(boot,[.025,.975]).tolist()
    coverage=float(np.mean((truth>=low)&(truth<=high)))
    report={'market':side,'posterior_sha256':meta['posterior_sha256'],'base_evaluation_sha256':meta['base_evaluation_sha256'],'baseline_reproduced':True,'status':'Completed challenger evaluation; existing production posterior unchanged',
      'method':'Risk-stratum validation calibration of saved Bayesian log correction: strength 0 to 2 in .25 increments plus a dollar-MAE weighted-median multiplier bounded to exp(+/-0.1); at least 500 selection cars per stratum, global fallback. This is a calibration layer, not a new posterior.',
      'weights':weights,'interval_scales':scale,'selection_rows':len(select),'interval_calibration_rows':len(cal),'test_rows':len(test),
      'current_bayesian_test_mae_usd':old_mae,'challenger_test_mae_usd':new_mae,'mae_saved_usd':old_mae-new_mae,
      'paired_vehicle_bootstrap_95_saved_usd':ci,'test_coverage_90':coverage,'median_interval_width_usd':float(np.median(high-low)),
      'eligible_for_promotion':bool(ci[0]>0 and .88<=coverage<=.95),
      'limits':'Strength and price-level multiplier selected only on selection validation; intervals calibrated on different validation vehicles; test used after selection. Historical test reused in prior development, not a new external benchmark. Conditional intervals ignore uncertainty in selected shrinkage weights. No current-date coverage guarantee. No merchant identity correction added.'}
    dest=Path('reports/bayesian-challenger');dest.mkdir(parents=True,exist_ok=True)
    (dest/(side+'.json')).write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
if __name__=='__main__':
    for side in ['ask','bid']:run(side)
