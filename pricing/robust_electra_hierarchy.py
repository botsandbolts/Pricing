"""Cross-classified heteroscedastic Bayesian calibration of OOF log prices.

Gaussian, Student-t(4) and a labelled two-scale contamination mixture share
one mean structure. Two conditional Gibbs chains sample genuine posteriors.
"""
import hashlib
import numpy as np
import pandas as pd
from scipy.special import expit, gammaln, logsumexp

METHOD='robust_cross_classified_gibbs_electra_umap_v1'
CATEGORIES=('manufacturer','model','state','type','transmission','fuel','drive','title_status','condition','cylinders','paint_color')

def text(frame,col):
    value=frame[col] if col in frame else pd.Series('unknown',index=frame.index)
    return value.fillna('unknown').astype(str).str.strip().str.lower().replace('','unknown')

def number(frame,col,fill):
    value=frame[col] if col in frame else pd.Series(fill,index=frame.index)
    return pd.to_numeric(value,errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(fill).to_numpy(float)

def keys(frame,col,specification=None):
    values=text(frame,col)
    if col=='model':
        values=text(frame,'manufacturer')+'::'+values
        if specification is not None:
            values=values.where(values.isin(specification['frequent']),text(frame,'manufacturer')+'::[pooled rare models]')
    return values

def design(frame):
    age=np.clip(number(frame,'age',8),0,35)
    miles=np.log1p(np.clip(number(frame,'odometer',100000),0,2000000))
    price=np.log(np.clip(number(frame,'base_price',10000),1,1e7))
    grade=number(frame,'auction_grade',np.nan)
    cols=[np.ones(len(frame)),(age-8)/10,(miles-np.log1p(100000))/2,(price-9.5)/2,np.nan_to_num((grade-3)/2),np.isnan(grade).astype(float)]
    names=['intercept','age','log_mileage','log_base_price','auction_grade','grade_missing']
    for values,knots,scale,label in [(age,[3,10,18],10.,'age'),(miles,np.log1p([40000,120000,200000]),2.,'mileage'),(price,[8.5,9.5,10.5],2.,'price')]:
        for knot in knots:
            cols.append(np.maximum(0,values-knot)/scale);names.append(label+'_hinge_'+str(round(float(knot),3)))
    cols.append(cols[1]*cols[2]);names.append('age_by_mileage')
    for j in range(32):
        name=f'electra_umap_{j}'
        if name not in frame:raise ValueError('Missing hierarchical UMAP feature: '+name)
        values=frame[name].to_numpy(float)
        if not np.isfinite(values).all():raise ValueError('Non-finite hierarchical UMAP feature')
        cols.append(values);names.append(name)
    return np.asfortranarray(np.column_stack(cols)),names

def cell(frame):
    price=np.log(np.clip(number(frame,'base_price',10000),1,1e7))
    return np.digitize(price,[8.5,9.5,10.5])*3+np.digitize(number(frame,'age',8),[5,12])

def rhat(chains):
    a=np.asarray(chains,float);n=a.shape[1]//2
    split=np.concatenate([a[:,:n],a[:,-n:]],axis=0)
    w=split.var(axis=1,ddof=1).mean(axis=0);b=n*split.mean(axis=1).var(axis=0,ddof=1)
    return np.sqrt(np.maximum(1,((n-1)*w/n+b/n)/np.maximum(w,1e-15)))

def fit(frame,likelihood='student_t',*,warmup=240,draws=240,seeds=(2718,3141)):
    if likelihood not in ('gaussian','student_t','mixture'):raise ValueError('Unsupported likelihood')
    if not frame.prediction_kind.eq('oof').all() or frame.base_fold.nunique()<2:raise ValueError('Genuine multi-fold OOF calibration is required')
    if len(seeds)!=2 or warmup<10 or draws<20:raise ValueError('Two chains and sufficient draws required')
    d=frame.reset_index(drop=True);r=np.log(number(d,'observed_price',np.nan)/number(d,'base_price',np.nan))
    if not np.isfinite(r).all():raise ValueError('Invalid log residuals')
    x,names=design(d);n,p=x.shape;cells=cell(d);specs={};indices={}
    for col in CATEGORIES:
        spec={}
        if col=='model':
            counts=keys(d,col).value_counts();spec['frequent']=sorted(counts[counts>=12].index.tolist())
        values=keys(d,col,spec);levels=sorted(values.unique().tolist())
        if len(levels)<2:continue
        index=pd.Index(levels).get_indexer(values)
        spec.update(levels=levels,counts=np.bincount(index,minlength=len(levels)).tolist());specs[col]=spec;indices[col]=index
    slope_col='manufacturer';slope_index=indices.get(slope_col);z=(number(d,'age',8)-8)/10;chains=[]
    for seed in seeds:
        rng=np.random.default_rng(seed);beta=np.zeros(p);beta[0]=np.median(r)
        effects={c:np.zeros(len(s['levels'])) for c,s in specs.items()};taus={c:.01 for c in specs}
        slopes=np.zeros(len(specs[slope_col]['levels'])) if slope_index is not None else np.zeros(1);slope_tau=.0025
        sigma=np.full(12,max(float(np.var(r)),.01));latent=np.ones(n);pi=.1;pred=x@beta
        saved={k:[] for k in ['beta','sigma2','pi','slope','slope_tau2']+['effect_'+c for c in specs]+['tau_'+c for c in specs]}
        for iteration in range(warmup+draws):
            error=r-pred
            if likelihood=='student_t':latent=rng.gamma(2.5,2/(4+error**2/sigma[cells]))
            elif likelihood=='mixture':
                logit=np.log(pi/(1-pi))-np.log(5.)+.5*error**2/sigma[cells]*(1-1/25)
                wide=rng.random(n)<expit(logit);latent=np.where(wide,1/25.,1.);pi=rng.beta(1+wide.sum(),9+n-wide.sum())
            weights=latent/sigma[cells]
            precision=x.T@(weights[:,None]*x)+np.diag([1]+[25]*(p-1));rhs=x.T@(weights*(r-pred+x@beta))
            chol=np.linalg.cholesky(precision);mean=np.linalg.solve(chol.T,np.linalg.solve(chol,rhs))
            new=mean+np.linalg.solve(chol.T,rng.normal(size=p));pred+=x@(new-beta);beta=new
            for col,index in indices.items():
                old=effects[col];k=len(old)
                sums=np.bincount(index,weights=weights*(r-pred+beta[0]+old[index]),minlength=k)
                counts=np.bincount(index,weights=weights,minlength=k);denom=1+counts*taus[col];precision=1+np.sum(counts/denom)
                intercept=rng.normal(np.sum(sums/denom)/precision,np.sqrt(1/precision));precision_e=counts+1/taus[col]
                new=rng.normal((sums-counts*intercept)/precision_e,np.sqrt(1/precision_e))
                pred+=intercept-beta[0]+(new-old)[index];beta[0]=intercept;effects[col]=new
                taus[col]=(.01+float(new@new)/2)/rng.gamma(2+k/2)
            if slope_index is not None:
                k=len(slopes);counts=np.bincount(slope_index,weights=weights*z*z,minlength=k)
                sums=np.bincount(slope_index,weights=weights*z*(r-pred+(beta[1]+slopes[slope_index])*z),minlength=k)
                denom=1+counts*slope_tau;global_precision=25+np.sum(counts/denom)
                age_slope=rng.normal(np.sum(sums/denom)/global_precision,np.sqrt(1/global_precision));precision=counts+1/slope_tau
                new=rng.normal((sums-counts*age_slope)/precision,np.sqrt(1/precision))
                pred+=((age_slope-beta[1])+(new-slopes)[slope_index])*z;beta[1]=age_slope;slopes=new
                slope_tau=(.0025+float(new@new)/2)/rng.gamma(2+k/2)
            error=r-pred;counts=np.bincount(cells,minlength=12);sums=np.bincount(cells,weights=latent*error**2,minlength=12)
            sigma=(.1+sums/2)/rng.gamma(3+counts/2)
            if iteration>=warmup:
                values=dict(beta=beta,sigma2=sigma,pi=pi,slope=slopes,slope_tau2=slope_tau)
                values.update({'effect_'+c:v for c,v in effects.items()});values.update({'tau_'+c:v for c,v in taus.items()})
                for key,value in values.items():saved[key].append(np.array(value,dtype=np.float32,copy=True))
        chains.append({k:np.asarray(v) for k,v in saved.items()});print(likelihood,'chain',seed,'completed',flush=True)
    diagnostics={k:np.asarray(rhat([c[k] for c in chains])).tolist() for k in ['beta','sigma2','slope_tau2']+['tau_'+c for c in specs]}
    thin=max(1,draws//40);arrays={k:np.concatenate([c[k][::thin] for c in chains]).astype(np.float32) for k in chains[0]}
    arrays['global']=arrays['beta'][:,0]
    meta=dict(method=METHOD,likelihood=likelihood,nu=4,contamination_scale=5,groups=specs,basis=names,slope_group=slope_col,
        raw_model_counts={str(k):int(v) for k,v in keys(d,'model').value_counts().items()},warmup_per_chain=warmup,draws_per_chain=draws,
        chains=2,thinning=thin,rhat=diagnostics,calibration_rows=n,base_model_id=str(d.base_model_id.iloc[0]),price_basis=str(d.price_basis.iloc[0]),
        priors={'coefficients':'intercept N(0,1); splines N(0,.2^2)','effects':'N(0,tau2); tau2~InvGamma(2,.01)',
        'make_age_slopes':'N(0,tau_slope2); tau_slope2~InvGamma(2,.0025)','cell_variance':'InvGamma(3,.1)','mixture_weight':'Beta(1,9)'})
    return meta,arrays

def locations(meta,arrays,frame):
    x,_=design(frame);delta=arrays['beta']@x.T;extra=np.zeros(len(frame));make_n=np.zeros(len(frame),int)
    for col,spec in meta['groups'].items():
        index=pd.Index(spec['levels']).get_indexer(keys(frame,col,spec));good=index>=0
        delta[:,good]+=arrays['effect_'+col][:,index[good]];extra[~good]+=float(np.mean(arrays['tau_'+col]))
        if col=='manufacturer':make_n[good]=np.asarray(spec['counts'])[index[good]]
        if col==meta['slope_group']:
            z=(number(frame,'age',8)-8)/10;delta[:,good]+=arrays['slope'][:,index[good]]*z[None,good]
            extra[~good]+=float(np.mean(arrays['slope_tau2']))*z[~good]**2
    model_n=keys(frame,'model').map(meta['raw_model_counts']).fillna(0).to_numpy(int)
    return delta,extra,make_n,model_n

def predict(meta,arrays,frame,*,batch=512):
    output=[]
    for start in range(0,len(frame),batch):
        f=frame.iloc[start:start+batch];delta,extra,mn,kn=locations(meta,arrays,f);center=np.median(delta,axis=0)
        s2=arrays['sigma2'][:,cell(f)].mean(axis=0)
        if meta['likelihood']=='student_t':s2*=meta['nu']/(meta['nu']-2)
        elif meta['likelihood']=='mixture':s2*=1+24*float(arrays['pi'].mean())
        variance=delta.var(axis=0,ddof=1)+extra+s2;base=number(f,'base_price',10000)
        output.append(pd.DataFrame(dict(base_price=base,corrected_price=base*np.exp(np.clip(center,-5,5)),delta_log=center,
            predictive_sd_log=np.sqrt(variance),make_n=mn,model_n=kn),index=f.index))
    result=pd.concat(output);calibration=meta.get('conditional_interval_calibration')
    if calibration:
        strata=interval_strata(frame,result.model_n.to_numpy())
        for field,key in [('interval_multiplier','multiplier'),('interval_calibration_n','rows'),('interval_calibration_source','source')]:
            result[field]=[calibration['strata'].get(k,calibration['global'])[key] for k in strata]
        result['interval_calibration_group']=strata
    return result

def interval_strata(frame,support):
    rare=np.asarray(support)<12;low=number(frame,'base_price',10000)<5000
    older=(number(frame,'age',8)>12)|(number(frame,'odometer',100000)>150000)
    return np.asarray([f"{'rare' if r else 'common'}|{'under5000' if l else '5000plus'}|{'older_highmiles' if o else 'other'}" for r,l,o in zip(rare,low,older)])

def calibrate_intervals(meta,arrays,frame,minimum=100):
    prediction=predict(meta,arrays,frame)
    errors=np.abs(np.log(number(frame,'observed_price',1)/prediction.corrected_price.to_numpy()))/prediction.predictive_sd_log.to_numpy()
    def receipt(values,source):
        rank=min(len(values),int(np.ceil((len(values)+1)*.9)))
        return dict(multiplier=float(np.partition(values,rank-1)[rank-1]),rows=len(values),source=source)
    global_receipt=receipt(errors,'all interval-calibration cars');strata=interval_strata(frame,prediction.model_n.to_numpy());groups={}
    rarity=np.asarray([s.split('|')[0] for s in strata])
    for name in sorted(set(strata)):
        mask=strata==name;broader=rarity==name.split('|')[0]
        groups[name]=receipt(errors[mask],'matching risk stratum') if mask.sum()>=minimum else (receipt(errors[broader],'pooled rarity class') if broader.sum()>=minimum else dict(global_receipt))
    for r in ['rare','common']:
        broader=rarity==r;fallback=receipt(errors[broader],'pooled rarity class') if broader.sum()>=minimum else dict(global_receipt)
        for price in ['under5000','5000plus']:
            for old in ['older_highmiles','other']:groups.setdefault(r+'|'+price+'|'+old,dict(fallback))
    result=dict(meta);result['conditional_interval_calibration']=dict(level=.9,minimum_stratum_rows=minimum,global_receipt=global_receipt,strata=groups,
        method='Mondrian split conformal: rarity x predicted-price<5000 x (age>12 or miles>150000); rarity/global fallback',
        limits='Requires within-stratum exchangeability; fallback does not prove sparse-group or current-date coverage.')
    result['conditional_interval_calibration']['global']=result['conditional_interval_calibration'].pop('global_receipt')
    return result,global_receipt['multiplier']

def pointwise_log_predictive(meta,arrays,frame,batch=256):
    """Held-out log residual predictive density, not LOO. Unseen-group variance
    convolution is an approximation for the non-Gaussian candidates."""
    result=[]
    for start in range(0,len(frame),batch):
        f=frame.iloc[start:start+batch];delta,extra,_,_=locations(meta,arrays,f)
        y=np.log(number(f,'observed_price',1)/number(f,'base_price',1));variance=arrays['sigma2'][:,cell(f)]+extra
        error=(y[None,:]-delta)**2/variance;normal=-.5*(np.log(2*np.pi*variance)+error)
        if meta['likelihood']=='student_t':
            nu=meta['nu'];lp=gammaln((nu+1)/2)-gammaln(nu/2)-.5*np.log(nu*np.pi*variance)-(nu+1)/2*np.log1p(error/nu)
        elif meta['likelihood']=='mixture':
            pi=arrays['pi'][:,None];wide=-.5*(np.log(2*np.pi*variance*25)+error/25)
            lp=np.logaddexp(np.log1p(-pi)+normal,np.log(pi)+wide)
        else:lp=normal
        result.append(logsumexp(lp,axis=0)-np.log(len(lp)))
    return np.concatenate(result)

def split_validation(frame):
    values=frame.vehicle_group.astype(str).map(lambda s:int(hashlib.sha256(('bayes-v2:'+s).encode()).hexdigest()[:8],16)%3)
    return frame.loc[values!=0].copy(),frame.loc[values==0].copy()
