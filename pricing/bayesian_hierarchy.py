"""Joint Gaussian multilevel posterior for out-of-fold forest log-price errors.

The forest is held fixed. Two independent Gibbs chains sample a *joint* posterior
for market, age, make, model-within-make, their variances and observation noise.
An untouched chronological validation set calibrates the predictive interval;
the test set measures its historical coverage without fitting any parameter.
"""
import hashlib
import gzip
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _ig(rng, shape, scale):
    return scale / rng.gamma(shape)


def fit(calibration, *, warmup=180, draws=160, seeds=(2718, 3141)):
    """Return compact posterior draws; all observed prices must be OOF targets."""
    if not calibration.prediction_kind.eq('oof').all():
        raise ValueError('Bayesian calibration requires out-of-fold forest predictions.')
    if calibration.base_fold.nunique() < 2:
        raise ValueError('At least two independent forest folds are required.')
    if warmup < 10 or draws < 20 or len(seeds) != 2:
        raise ValueError('Two chains with sufficient warmup and draws are required.')
    d = calibration.reset_index(drop=True)
    r = np.log(d.observed_price.to_numpy(float) / d.base_price.to_numpy(float))
    if not np.isfinite(r).all():
        raise ValueError('Invalid log-price residual.')
    makes = sorted(d.manufacturer.astype(str).str.lower().unique())
    models = sorted((d.manufacturer.astype(str).str.lower() + '::' + d.model.astype(str).str.lower()).unique())
    make_index = pd.Index(makes).get_indexer(d.manufacturer.astype(str).str.lower())
    model_index = pd.Index(models).get_indexer(d.manufacturer.astype(str).str.lower() + '::' + d.model.astype(str).str.lower())
    age_center = float(d.age.mean())
    x = (d.age.to_numpy(float) - age_center) / 10.
    n, nm, nk = len(d), len(makes), len(models)
    make_count = np.bincount(make_index, minlength=nm)
    model_count = np.bincount(model_index, minlength=nk)
    chains = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        mu = float(r.mean()); beta = 0.; a = np.zeros(nm); b = np.zeros(nk)
        sigma2 = max(float(r.var()), .01); tau_make2 = .04; tau_model2 = .04
        prediction = np.full(n, mu)
        saved = {key: [] for key in ('global', 'age', 'make', 'model', 'sigma2', 'make_tau2', 'model_tau2')}
        for iteration in range(warmup + draws):
            precision = float(x @ x) / sigma2 + 25.  # age / decade ~ N(0, .2**2)
            new = rng.normal(float(x @ (r - prediction + beta*x)) / sigma2 / precision, np.sqrt(1 / precision))
            prediction += (new - beta)*x; beta = new

            # Integrate each entire random-effect block out when drawing the
            # global intercept, then draw its effects conditionally. This is
            # exact blocked Gibbs and avoids a nearly immobile global mean.
            for index, counts, effects, variance, size in (
                (make_index, make_count, a, tau_make2, nm),
                (model_index, model_count, b, tau_model2, nk)):
                sums = np.bincount(index, weights=r-prediction+mu+effects[index], minlength=size)
                marginal_precision = counts/(sigma2+counts*variance)
                global_precision = 1. + marginal_precision.sum()
                global_mean = np.sum(sums/(sigma2+counts*variance))/global_precision
                new_mu = rng.normal(global_mean, np.sqrt(1/global_precision))
                conditional_precision = counts/sigma2+1/variance
                new_effects = rng.normal((sums-counts*new_mu)/sigma2/conditional_precision,
                                         np.sqrt(1/conditional_precision))
                prediction += new_mu-mu+(new_effects-effects)[index]
                mu = new_mu
                if size==nm:a=new_effects
                else:b=new_effects

            sigma2 = _ig(rng, 2 + n/2, .1 + float((r-prediction) @ (r-prediction))/2)
            tau_make2 = _ig(rng, 2 + nm/2, .04 + float(a @ a)/2)
            tau_model2 = _ig(rng, 2 + nk/2, .04 + float(b @ b)/2)
            if iteration >= warmup:
                for key, value in [('global', mu), ('age', beta), ('make', a), ('model', b),
                                   ('sigma2', sigma2), ('make_tau2', tau_make2), ('model_tau2', tau_model2)]:
                    saved[key].append(np.array(value, dtype=np.float32, copy=True))
        chains.append(saved)
    full = {k: np.concatenate([np.stack(c[k]) for c in chains]) for k in chains[0]}
    # Classical split-chain R-hat for scalar diagnostics. Report, never silently suppress.
    diagnostics = {}
    for k in ('global', 'age', 'sigma2', 'make_tau2', 'model_tau2'):
        v = np.stack([np.array(c[k], dtype=float) for c in chains]); within = v.var(axis=1, ddof=1).mean()
        between = draws * v.mean(axis=1).var(ddof=1)
        diagnostics[k] = float(np.sqrt(((draws-1)*within/draws + between/draws)/within)) if within else float('inf')
    # The full chains are used for diagnostics. Every fourth draw per chain is
    # enough for deterministic CPU serving; effects use compact float16 storage.
    thin = 4 if draws>=100 else 1
    arrays = {k: (v.reshape(2,draws,*v.shape[1:])[:,::thin].reshape(-1,*v.shape[1:])
                   .astype(np.float16 if k in ('make','model') else np.float32)) for k,v in full.items()}
    metadata = {'method':'joint_gibbs_hierarchical_gaussian_log_residuals',
                'priors':{'global':'Normal(0,1)', 'age_per_decade':'Normal(0,0.2)',
                          'make_effect':'Normal(0,make_tau2)', 'model_effect':'Normal(0,model_tau2)',
                          'sigma2':'InvGamma(2,0.1)', 'make_tau2':'InvGamma(2,0.04)',
                          'model_tau2':'InvGamma(2,0.04)'},
                'age_center':age_center, 'makes':makes, 'models':models,
                'make_counts':make_count.tolist(), 'model_counts':model_count.tolist(),
                'draws_per_chain':draws,'stored_draws_per_chain':len(arrays['global'])//2,
                'thinning':thin,'warmup_per_chain':warmup,'chains':2,
                'rhat':diagnostics,'calibration_rows':n,
                'base_model_id':str(d.base_model_id.iloc[0]),'price_basis':str(d.price_basis.iloc[0])}
    return metadata, arrays


def predict(metadata, arrays, frame, *, batch=512):
    """Integrate joint parameter draws; return log predictive SD incl. car noise."""
    make_index = pd.Index(metadata['makes']).get_indexer(frame.manufacturer.astype(str).str.lower())
    model_keys = frame.manufacturer.astype(str).str.lower() + '::' + frame.model.astype(str).str.lower()
    model_index = pd.Index(metadata['models']).get_indexer(model_keys)
    age = (frame.age.to_numpy(float)-metadata['age_center'])/10.
    base = frame.base_price.to_numpy(float)
    means = np.empty(len(frame));variances = np.empty(len(frame))
    draws = len(arrays['global'])
    for begin in range(0,len(frame),batch):
        end=min(len(frame),begin+batch)
        delta = arrays['global'][:,None].astype(float) + arrays['age'][:,None]*age[None,begin:end]
        mi = make_index[begin:end]; ki = model_index[begin:end]
        if (mi>=0).any():
            mask=mi>=0;delta[:,mask] += arrays['make'][:,mi[mask]]
        if (ki>=0).any():
            mask=ki>=0;delta[:,mask] += arrays['model'][:,ki[mask]]
        means[begin:end]=delta.mean(axis=0)
        variances[begin:end]=delta.var(axis=0,ddof=1)
        if (mi<0).any():variances[begin:end][mi<0] += float(np.mean(arrays['make_tau2']))
        if (ki<0).any():variances[begin:end][ki<0] += float(np.mean(arrays['model_tau2']))
    predictive_sd=np.sqrt(variances+float(np.mean(arrays['sigma2'])))
    result=pd.DataFrame({'base_price':base,'corrected_price':base*np.exp(means),
                         'delta_log':means,'predictive_sd_log':predictive_sd,
                         'make_n':np.where(make_index>=0,np.array(metadata['make_counts'])[np.maximum(make_index,0)],0),
                         'model_n':np.where(model_index>=0,np.array(metadata['model_counts'])[np.maximum(model_index,0)],0)},index=frame.index)
    return result


def interval(prediction, multiplier):
    width=np.asarray(prediction.predictive_sd_log)*multiplier
    price=np.asarray(prediction.corrected_price)
    return np.maximum(1,price*np.exp(-width)), np.minimum(1e7,price*np.exp(width))


def assess(metadata,arrays,validation,test):
    """Calibrate on validation only; report untouched chronological test coverage."""
    from pricing.hierarchical import scores
    pv=predict(metadata,arrays,validation);pt=predict(metadata,arrays,test)
    residual=np.abs(np.log(validation.observed_price.to_numpy(float)/pv.corrected_price.to_numpy(float))) / pv.predictive_sd_log.to_numpy(float)
    # Finite-sample split-conformal quantile; posterior variance sets local scale.
    level=.9;n=len(residual);rank=min(n,int(np.ceil((n+1)*level)))
    multiplier=float(np.partition(residual,rank-1)[rank-1])
    low,high=interval(pt,multiplier)
    y=test.observed_price.to_numpy(float)
    summary={'nominal_coverage':level,'validation_rows':n,'test_rows':len(test),
             'validation_multiplier':multiplier,'test_coverage':float(np.mean((low<=y)&(y<=high))),
             'test_median_interval_width_usd':float(np.median(high-low)),
             'test_median_interval_ratio':float(np.median(high/low)),
             'point_scores':scores(y,pt.corrected_price),
             'limits':'Historical individual outcome interval only; no current-date market/age transport validation or causal interval.'}
    return summary,pv,pt


def save(folder,metadata,arrays,report):
    import gzip
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    packed=io.BytesIO();np.savez_compressed(packed,**arrays)
    from pricing.suite_storage import split_model
    for previous in folder.glob('posterior.npz.*.part'):previous.unlink()
    (folder/'posterior.npz.parts.json').unlink(missing_ok=True)
    (folder/'posterior.npz').write_bytes(packed.getvalue())
    if len(packed.getvalue()) > 480000:
        split_model(folder/'posterior.npz',part_bytes=480000)
    metadata={**metadata,'posterior_sha256':hashlib.sha256(packed.getvalue()).hexdigest(),
              'interval':{'level':.9,'multiplier':report['validation_multiplier']}}
    with gzip.open(folder/'bayesian.json.gz','wt') as file:
        json.dump(metadata,file,separators=(',',':'))
    (folder/'bayesian-evaluation.json').write_text(json.dumps(report,indent=2))


def load(folder):
    from pricing.suite_storage import read_model_file
    folder=Path(folder)
    with gzip.open(folder/'bayesian.json.gz','rt') as file:metadata=json.load(file)
    packed=read_model_file(folder/'posterior.npz')
    if hashlib.sha256(packed).hexdigest()!=metadata['posterior_sha256']:
        raise ValueError('Bayesian posterior checksum mismatch')
    with np.load(io.BytesIO(packed),allow_pickle=False) as archive:
        arrays={k:archive[k] for k in archive.files}
    return metadata,arrays
