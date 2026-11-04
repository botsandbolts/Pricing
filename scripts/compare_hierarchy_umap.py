"""Compare Bayesian UMAP effects with an identical frozen OOF XGBoost base."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from umap import UMAP
from sklearn.preprocessing import StandardScaler
from pricing import robust_hierarchy as baseline
from pricing import robust_umap_hierarchy as augmented
from pricing.bayesian_hierarchy import interval, save
from pricing.hierarchical import scores
from scripts.compare_robust_bayes import prepare, paired_bootstrap

def main():
    out=Path('hierarchy-umap-results');out.mkdir(exist_ok=True)
    d=prepare('frozen-baseline/ask/hierarchy-input.csv.gz','ask')
    calibration=d.split.eq('calibration')
    archive=np.load('bge-shards/merged.npz')
    ids=pd.Index(archive['ids'].astype(str))
    if ids.has_duplicates:raise ValueError('Duplicate embedding IDs')
    lookup=ids.get_indexer(d.row_id.astype(str))
    if (lookup<0).any():raise ValueError('Missing embedding row')
    vectors=archive['embeddings'][lookup]
    reducer=UMAP(n_components=32,n_neighbors=30,min_dist=.1,metric='cosine',
                 n_epochs=200,random_state=42,transform_seed=42,n_jobs=1,low_memory=True)
    print('Fit cosine UMAP on calibration descriptions only',flush=True)
    coords=np.empty((len(d),32),dtype=np.float32)
    coords[calibration]=reducer.fit_transform(vectors[calibration])
    coords[~calibration]=reducer.transform(vectors[~calibration])
    scaler=StandardScaler().fit(coords[calibration])
    coords=scaler.transform(coords)
    if not np.isfinite(coords).all():raise ValueError('Non-finite UMAP output')
    for j in range(32):d[f'bge_umap_{j}']=coords[:,j]
    joblib.dump(reducer,out/'calibration-umap.joblib')
    joblib.dump(scaler,out/'calibration-scaler.joblib')
    cal=d.loc[calibration].copy()
    select,interval_cal=baseline.split_validation(d.loc[d.split.eq('validation')])
    test=d.loc[d.split.eq('test')].copy()
    rows=[];predictions={};details={}
    for arm,module in [('baseline',baseline),('hierarchy_umap',augmented)]:
        candidates=[];bundles={}
        for likelihood in ['gaussian','student_t','mixture']:
            print(arm,likelihood,'fitting',flush=True)
            meta,arrays=module.fit(cal,likelihood,warmup=240,draws=240)
            pt=module.predict(meta,arrays,select)
            entry={'likelihood':likelihood,
                   'validation_mae_usd':scores(select.observed_price,pt.corrected_price)['mae_usd'],
                   'max_split_rhat':float(max(np.max(v) for v in meta['rhat'].values()))}
            candidates.append(entry);bundles[likelihood]=(meta,arrays)
        eligible=[c for c in candidates if c['max_split_rhat']<1.1]
        details[arm]={'candidates':candidates}
        if not eligible:
            raise ValueError(arm+': no candidate passed convergence; no result published')
        best=min(eligible,key=lambda c:c['validation_mae_usd'])
        meta,arrays=bundles[best['likelihood']]
        meta,multiplier=module.calibrate_intervals(meta,arrays,interval_cal)
        pt=module.predict(meta,arrays,test)
        low,high=interval(pt,multiplier)
        predictions[arm]=pt.corrected_price.to_numpy()
        report={'point_scores':scores(test.observed_price,pt.corrected_price),
                'test_rows':len(test),'test_coverage':float(((test.observed_price>=low)&(test.observed_price<=high)).mean()),
                'selected_likelihood':best['likelihood']}
        details[arm].update(report)
        save(out/arm,meta,arrays,report)
        rows.append({'Model':arm,'MAE ($)':report['point_scores']['mae_usd'],
                     '90% coverage':report['test_coverage'],'Test cars':len(test)})
    paired=paired_bootstrap(test,predictions['hierarchy_umap'],predictions['baseline'])
    report={'comparison':rows,'paired_improvement':paired,'details':details,
            'architecture':'Identical frozen vehicle facts + MiniLM-price XGBoost AFT predictions in both arms. UMAP coordinates enter only the Bayesian residual mean in hierarchy_umap.',
            'umap':{'metric':'cosine','n_components':32,'n_neighbors':30,'min_dist':.1,'n_epochs':200,
                    'fit_scope':'calibration descriptions only; validation and test transformed only',
                    'scaling':'calibration-only StandardScaler; Bayesian coefficients N(0,0.2^2)'},
            'previous_xgboost_umap':json.loads(Path('reports/bge-umap-experiment/comparison.json').read_text()),
            'limits':'Historical, previously examined test cohort; fixed earlier Lemon cleaning. Likelihood chosen on selection-validation cars, intervals calibrated on separate validation cars. No current-market or serving-readiness claim.'}
    dest=Path('reports/hierarchy-umap-experiment');dest.mkdir(parents=True,exist_ok=True)
    (dest/'comparison.json').write_text(json.dumps(report,indent=2))
    pd.DataFrame({'row_id':test.row_id,'vehicle_group':test.vehicle_group,'observed':test.observed_price,
                  **predictions}).to_csv(out/'test-predictions.csv.gz',index=False)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
