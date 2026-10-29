"""Train categorical-token TF-IDF groups; tune validation coverage, freeze test."""
import argparse, hashlib, json, time
from pathlib import Path
from importlib.metadata import version
import joblib
import numpy as np
import pandas as pd
import hdbscan
from pricing.advert_clusters import clean_description, encode, fit, distinctive_terms
from pricing.suite_storage import split_model


def coverage(clusterer, embedding, usable, n):
    labels = np.full(n,-1,dtype=int); strengths = np.zeros(n)
    if usable.any():
        labels[usable], strengths[usable] = hdbscan.approximate_predict(clusterer,embedding)
    assigned=labels>=0
    return dict(rows=n,assigned=int(assigned.sum()),coverage=float(assigned.mean()),
                median_assigned_strength=float(np.median(strengths[assigned])) if assigned.any() else None,
                clusters_reached=len(set(labels[assigned]))), labels


def samples(source, maximum, holdout, seed):
    source=Path(source)
    signature=hashlib.sha256(f'{source.resolve()}:{source.stat().st_size}:{source.stat().st_mtime_ns}:{maximum}:{holdout}:{seed}'.encode()).hexdigest()[:20]
    cache=Path(f'training-cache/advert-v2-sample-{signature}.parquet')
    if cache.exists():
        data=pd.read_parquet(cache)
    else:
        data=pd.read_json(source,lines=True)
        data['text']=data.text.fillna('').map(clean_description)
        # Preserve original chronological phases. Empty descriptions stay in denominators.
        data['priority']=data.row_id.map(lambda x:hashlib.sha256(f'{seed}:{x}'.encode()).hexdigest())
        seen=set(data.loc[data.phase=='train','text'])
        train=data[data.phase=='train'].sort_values('priority').drop_duplicates('text').head(maximum)
        parts=[train]
        for phase in ['validation','test']:
            d=data[(data.phase==phase)&~data.text.isin(seen)].sort_values('priority').drop_duplicates('text').head(holdout)
            parts.append(d)
        data=pd.concat(parts,ignore_index=True).drop(columns=['price'],errors='ignore')
        data.to_parquet(cache,index=False)
    return {p:data[data.phase==p].copy() for p in ['train','validation','test']}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('cohort');p.add_argument('--output',default='models/advert-clusters')
    p.add_argument('--max-train',type=int,default=20000);p.add_argument('--holdout',type=int,default=1000)
    p.add_argument('--finalize-known',action='store_true',help='Finalize accepted 92.1% configuration without further search')
    p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    if a.max_train < 100 or a.holdout < 1:
        p.error('--max-train must be at least 100 and --holdout must be positive')
    began=time.monotonic(); data=samples(a.cohort,a.max_train,a.holdout,a.seed)
    trials=[];best=None
    layouts=[(5,2,0.)] if a.finalize_known else [(5,2,0.),(10,5,0.),(30,5,.1),(60,10,.1)]
    for neighbors,dimensions,distance in layouts:
        print(f'UMAP neighbors={neighbors} dimensions={dimensions} min_dist={distance}',flush=True)
        bundle,report=fit(data['train'],seed=a.seed,n_neighbors=neighbors,n_components=dimensions,min_dist=distance,
                          min_cluster_size=80 if a.finalize_known else 15,min_samples=5 if a.finalize_known else 1)
        features,tfidf,usable_train=encode(bundle,data['train'])
        val_features,_,usable=encode(bundle,data['validation'])
        val_embedding=bundle['reducer'].transform(val_features[usable])
        embedding=bundle['reducer'].embedding
        for size in ([80] if a.finalize_known else [15,40,80]):
            for minimum in ([5] if a.finalize_known else [1,2,5]):
                clusterer=hdbscan.HDBSCAN(min_cluster_size=size,min_samples=minimum,prediction_data=True,core_dist_n_jobs=1).fit(embedding)
                labels=clusterer.labels_; k=len(set(labels)-{-1})
                largest=max((int((labels==l).sum()) for l in set(labels)-{-1}),default=0)/len(labels)
                scores,_=coverage(clusterer,val_embedding,usable,len(data['validation']))
                candidate=dict(umap=report['umap'],min_cluster_size=size,min_samples=minimum,clusters=k,largest_training_group_fraction=largest,
                               training_coverage=float(np.mean(labels>=0)),validation=scores)
                trials.append(candidate);print(json.dumps(candidate),flush=True)
                # Avoid achieving coverage by returning a single giant bucket.
                admissible=k>=5 and largest<=.70
                if admissible and (best is None or scores['coverage']>best[0]):
                    names=distinctive_terms(tfidf[usable_train],labels,bundle['vectorizer'].get_feature_names_out())
                    for label,summary in names.items():
                        summary['mean_membership_strength']=float(clusterer.probabilities_[labels==int(label)].mean())
                    chosen=dict(bundle);chosen.update(clusterer=clusterer,clusters=names,min_strength=0.)
                    chosen_report=dict(report);chosen_report.update(clusters=names,cluster_count=k,
                        training_noise_fraction=float(np.mean(labels<0)),hdbscan=dict(min_cluster_size=size,min_samples=minimum))
                    best=(scores['coverage'],chosen,chosen_report,candidate)
        if best and best[0]>=.97:
            break
    if best is None:raise RuntimeError('No noncollapsed partition; previous deployment left intact')
    _,bundle,report,chosen=best
    # The test phase is evaluated only AFTER selecting a frozen fit with validation.
    test_features,_,test_usable=encode(bundle,data['test'])
    test_embedding=bundle['reducer'].transform(test_features[test_usable])
    test,_=coverage(bundle['clusterer'],test_embedding,test_usable,len(data['test']))
    report.update(schema_version=2,source='Tom, Used Cars Dataset, Zenodo 10457828, CC BY 4.0',source_url='https://zenodo.org/records/10457828',
        seed=a.seed,holdouts={'validation':chosen['validation'],'test':test},trials=trials,
        selection='Highest validation native HDBSCAN assignment coverage; at least 5 groups, largest group <=70%; test used once after selection',
        coverage_target=.95,validation_target_met=chosen['validation']['coverage']>=.95,test_target_met=test['coverage']>=.95,
        assignment='Native approximate_predict labels; no nearest-cluster fallback, no forced noise assignments; strength threshold 0',
        rare_categories='All observed fit-sample levels included as field-qualified TF-IDF tokens, min_df=1; unseen levels at inference ignored',
        original_split='Original chronological 70/15/15; sampled train only; holdout descriptions seen in train excluded',
        price_features_used=False,fit_seconds=round(time.monotonic()-began,2),
        limitations=['Coverage is not semantic accuracy or dealer/private accuracy.','Uncalibrated membership strength; low-strength assignments retained.',
                     'UMAP can distort density. Hyperparameters selected on 1000 validation adverts, not guaranteed 95% future coverage.',
                     'All model categorical fields included; raw source IDs, contacts and coordinates excluded.'])
    if a.finalize_known:
        report['selection']='User accepted the previously measured 92.1% validation configuration and stopped further search; finalize UMAP 5 neighbors/2 dimensions/min_dist 0, HDBSCAN min_cluster_size 80/min_samples 5'
        report['accepted_validation_coverage']=.921
        report['original_requested_coverage_target']=.95
        report['coverage_target']=.921
        report['validation_target_met']=chosen['validation']['coverage']>=.921
        report['test_target_met']=test['coverage']>=.921
        log=Path('training-cache/advert-v2-tuning.log')
        report['initial_search_trials']=[json.loads(line) for line in log.read_text().splitlines() if line.startswith('{"umap":')] if log.exists() else []
        report['search_stopped_by_user']=True
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);path=out/'model.joblib'
    joblib.dump(bundle,path,compress=3)
    versions={key:version(key) for key in ['numpy','scipy','scikit-learn','umap-learn','hdbscan','numba','llvmlite','pynndescent','joblib']}
    report['versions']=versions
    manifest=dict(schema_version=2,status='ready',sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,versions=versions,
                  training_rows=report['training_rows'],cluster_count=report['cluster_count'])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # Remove obsolete parts from the previous bundle only after the new fit succeeds.
    for old in out.glob('model.joblib.*.part'):old.unlink()
    split_model(path,part_bytes=512*1024)
    print(json.dumps({'chosen':chosen,'test':test,'manifest':manifest},indent=2),flush=True)

if __name__=='__main__':main()
