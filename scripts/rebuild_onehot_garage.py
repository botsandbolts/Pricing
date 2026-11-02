"""Rebuild both profile tests from the verified existing canonical catalogue."""
import argparse,gc,hashlib,json,sqlite3,time,zlib
from pathlib import Path
import joblib,numpy as np,pandas as pd
from pricing.taste_ranking import load,one_hot_geometry,records
from scripts.build_taste_garage import density_profiles
from scripts.build_long_taste import build_long


def rebuild(root,min_count=100,min_fraction=.001,core_jobs=4):
    root=Path(root);folder=root/'data/taste-garage';meta,a=load(root)
    started=time.time();db=folder/'cars.sqlite'
    with sqlite3.connect(db) as conn:
        rows=[json.loads(zlib.decompress(r[0])) for r in conn.execute('SELECT payload FROM cars ORDER BY position')]
    assert len(rows)==meta['rows']==len(a['codes'])
    frame=pd.DataFrame(rows);ids=frame.id.astype(str)
    assert ids.nunique()==meta['rows']
    geometry,_,encoder=one_hot_geometry(frame,min_count=min_count,min_fraction=min_fraction,return_state=True)
    del _
    print(json.dumps({'stage':'encoded','rows':len(frame),'dimensions':geometry.shape[1],
                      'common_categories':{k:len(v) for k,v in encoder['categories'].items()},
                      'rare_threshold':encoder['threshold']}),flush=True)
    offline=root/'training-cache/taste-incremental';offline.mkdir(parents=True,exist_ok=True)
    # Cache the expensive neighbour graph/MST while density support is tuned.
    memory=joblib.Memory(root/'training-cache/onehot-mst',verbose=0)
    groups,noise,positions,attempts,native,memberships,model,macro=density_profiles(geometry,True,memory,core_jobs)
    assert np.isfinite(geometry).all() and len(set(positions))==10 and not noise[positions].any()
    bundle={'encoder':encoder,'hdbscan':model,'macro':macro,'medoids':geometry[positions],
            'source_rows':len(frame),'neighbour_metric':'cosine'}
    joblib.dump(bundle,offline/'frozen-profiles.joblib',compress=3)
    validation=np.array([int(hashlib.sha256(('taste-validation:'+rid).encode()).hexdigest()[:8],16)%5==0 for rid in ids]);validation[positions]=False
    # Ranking retains its native categorical inputs; only clustering geometry changes.
    a={k:a[k] for k in ['codes','numeric','prices','source_days']}
    a.update(memberships=memberships,clusters=groups,noise=noise,validation=validation)
    np.savez_compressed(folder/'features.npz',**a)
    cars=records(root,positions)
    meta.update(version=4,encoding_kind='common-onehot-v1',
        category_encoding='One-hot common categories; rare and unseen values pooled per field. Threshold max(100, ceil(0.001 * reference rows)). Indicators unscaled; numeric features standardized and clipped to six standard deviations.',
        category_min_count=min_count,category_min_fraction=min_fraction,category_count_threshold=encoder['threshold'],
        geometry_dimensions=geometry.shape[1],geometry_features=encoder['feature_names'],features=encoder['feature_names'],
        clustering_target_free=True,neighbour_metric='cosine',
        method_description='HDBSCAN fits common-category one-hot indicators and standardized numeric vehicle features, without asking-price target encoding or dimensionality reduction. Rare and unseen categories share an Other bucket. Real medoids, memberships and both tests are rebuilt; FAISS cosine drives preference comparisons, and XGBoost ranks eligible matches.',
        profiles=[{'position':int(p),'cluster':i,'name':f'Profile {i+1}','cars':int((groups==i).sum()),'prototype':cars[i]} for i,p in enumerate(positions)],
        density_clusters=native,noise_fraction=float(noise.mean()),clustering_attempts=attempts,
        elapsed_seconds=time.time()-started,
        sha256={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in ['features.npz','cars.sqlite']})
    (folder/'manifest.json').write_text(json.dumps(meta,indent=2))
    del frame,rows,a,geometry,memberships;gc.collect()
    build_long(root,bundle)
    long=json.loads((folder/'profiles-25.json').read_text())
    for key in ['encoding_kind','category_encoding','category_min_count','category_min_fraction','category_count_threshold','geometry_dimensions','clustering_target_free','method_description']:
        long[key]=meta[key]
    (folder/'profiles-25.json').write_text(json.dumps(long,indent=2))
    print(json.dumps({'stage':'built','seconds':time.time()-started,'rows':meta['rows'],'encoding':meta['encoding_kind']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--min-count',type=int,default=100);p.add_argument('--min-fraction',type=float,default=.001);p.add_argument('--core-jobs',type=int,default=4);args=p.parse_args()
    rebuild(args.root,args.min_count,args.min_fraction,args.core_jobs)
