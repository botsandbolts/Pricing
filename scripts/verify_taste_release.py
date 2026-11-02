"""Reuse original HDBSCAN profiles; verify FAISS cosine and both rating UIs."""
from pathlib import Path
import argparse,hashlib,json,time,resource
import joblib,numpy as np,pandas as pd
from pricing.taste_ranking import load,faiss_cosine,train_rank,adjusted_price_features,predict_new_profiles,records

def main():
 p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);args=p.parse_args();root=Path('.')
 bundle=joblib.load(args.bundle)
 from marketbridge.time_engine import TimeTransport,today
 reports={};current=None
 for k in [10,25]:
    meta,a=load(root,k)
    selected=bundle if k==10 else joblib.load('training-cache/taste-incremental/frozen-profiles-25.joblib')
    assert 'umap' not in selected and selected['source_rows']==meta['rows'] and meta['rows']>100000
    if meta.get('encoding_kind')=='common-onehot-v1':
        assert selected['encoder'].get('kind')=='common-onehot-v1','Refusing to mix one-hot assets with a legacy frozen model'
    geometry=selected['hdbscan']._raw_data
    positions=[p['position'] for p in meta['profiles']]
    # Preserve every original representative and membership. Only replace distances.
    selected['medoids']=geometry[positions];selected['neighbour_metric']='cosine'
    d=faiss_cosine(geometry,selected['medoids'])[0]
    for key in ['medoid_l2','medoid_l1','medoid_rmse']:a.pop(key,None)
    a['medoid_cosine']=d
    folder=Path('data/taste-garage');file=folder/('features.npz' if k==10 else 'features-25.npz')
    payload=a if k==10 else {key:a[key] for key in ['memberships','clusters','noise','validation','medoid_cosine']}
    np.savez_compressed(file,**payload)
    meta.pop('embedding',None);meta['neighbour_metric']='cosine'
    meta['neighbour_distance']='Exact FAISS IndexFlatIP on L2-normalized scaled features; distance = 1 - cosine similarity'
    meta.setdefault('method_description','Existing HDBSCAN on scaled vehicle features, without dimensionality reduction. Original representatives and memberships preserved. FAISS cosine distance carries liked and disliked representative evidence; a separate XGBoost ranker orders eligible cars.')
    meta['sha256']={file.name:hashlib.sha256(file.read_bytes()).hexdigest()}
    if k==10:meta['sha256']['cars.sqlite']=hashlib.sha256((folder/'cars.sqlite').read_bytes()).hexdigest()
    (folder/('manifest.json' if k==10 else 'profiles-25.json')).write_text(json.dumps(meta,indent=2))
    joblib.dump(selected,Path(args.bundle) if k==10 else Path('training-cache/taste-incremental/frozen-profiles-25.joblib'),compress=3)
    nm,nd=predict_new_profiles(pd.DataFrame(records(root,positions)),selected)
    assert nm.shape==nd.shape==(k,k) and np.isfinite(nd).all()
    m=a['memberships'];assert m.shape==(meta['rows'],k) and (m.sum(axis=1)>0).mean()>.5
    if current is None:current=adjusted_price_features(a['codes'],a['numeric'],meta['levels'],a['prices'],a['source_days'],today(),TimeTransport(root))
    ratings=[1+i%4 for i in range(k)];started=time.time()
    result=train_rank(a['codes'],a['numeric'],meta['levels'],a['clusters'],ratings,positions,a['validation'],prices=a['prices'],adjusted_prices=current,memberships=m,medoid_distances=d)
    good=np.array(ratings)>=3;chosen=np.array(result['positions'])
    assert np.all(d[chosen][:,good].min(axis=1)<d[chosen][:,~good].min(axis=1))
    assert all(x['signed']>0 for x in result['evidence'])
    assert result['score_range']>1e-7 and len(set(result['positions']))==10
    from scripts.taste_ui_check import exercise_taste_ui
    reports[str(k)]={'test_type':'Mechanical full-cohort artificial-rating check, not customer satisfaction','rows':meta['rows'],'profile_count':k,'distance':'cosine','profiles_reused_for_distance_check':True,'encoding_kind':meta.get('encoding_kind','legacy-leave-one-out'),'ranking_seconds':time.time()-started,'model_result':result,'ui_check':exercise_taste_ui(root,k,ratings)}
 reports['process_peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
 Path('data/taste-garage/release-check.json').write_text(json.dumps(reports,indent=2));print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
