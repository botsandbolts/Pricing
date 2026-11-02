"""Reuse frozen HDBSCAN; publish L2 and signed-preference evidence and verified 10/25-car experiences."""
from pathlib import Path
import argparse,hashlib,json,time,resource
import joblib,numpy as np,pandas as pd
from pricing.taste_ranking import load,faiss_l2,train_rank,adjusted_price_features,predict_new_profiles,records

def main():
 p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);args=p.parse_args();root=Path('.')
 meta,a=load(root);bundle=joblib.load(args.bundle)
 assert meta['rows']>100000 and bundle['source_rows']==meta['rows']
 d,_=faiss_l2(bundle['hdbscan']._raw_data,bundle['medoids']);a.pop('medoid_rmse',None);a.pop('medoid_l1',None);a['medoid_l2']=d
 np.savez_compressed('data/taste-garage/features.npz',**a)
 meta['sha256']['features.npz']=hashlib.sha256(Path('data/taste-garage/features.npz').read_bytes()).hexdigest();meta['profile_count']=10
 meta['neighbour_distance']='Exact FAISS IndexFlatL2; Euclidean square root of squared-L2 output in scaled features; not dollars'
 Path('data/taste-garage/manifest.json').write_text(json.dumps(meta,indent=2))
 from scripts.build_long_taste import build_long
 if not (Path('data/taste-garage/profiles-25.json').exists() and Path('training-cache/taste-incremental/frozen-profiles-25.joblib').exists()):build_long(root,bundle)
 from marketbridge.time_engine import TimeTransport,today
 current=adjusted_price_features(a['codes'],a['numeric'],meta['levels'],a['prices'],a['source_days'],today(),TimeTransport(root))
 reports={}
 for k in [10,25]:
    meta,a=load(root,k);m=a['memberships'];assert m.shape==(meta['rows'],k) and (m.sum(axis=1)>0).mean()>.5
    selected=bundle if k==10 else joblib.load('training-cache/taste-incremental/frozen-profiles-25.joblib')
    if k==25:
        from scripts.build_long_taste import l2_medoid
        geometry=selected['hdbscan']._raw_data;positions=[]
        for j in range(25):
            ids=np.flatnonzero(selected['hdbscan'].labels_==j);positions.append(l2_medoid(geometry[ids],ids))
        selected['medoids']=geometry[positions]
        joblib.dump(selected,'training-cache/taste-incremental/frozen-profiles-25.joblib',compress=3)
        cars=records(root,positions)
        for j,(pos,car) in enumerate(zip(positions,cars)):meta['profiles'][j].update(position=pos,prototype=car)
        a['validation'][positions]=False;a.pop('medoid_l1',None)
        a['medoid_l2']=faiss_l2(geometry,selected['medoids'])[0]
        file=Path('data/taste-garage/features-25.npz')
        np.savez_compressed(file,**{key:a[key] for key in ['memberships','clusters','noise','validation','medoid_l2']})
        meta['sha256']={file.name:hashlib.sha256(file.read_bytes()).hexdigest()}
        meta['representatives']='Real Euclidean L2 medoids; exact to 2000 members, deterministic 256-candidate approximation for larger groups'
        Path('data/taste-garage/profiles-25.json').write_text(json.dumps(meta,indent=2))
    nm,nd=predict_new_profiles(pd.DataFrame(records(root,[x['position'] for x in meta['profiles']])),selected)
    assert nm.shape==nd.shape==(k,k) and np.isfinite(nd).all()
    ratings=[1+i%4 for i in range(k)];started=time.time()
    result=train_rank(a['codes'],a['numeric'],meta['levels'],a['clusters'],ratings,[x['position'] for x in meta['profiles']],a['validation'],prices=a['prices'],adjusted_prices=current,memberships=m,medoid_distances=a['medoid_l2'])
    good=np.array(ratings)>=3;bad=~good;chosen=np.array(result['positions']);distances=a['medoid_l2'][chosen]
    assert np.all(distances[:,good].min(axis=1)<distances[:,bad].min(axis=1))
    assert all(x['signed']>0 for x in result['evidence'])
    assert result['score_range']>1e-7 and sum(result['feature_importance'].values())>0 and len(set(result['positions']))==10
    reports[str(k)]={'test_type':'Mechanical full-cohort test with artificial ratings; not user satisfaction','rows':meta['rows'],'profile_count':k,'ranking_seconds':time.time()-started,'model_result':result}
    from streamlit.testing.v1 import AppTest
    at=AppTest.from_string("from pathlib import Path\nfrom pricing.taste_garage_ui import render\nrender(Path('.'))").run(timeout=60)
    if k==25:
        next(r for r in at.radio if r.label=='Choose your test').set_value('The Long Drive · 25 cars, zero commitment').run(timeout=60)
    assert not at.exception
    radios=[r for r in at.radio if r.label=='Your reaction'];assert len(radios)==k
    for r,v in zip(radios,ratings):r.set_value(v)
    at.run(timeout=60);next(b for b in at.button if b.label=='Find my ten cars →').click().run(timeout=120)
    assert not at.exception and any('Your 10' in h.value for h in at.markdown)
 reports['process_peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
 Path('data/taste-garage/release-check.json').write_text(json.dumps(reports,indent=2));print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
