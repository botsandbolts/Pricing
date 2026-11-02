"""Reuse frozen HDBSCAN; publish L1 evidence and verified 10/25-car experiences."""
from pathlib import Path
import argparse,hashlib,json,time,resource
import joblib,numpy as np,pandas as pd
from pricing.taste_ranking import load,faiss_l1,train_rank,adjusted_price_features,predict_new_profiles,records

def main():
 p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);args=p.parse_args();root=Path('.')
 meta,a=load(root);bundle=joblib.load(args.bundle)
 assert meta['rows']>100000 and bundle['source_rows']==meta['rows']
 d,_=faiss_l1(bundle['hdbscan']._raw_data,bundle['medoids']);a.pop('medoid_rmse',None);a['medoid_l1']=d
 np.savez_compressed('data/taste-garage/features.npz',**a)
 meta['sha256']['features.npz']=hashlib.sha256(Path('data/taste-garage/features.npz').read_bytes()).hexdigest();meta['profile_count']=10
 meta['neighbour_distance']='Exact FAISS METRIC_L1; sum of absolute differences in scaled features; not dollars'
 Path('data/taste-garage/manifest.json').write_text(json.dumps(meta,indent=2))
 from scripts.build_long_taste import build_long
 build_long(root,bundle)
 from marketbridge.time_engine import TimeTransport,today
 current=adjusted_price_features(a['codes'],a['numeric'],meta['levels'],a['prices'],a['source_days'],today(),TimeTransport(root))
 reports={}
 for k in [10,25]:
    meta,a=load(root,k);m=a['memberships'];assert m.shape==(meta['rows'],k) and (m.sum(axis=1)>0).mean()>.5
    selected=bundle if k==10 else joblib.load('training-cache/taste-incremental/frozen-profiles-25.joblib')
    nm,nd=predict_new_profiles(pd.DataFrame(records(root,[x['position'] for x in meta['profiles']])),selected)
    assert nm.shape==nd.shape==(k,k) and np.isfinite(nd).all()
    ratings=[1+i%4 for i in range(k)];started=time.time()
    result=train_rank(a['codes'],a['numeric'],meta['levels'],a['clusters'],ratings,[x['position'] for x in meta['profiles']],a['validation'],prices=a['prices'],adjusted_prices=current,memberships=m)
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
    assert not at.exception and any('Your ten' in h.value for h in at.markdown)
 reports['process_peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
 Path('data/taste-garage/release-check.json').write_text(json.dumps(reports,indent=2));print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
