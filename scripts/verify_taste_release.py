"""Reuse the verified HDBSCAN fit; export exact FAISS RMSE and test real ranking."""
from pathlib import Path
import argparse,hashlib,json,time,resource
import joblib,numpy as np
from pricing.taste_ranking import load,faiss_rmse,train_rank,adjusted_price_features,predict_new_profiles,records

def main():
 p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);args=p.parse_args();root=Path('.')
 meta,a=load(root);bundle=joblib.load(args.bundle)
 assert meta['version']==2 and meta['rows']>100000 and bundle['source_rows']==meta['rows']
 m=a['memberships'];assert m.shape==(meta['rows'],10) and (m.sum(axis=1)>0).mean()>.5 and (m.sum(axis=0)>0).all()
 d,order=faiss_rmse(bundle['hdbscan']._raw_data,bundle['medoids']);a['medoid_rmse']=d
 np.savez_compressed('data/taste-garage/features.npz',**a)
 meta['sha256']['features.npz']=hashlib.sha256(Path('data/taste-garage/features.npz').read_bytes()).hexdigest()
 meta['neighbour_distance']='Exact FAISS IndexFlatL2; RMSE = sqrt(squared L2 / 14 scaled features); not dollars'
 Path('data/taste-garage/manifest.json').write_text(json.dumps(meta,indent=2))
 import pandas as pd
 new=pd.DataFrame(records(root,[p['position'] for p in meta['profiles']]))
 nm,nd=predict_new_profiles(new,bundle);assert nm.shape==nd.shape==(10,10) and np.isfinite(nd).all()
 from marketbridge.time_engine import TimeTransport,today
 current=adjusted_price_features(a['codes'],a['numeric'],meta['levels'],a['prices'],a['source_days'],today(),TimeTransport(root))
 ratings=[1,2,3,4,1,2,3,4,2,3];started=time.time()
 result=train_rank(a['codes'],a['numeric'],meta['levels'],a['clusters'],ratings,[p['position'] for p in meta['profiles']],a['validation'],prices=a['prices'],adjusted_prices=current,memberships=m)
 assert result['score_range']>1e-7 and sum(result['feature_importance'].values())>0
 assert len(set(result['positions']))==10
 report={'test_type':'Mechanical full-cohort smoke test with artificial prototype ratings, not user satisfaction','rows':meta['rows'],'ranking_seconds':time.time()-started,'process_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'membership_nonzero_fraction':float((m.sum(axis=1)>0).mean()),'model_result':result}
 Path('data/taste-garage/release-check.json').write_text(json.dumps(report,indent=2))
 from streamlit.testing.v1 import AppTest
 at=AppTest.from_string("from pathlib import Path\nfrom pricing.taste_garage_ui import render\nrender(Path('.'))").run(timeout=60)
 assert not at.exception
 radios=[r for r in at.radio if r.label=='Your reaction'];assert len(radios)==10
 for r,v in zip(radios,ratings):r.set_value(v)
 at.run(timeout=60);next(b for b in at.button if b.label=='Find my ten cars →').click().run(timeout=120)
 assert not at.exception and any('Your ten' in h.value for h in at.markdown)
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
