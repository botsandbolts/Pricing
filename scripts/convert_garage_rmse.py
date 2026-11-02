"""Recompute comparisons only; preserve fitted clusters, memberships and representatives."""
from pathlib import Path
import hashlib,json,sqlite3,zlib
from unittest.mock import patch
import numpy as np,pandas as pd
from pricing.taste_ranking import load,faiss_rmse,train_rank
from pricing.mixed_garage import fit_geometry
from marketbridge.time_engine import TimeTransport
from scripts.taste_ui_check import exercise_taste_ui

def main():
 root=Path('.');folder=root/'data/taste-garage-mixed'
 with sqlite3.connect(folder/'cars.sqlite') as conn:
    rows=conn.execute('SELECT position,payload FROM cars ORDER BY position').fetchall()
 assert [r[0] for r in rows]==list(range(len(rows)))
 frame=pd.DataFrame([json.loads(zlib.decompress(r[1])) for r in rows])
 frame['date']=frame['observed_date']
 geometry,state,adjusted=fit_geometry(frame,TimeTransport(root))
 assert np.isfinite(geometry).all() and len(geometry)==110832
 reports={}
 with patch('pricing.taste_ranking.active_catalogue_folder',return_value=folder),patch('pricing.taste_garage_ui.active_catalogue_folder',return_value=folder):
  for k in [10,25]:
    meta,a=load(root,k);positions=[p['position'] for p in meta['profiles']]
    d=faiss_rmse(geometry,geometry[positions])[0]
    for key in ['medoid_cosine','medoid_l2','medoid_l1']:a.pop(key,None)
    a['medoid_rmse']=d
    file=folder/('features.npz' if k==10 else 'features-25.npz')
    payload=a if k==10 else {key:a[key] for key in ['memberships','clusters','noise','validation','medoid_rmse']}
    np.savez_compressed(file,**payload)
    meta['sha256'][file.name]=hashlib.file_digest(open(file,'rb'),'sha256').hexdigest()
    meta['neighbour_metric']='rmse'
    meta['rating_transform']='5/12 - 1/rating; no normalization'
    meta['method_description']='Existing mixed UMAP/HDBSCAN clusters and real representatives preserved. FAISS RMSE compares original common-category indicators and robust range-scaled numeric features directly, without vector unit normalization. Inverse ratings with a fixed neutral offset shape preference evidence; XGBoost orders eligible cars.'
    (folder/('manifest.json' if k==10 else 'profiles-25.json')).write_text(json.dumps(meta,indent=2))
    ratings=[1+i%4 for i in range(k)]
    result=train_rank(a['codes'],a['numeric'],meta['levels'],a['clusters'],ratings,positions,a['validation'],prices=a['prices'],adjusted_prices=adjusted,memberships=a['memberships'],medoid_distances=d)
    chosen=np.array(result['positions']);good=np.array(ratings)>=3
    assert len(set(chosen))==10 and np.all(d[chosen][:,good].min(axis=1)<d[chosen][:,~good].min(axis=1))
    assert all(e['signed']>0 for e in result['evidence'])
    ui=exercise_taste_ui(root,k,ratings)
    reports[str(k)]={'passed':True,'rows':len(frame),'metric':'rmse','unit_normalization':False,'dimensions':geometry.shape[1],'rating_transform':meta['rating_transform'],'test_type':'Artificial-rating mechanical validation, not user satisfaction','model_result':result,'ui_check':ui}
 (folder/'release-check.json').write_text(json.dumps(reports,indent=2))
 print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
