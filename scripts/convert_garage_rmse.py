"""Recompute comparisons only; preserve fitted clusters, memberships and representatives."""
from pathlib import Path
import hashlib,json,sqlite3,zlib
import joblib
from unittest.mock import patch
import numpy as np,pandas as pd
from pricing.taste_ranking import load,faiss_cosine,train_rank
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
 offline=root/'training-cache/cosine-unscaled';offline.mkdir(parents=True,exist_ok=True)
 joblib.dump({'encoder':state,'neighbour_metric':'cosine-unscaled'},offline/'faiss-preprocessor.joblib',compress=3)
 reports={}
 with patch('pricing.taste_ranking.active_catalogue_folder',return_value=folder),patch('pricing.taste_garage_ui.active_catalogue_folder',return_value=folder):
  for k in [10,25]:
    meta,a=load(root,k);positions=[p['position'] for p in meta['profiles']]
    d=faiss_cosine(geometry,geometry[positions])[0]
    for key in ['medoid_rmse','medoid_l2','medoid_l1']:a.pop(key,None)
    a['medoid_cosine']=d
    file=folder/('features.npz' if k==10 else 'features-25.npz')
    payload=a if k==10 else {key:a[key] for key in ['memberships','clusters','noise','validation','medoid_cosine']}
    np.savez_compressed(file,**payload)
    meta['sha256'][file.name]=hashlib.file_digest(open(file,'rb'),'sha256').hexdigest()
    meta['neighbour_metric']='cosine-unscaled'
    meta['faiss_scaling']='No additional feature-column scaling; cosine on existing encoded features'
    meta['sha256'].pop('faiss-scaler.json',None)
    meta['rating_transform']='Direct ratings 1,2,3,4; signed evidence rating - 2.5; no inversion or scaler'
    meta['method_description']='Existing mixed UMAP/HDBSCAN clusters and real representatives preserved. FAISS cosine compares the existing encoded features without extra column StandardScaler or block normalization. L2 normalization is used only to calculate cosine. Direct ratings keep 4 highest and 1 lowest. Signed evidence uses rating minus 2.5, without reciprocal transform or session scaler. Preference evidence shapes the shortlist; XGBoost orders eligible cars.'
    (folder/('manifest.json' if k==10 else 'profiles-25.json')).write_text(json.dumps(meta,indent=2))
    ratings=[1+i%4 for i in range(k)]
    result=train_rank(a['codes'],a['numeric'],meta['levels'],a['clusters'],ratings,positions,a['validation'],prices=a['prices'],adjusted_prices=adjusted,memberships=a['memberships'],medoid_distances=d)
    chosen=np.array(result['positions']);good=np.array(ratings)>=3
    assert len(set(chosen))==10 and np.all(d[chosen][:,good].min(axis=1)<d[chosen][:,~good].min(axis=1))
    assert all(e['signed']>0 for e in result['evidence'])
    assert np.all(np.diff([e['liked_distance'] for e in result['evidence']])<=1e-7)
    checked_meta,checked_arrays=load(root,k)
    assert checked_meta['neighbour_metric']=='cosine-unscaled' and 'medoid_cosine' in checked_arrays
    assert 'medoid_rmse' not in checked_arrays
    np.testing.assert_allclose(checked_arrays['medoid_cosine'],d)
    ui=exercise_taste_ui(root,k,ratings)
    reports[str(k)]={'passed':True,'rows':len(frame),'metric':'cosine','cosine_l2_normalization':True,'column_standard_scaler':False,'dimensions':geometry.shape[1],'rating_transform':meta['rating_transform'],'test_type':'Artificial-rating mechanical validation, not user satisfaction','model_result':result,'ui_check':ui}
 (folder/'release-check.json').write_text(json.dumps(reports,indent=2))
 print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
