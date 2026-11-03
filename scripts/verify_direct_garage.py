"""Verify retained-only mixed catalogue, stored price feature and both real UIs."""
from pathlib import Path
from unittest.mock import patch
import gzip,hashlib,json,time
import joblib,numpy as np,pandas as pd
from pricing.taste_ranking import load,records,train_rank,adjusted_price_features,predict_new_profiles,transform_new_geometry
from pricing.mixed_garage import KIND,comparison_vectors


def verify(root):
    from marketbridge.time_engine import TimeTransport,today
    from scripts.taste_ui_check import exercise_taste_ui
    root=Path(root);folder=root/'data/taste-garage-mixed';reports={};current=None
    with gzip.open(folder/'row-ids.json.gz','rt') as f:ids=json.load(f)
    selection=json.loads((folder/'selection.json').read_text())
    assert len(ids)==len(set(ids))==selection['retained_rows']==110832
    assert hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()==selection['retained_id_sha256']
    with patch('pricing.taste_ranking.active_catalogue_folder',return_value=folder), patch('pricing.taste_garage_ui.active_catalogue_folder',return_value=folder):
      for k in [10,25]:
        meta,a=load(root,k);assert meta['version']==5 and meta['encoding_kind']==KIND and meta['rows']==110832
        assert meta['density_method']=='direct-weighted-euclidean' and 'embedding' not in meta
        bundle=joblib.load(root/f'training-cache/lemon-mixed/frozen-profiles-{k}.joblib')
        assert bundle['encoder']['kind']==KIND and 'umap' not in bundle and bundle['hdbscan'].metric=='euclidean'
        assert 'log_mileage' in bundle['encoder']['numeric_names'] and 'log_predicted_text_price' in bundle['encoder']['numeric_names']
        prototypes=[p['position'] for p in meta['profiles']]
        # Probe contains the stored scalar price, not MiniLM text or embedding inputs.
        probe=pd.DataFrame(records(root,prototypes[:3])).drop(columns=['description'])
        geometry=transform_new_geometry(probe,bundle['encoder'])
        np.testing.assert_allclose(geometry*bundle['cluster_weights'],bundle['hdbscan']._raw_data[prototypes[:3]],atol=2e-5)
        m,d=predict_new_profiles(probe,bundle)
        assert m.shape==d.shape==(3,k) and np.isfinite(m).all() and np.isfinite(d).all()
        if current is None:current=adjusted_price_features(a['codes'],a['numeric'],meta['levels'],a['prices'],a['source_days'],today(),TimeTransport(root))
        ratings=[1+i%4 for i in range(k)];started=time.time()
        r=train_rank(a['codes'],a['numeric'],meta['levels'],a['clusters'],ratings,prototypes,a['validation'],prices=a['prices'],adjusted_prices=current,memberships=a['memberships'],medoid_distances=a['medoid_cosine'])
        chosen=np.array(r['positions']);dist=a['medoid_cosine'][chosen];good=np.array(ratings)>=3
        assert len(set(chosen))==10 and not set(chosen).intersection(prototypes)
        assert np.all(dist[:,good].min(axis=1)<dist[:,~good].min(axis=1))
        assert all(e['signed']>0 for e in r['evidence']) and r['score_range']>1e-7
        assert np.all(np.diff([e['liked_distance'] for e in r['evidence']])<=1e-7)
        ui=exercise_taste_ui(root,k,ratings)
        reports[str(k)]={'passed':True,'test_type':'Retained-cohort mechanical validation with artificial ratings; not measured satisfaction',
                         'rows':meta['rows'],'encoding_kind':KIND,'stored_text_price':True,'raw_text_embeddings':False,
                         'price_date':selection['price_date'],'clustering':meta['clustering'],'umap':False,
                         'ranking_seconds':time.time()-started,'model_result':r,'ui_check':ui}
        print('VERIFIED',k,'profiles',flush=True)
    (folder/'release-check.json').write_text(json.dumps(reports,indent=2))
    (root/'data/taste-garage-release.json').write_text(json.dumps({'directory':'taste-garage-mixed','encoding_kind':KIND,'verified':True,'rows':110832,'price_date':selection['price_date']},indent=2))
    print(json.dumps(reports,indent=2));return reports


if __name__=='__main__':verify(Path('.'))
