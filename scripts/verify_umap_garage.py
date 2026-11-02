"""Validate complete serving arrays, real ranking and actual rating UI for both tests."""
from pathlib import Path
import json,time
import numpy as np
from pricing.taste_ranking import load,active_catalogue_folder,adjusted_price_features,train_rank
from scripts.taste_ui_check import exercise_taste_ui

def verify(root):
    from marketbridge.time_engine import TimeTransport,today
    root=Path(root);reports={};current=None
    for k in [10,25]:
        meta,a=load(root,k);assert meta['version']==4 and meta['rows']>100000
        assert meta['embedding']['min_dist']==0 and meta['embedding']['n_neighbors']==30 and meta['embedding']['n_components']==5
        if current is None:current=adjusted_price_features(a['codes'],a['numeric'],meta['levels'],a['prices'],a['source_days'],today(),TimeTransport(root))
        ratings=[1+i%4 for i in range(k)];started=time.time()
        r=train_rank(a['codes'],a['numeric'],meta['levels'],a['clusters'],ratings,[p['position'] for p in meta['profiles']],a['validation'],prices=a['prices'],adjusted_prices=current,memberships=a['memberships'],medoid_distances=a['medoid_cosine'])
        positions=np.array(r['positions']);d=a['medoid_cosine'][positions];good=np.array(ratings)>=3
        assert len(set(positions))==10 and np.all(d[:,good].min(axis=1)<d[:,~good].min(axis=1))
        assert all(e['signed']>0 for e in r['evidence']) and r['score_range']>1e-7
        ui=exercise_taste_ui(root,k,ratings)
        reports[str(k)]={'test_type':'Full-cohort mechanical validation with artificial ratings; no independent satisfaction claim','rows':meta['rows'],'embedding':meta['embedding'],'ranking_seconds':time.time()-started,'model_result':r,'ui_check':ui}
    (active_catalogue_folder(root)/'release-check.json').write_text(json.dumps(reports,indent=2));print(json.dumps(reports,indent=2))
    return reports

if __name__=='__main__':verify(Path('.'))
