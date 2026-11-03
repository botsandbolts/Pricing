"""Lemon-only stored text-price + common one-hot + mileage/Nov-2026 prices → UMAP → HDBSCAN."""
import argparse,gc,gzip,hashlib,json,re,sqlite3,time,zlib
from pathlib import Path
import joblib,numpy as np,pandas as pd
from pricing.mixed_garage import KIND,AS_OF,mixed_distance,fit_geometry,comparison_vectors,normalized
from pricing.taste_ranking import CATEGORICAL,encode,faiss_cosine
from scripts.build_long_taste import select_long_model


UMAP_PARAMETERS={'n_neighbors':30,'min_dist':0.,'n_components':5,'n_epochs':200,'random_state':42,'low_memory':True,'n_jobs':1}


def make_embedding(state,index=None):
    import umap
    options=dict(UMAP_PARAMETERS,metric='cosine',verbose=True)
    if index is not None:options['precomputed_knn']=(*index.neighbor_graph,index)
    return umap.UMAP(**options)


def digest(path):return hashlib.file_digest(open(path,'rb'),'sha256').hexdigest()


def actual_cosine_medoid(vectors,positions):
    # Unit-vector cosine sum medoid: maximize dot with the sum of unit vectors.
    unit=normalized(vectors);centre=unit.sum(axis=0,dtype=np.float64)
    return int(positions[np.argmax(unit@centre)])


def build(root,prepared,jobs=4):
    import umap,hdbscan
    from pynndescent import NNDescent
    from hdbscan.flat import all_points_membership_vectors_flat
    from marketbridge.time_engine import TimeTransport
    started=time.time();root=Path(root);prepared=Path(prepared)
    selection=json.loads((prepared/'selection.json').read_text())
    frame=pd.read_json(prepared/'cohort.jsonl',lines=True);frame['row_id']=frame.row_id.astype(str)
    ids=json.loads((prepared/'row-ids.json').read_text());assert frame.row_id.tolist()==ids and len(ids)==selection['retained_rows']
    geometry,state,prices_nov=fit_geometry(frame,TimeTransport(root))
    comparison=geometry
    print(json.dumps({'stage':'mixed geometry','rows':len(frame),'dimensions':geometry.shape[1],
                      'common_category_threshold':state['categorical']['threshold'],'price_date':AS_OF.isoformat(),
                      'price_scenarios_available':int(np.isfinite(prices_nov).sum()),'metric':state['metric']}),flush=True)
    offline=root/'training-cache/lemon-mixed';offline.mkdir(parents=True,exist_ok=True)
    kwargs={'n_cat':state['onehot_dimensions'],'n_num':state['numeric_dimensions']}
    graph_file=offline/'knn.joblib'
    signature=hashlib.sha256((hashlib.sha256(memoryview(geometry)).hexdigest()+digest(root/'pricing/mixed_garage.py')+json.dumps(kwargs,sort_keys=True)+'cosine-30-20-10').encode()).hexdigest()
    if graph_file.exists() and (offline/'knn-signature.txt').read_text()==signature:
        index=joblib.load(graph_file);print('Restored mixed-distance nearest-neighbour graph',flush=True)
    else:
        print('Building mixed-distance approximate neighbour graph over every retained car',flush=True)
        index=NNDescent(geometry,n_neighbors=30,metric='cosine',n_jobs=jobs,
                        random_state=42,low_memory=True,n_trees=20,n_iters=10,verbose=True)
        joblib.dump(index,graph_file,compress=0);(offline/'knn-signature.txt').write_text(signature)
    indices,distances=index.neighbor_graph
    assert indices.shape==distances.shape==(len(frame),30) and np.isfinite(distances).all()
    reducer=make_embedding(state,index)
    print('Embedding mixed graph in five UMAP dimensions',flush=True)
    embedded=reducer.fit_transform(geometry).astype(np.float64)
    density=hdbscan.HDBSCAN(min_cluster_size=500,min_samples=10,cluster_selection_method='leaf',
                           metric='euclidean',prediction_data=True,core_dist_n_jobs=jobs).fit(embedded)
    print('Fitted low-dimensional HDBSCAN',flush=True)
    folder=root/'data/taste-garage-mixed';folder.mkdir(parents=True,exist_ok=True)
    db=folder/'cars.sqlite';db.unlink(missing_ok=True)
    conn=sqlite3.connect(db);conn.execute('CREATE TABLE cars(position INTEGER PRIMARY KEY,payload BLOB NOT NULL)')
    cars=[]
    for i,row in enumerate(frame.to_dict('records')):
        car={k:row.get(k) for k in list(CATEGORICAL)+['year','odometer','predicted_text_price']};car={k:None if pd.isna(v) else v for k,v in car.items()}
        text=str(row.get('text') or '')
        text=re.sub(r'(?i)\b[A-HJ-NPR-Z0-9]{17}\b','[VIN removed]',text)
        text=re.sub(r'(?<!\d)(?:\+?1[ .-]*)?(?:\(\d{3}\)|\d{3})[ .-]*\d{3}[ .-]*\d{4}','[phone removed]',text)
        text=re.sub(r'\S+@\S+','[email removed]',text)
        car.update(id=str(row['row_id']),price=float(row['price']),observed_date=str(row['date']),description=text[:100])
        cars.append(car);conn.execute('INSERT INTO cars VALUES(?,?)',(i,zlib.compress(json.dumps(car,allow_nan=False,separators=(',',':')).encode(),6)))
    conn.commit();conn.execute('VACUUM');conn.close()
    codes,numeric,levels=encode(frame)
    days=(pd.to_datetime(frame.date,utc=True).astype('int64')//86400000000000).to_numpy(dtype=np.int32)
    validation=np.array([int(hashlib.sha256(('taste-validation:'+rid).encode()).hexdigest()[:8],16)%5==0 for rid in ids])
    with gzip.open(folder/'row-ids.json.gz','wt') as f:json.dump(ids,f)
    selection.update(price_date=AS_OF.isoformat(),price_scenario_assumption='Recorded mileage held constant; age/market-index transport, not a live transaction quote',
                     supported_price_scenarios=int(np.isfinite(prices_nov).sum()),numeric_features=state['numeric_names'],
                     rarity_threshold=state['categorical']['threshold'],geometry_dimensions=geometry.shape[1],metric=state['metric'],weights=state['weights'],text_price_use='Existing cross-fitted scalar dollar feature; no embeddings or fresh MiniLM inference')
    (folder/'selection.json').write_text(json.dumps(selection,indent=2))
    for k in [10,25]:
        model=select_long_model(density,count=k)
        memberships=np.nan_to_num(all_points_membership_vectors_flat(model),nan=0.,posinf=0.,neginf=0.).astype(np.float32)
        memberships=np.maximum(memberships,0);memberships/=np.maximum(1,memberships.sum(axis=1,keepdims=True))
        assert memberships.shape==(len(frame),k) and (memberships.sum(axis=0)>0).all()
        positions=[]
        for group in range(k):
            members=np.flatnonzero(model.labels_==group)
            positions.append(actual_cosine_medoid(comparison[members],members))
        assert len(set(positions))==k and not (model.labels_[positions]<0).any()
        d,_=faiss_cosine(comparison,comparison[positions]);valid=validation.copy();valid[positions]=False
        arrays=dict(memberships=memberships,clusters=np.argmax(memberships,axis=1).astype(np.int16),noise=model.labels_<0,validation=valid,medoid_cosine=d)
        if k==10:arrays.update(codes=codes,numeric=numeric,prices=frame.price.to_numpy(dtype=np.float32),source_days=days,asking_prices_nov2026=prices_nov.astype(np.float32))
        file=folder/('features.npz' if k==10 else 'features-25.npz');np.savez_compressed(file,**arrays)
        meta=dict(version=5,encoding_kind=KIND,profile_count=k,rows=len(frame),source_sha256=selection['source_sha256'],
                  selection=selection,category_count_threshold=state['categorical']['threshold'],neighbour_metric='cosine-unscaled',density_method='umap-cosine-hdbscan',
                  levels=levels,profiles=[dict(position=int(p),cluster=i,name=f'Profile {i+1}',cars=int((model.labels_==i).sum()),prototype=cars[p]) for i,p in enumerate(positions)],
                  density_clusters=k,noise_fraction=float(np.mean(model.labels_<0)),
                  embedding={'n_neighbors':30,'min_dist':0.,'n_components':5,'n_epochs':200,'random_state':42,'metric':'cosine'},
                  method_description='Only Lemon Squeezer retained training cars. Common-category one-hot indicators, mileage/year, log asking-price scenarios for November 2026, and the existing cross-fitted text-price scalar enter a cosine-distance UMAP. HDBSCAN clusters its 5D output. FAISS cosine compares original encoded features without added column scaling; a separate XGBoost ranker orders eligible cars.',
                  representatives='Exact sum-cosine-distance medoid among actual non-noise members, in balanced original feature blocks',
                  sha256={file.name:digest(file)})
        if k==10:meta['sha256'].update({name:digest(folder/name) for name in ['cars.sqlite','row-ids.json.gz','selection.json']})
        (folder/('manifest.json' if k==10 else 'profiles-25.json')).write_text(json.dumps(meta,indent=2))
        bundle={'encoder':state,'umap':reducer,'hdbscan':model,'macro':np.arange(k),'medoids':comparison[positions],
                'source_rows':len(frame),'source_sha256':selection['source_sha256'],'neighbour_metric':'cosine-unscaled','prediction_method':'flat'}
        joblib.dump(bundle,offline/f'frozen-profiles-{k}.joblib',compress=3)
        print('Exported',k,'profiles',flush=True)
        del arrays,memberships,model,bundle;gc.collect()
    print(json.dumps({'stage':'built','rows':len(frame),'seconds':time.time()-started}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--prepared',default='training-cache/lemon-mixed');p.add_argument('--jobs',type=int,default=4);a=p.parse_args();build(a.root,a.prepared,a.jobs)
