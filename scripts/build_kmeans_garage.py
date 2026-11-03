"""StandardScaler then MiniBatchKMeans; saved scaler reused for inference."""
import argparse,gc,gzip,hashlib,json,re,sqlite3,time,zlib
from pathlib import Path
import joblib,numpy as np,pandas as pd
from pricing.mixed_garage import KIND,AS_OF,mixed_distance,fit_geometry,comparison_vectors,normalized,euclidean_hdbscan_geometry
from pricing.taste_ranking import CATEGORICAL,encode,faiss_cosine
from scripts.build_long_taste import select_long_model



def digest(path):return hashlib.file_digest(open(path,'rb'),'sha256').hexdigest()


def actual_cosine_medoid(vectors,positions):
    # Unit-vector cosine sum medoid: maximize dot with the sum of unit vectors.
    unit=normalized(vectors);centre=unit.sum(axis=0,dtype=np.float64)
    return int(positions[np.argmax(unit@centre)])


def build(root,prepared,jobs=4):
    import hdbscan
    from hdbscan.flat import all_points_membership_vectors_flat
    from marketbridge.time_engine import TimeTransport
    started=time.time();root=Path(root);prepared=Path(prepared)
    selection=json.loads((prepared/'selection.json').read_text())
    frame=pd.read_json(prepared/'cohort.jsonl',lines=True);frame['row_id']=frame.row_id.astype(str)
    ids=json.loads((prepared/'row-ids.json').read_text());assert frame.row_id.tolist()==ids and len(ids)==selection['retained_rows']
    geometry,state,prices_nov=fit_geometry(frame,TimeTransport(root))
    comparison=geometry
    from sklearn.preprocessing import StandardScaler
    cluster_scaler=StandardScaler().fit(geometry)
    clustered=np.ascontiguousarray(cluster_scaler.transform(geometry),dtype=np.float64)
    cluster_weights=np.ones(geometry.shape[1])
    np.testing.assert_allclose(clustered.mean(axis=0),0,atol=2e-4)
    state['metric']='Euclidean distance after sklearn StandardScaler fitted per training feature column'
    print('Direct feature dimensions:',clustered.shape,flush=True)
    print(json.dumps({'stage':'mixed geometry','rows':len(frame),'dimensions':geometry.shape[1],
                      'common_category_threshold':state['categorical']['threshold'],'price_date':AS_OF.isoformat(),
                      'price_scenarios_available':int(np.isfinite(prices_nov).sum()),'metric':state['metric']}),flush=True)
    offline=root/'training-cache/lemon-mixed';offline.mkdir(parents=True,exist_ok=True)
    density=None
    folder=root/'data/taste-garage-kmeans';folder.mkdir(parents=True,exist_ok=True)
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
        from sklearn.cluster import MiniBatchKMeans
        from scipy.special import softmax
        model=MiniBatchKMeans(n_clusters=k,batch_size=2048,n_init=5,max_iter=150,random_state=42,reassignment_ratio=.01).fit(clustered)
        distances=model.transform(clustered)
        temperature=max(float(np.sqrt(np.median(np.partition(distances**2,1,axis=1)[:,1]-np.min(distances**2,axis=1)))),1e-6)
        logits=-distances**2/(2*temperature**2)
        nearest=np.argpartition(distances,2,axis=1)[:,:3]
        mask=np.zeros_like(logits,dtype=bool);np.put_along_axis(mask,nearest,True,axis=1)
        memberships=softmax(np.where(mask,logits,-np.inf),axis=1).astype(np.float32)
        print('MiniBatchKMeans fitted',k,'temperature',temperature,flush=True)
        positions=[]
        for group in range(k):
            members=np.flatnonzero(model.labels_==group)
            if not len(members):raise ValueError('Empty KMeans group')
            positions.append(int(members[np.argmin(distances[members,group])]))
        assert len(set(positions))==k and not (model.labels_[positions]<0).any()
        d,_=faiss_cosine(comparison,comparison[positions]);valid=validation.copy();valid[positions]=False
        arrays=dict(memberships=memberships,clusters=np.argmax(memberships,axis=1).astype(np.int16),noise=model.labels_<0,validation=valid,medoid_cosine=d)
        if k==10:arrays.update(codes=codes,numeric=numeric,prices=frame.price.to_numpy(dtype=np.float32),source_days=days,asking_prices_nov2026=prices_nov.astype(np.float32))
        file=folder/('features.npz' if k==10 else 'features-25.npz');np.savez_compressed(file,**arrays)
        meta=dict(version=5,encoding_kind=KIND,profile_count=k,rows=len(frame),source_sha256=selection['source_sha256'],
                  selection=selection,category_count_threshold=state['categorical']['threshold'],neighbour_metric='cosine-unscaled',density_method='minibatch-kmeans-soft',
                  levels=levels,profiles=[dict(position=int(p),cluster=i,name=f'Profile {i+1}',cars=int((model.labels_==i).sum()),prototype=cars[p]) for i,p in enumerate(positions)],
                  density_clusters=k,noise_fraction=float(np.mean(model.labels_<0)),
                  clustering={'scaler':'sklearn StandardScaler; fitted on full reference training matrix','method':'MiniBatchKMeans','temperature':temperature,'metric':'euclidean','categorical_weight':.5,'numeric_weight':.5,'dimensions':geometry.shape[1],'umap':False,'min_samples':10},
                  method_description='Temporary MiniBatchKMeans with softmax of negative squared center distances. These are normalized distance affinities, not density memberships or calibrated probabilities. Only Lemon Squeezer retained training cars. MiniBatchKMeans fits column-standardized Euclidean features; affinities retain the three closest centers and use a temperature fitted from median squared-distance gaps. No UMAP or dimensionality reduction. Features include common one-hot categories, mileage/year, log November-2026 ask prices and stored text-price scalar. FAISS cosine compares original encoded vectors without extra column scaling; direct user ratings shape the shortlist. Display order is descending cosine distance.',
                  representatives='Real assigned car nearest its MiniBatchKMeans center',
                  sha256={file.name:digest(file)})
        if k==10:meta['sha256'].update({name:digest(folder/name) for name in ['cars.sqlite','row-ids.json.gz','selection.json']})
        (folder/('manifest.json' if k==10 else 'profiles-25.json')).write_text(json.dumps(meta,indent=2))
        bundle={'encoder':state,'cluster_weights':cluster_weights,'cluster_scaler':cluster_scaler,'kmeans':model,'temperature':temperature,'macro':np.arange(k),'medoids':comparison[positions],
                'source_rows':len(frame),'source_sha256':selection['source_sha256'],'neighbour_metric':'cosine-unscaled','prediction_method':'flat'}
        joblib.dump(bundle,offline/f'frozen-profiles-{k}.joblib',compress=3)
        print('Exported',k,'profiles',flush=True)
        del arrays,memberships,model,bundle;gc.collect()
    print(json.dumps({'stage':'built','rows':len(frame),'seconds':time.time()-started}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--prepared',default='training-cache/lemon-mixed');p.add_argument('--jobs',type=int,default=4);a=p.parse_args();build(a.root,a.prepared,a.jobs)
