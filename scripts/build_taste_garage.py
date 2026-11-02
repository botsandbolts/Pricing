"""Full canonical asking-cohort common-category one-hot HDBSCAN and real medoids."""
import argparse,gc,hashlib,json,re,sqlite3,time,zlib
from pathlib import Path
import numpy as np
import pandas as pd
from pricing.taste_ranking import CATEGORICAL,encode,clean,one_hot_geometry

def medoid_position(points,positions):
    """Exact medoid for small groups; bounded candidate medoid for large groups."""
    from scipy.spatial.distance import cdist
    positions=np.asarray(positions);n=len(positions)
    if n<=2000:candidates=np.arange(n)
    else:
        rng=np.random.default_rng(42)
        near=int(np.argmin(np.sum((points-points.mean(axis=0))**2,axis=1)))
        candidates=np.unique(np.r_[near,rng.choice(n,255,replace=False)])
    costs=np.zeros(len(candidates))
    for start in range(0,n,2048):costs+=cdist(points[candidates],points[start:start+2048]).sum(axis=1)
    return int(positions[candidates[np.argmin(costs)]])

def density_profiles(geometry,return_model=False,memory=None,core_jobs=2):
    import hdbscan
    from sklearn.cluster import AgglomerativeClustering
    n=len(geometry);attempts=[];best=None
    # Tune density support, never create a partition with KMeans or UMAP.
    for divisor in [50,100,200,400,800]:
        minimum=max(30,n//divisor)
        model=hdbscan.HDBSCAN(min_cluster_size=minimum,min_samples=10,cluster_selection_method='eom',metric='euclidean',core_dist_n_jobs=core_jobs,algorithm='boruvka_kdtree',memory=memory or __import__('joblib').Memory(None),prediction_data=True).fit(geometry)
        labels=model.labels_;groups=np.unique(labels[labels>=0])
        attempts.append({'min_cluster_size':minimum,'density_clusters':len(groups),'noise_fraction':float(np.mean(labels<0))})
        if len(groups)>=10 and (best is None or len(groups)<len(best[2])):best=(model,labels,groups)
        if len(groups)==10:break
        if best is not None:break
    if best is None:raise ValueError('HDBSCAN did not support ten dense groups; no fabricated profiles released')
    model,labels,groups=best
    centres=np.array([geometry[labels==g].mean(axis=0) for g in groups])
    macro=np.arange(10) if len(groups)==10 else AgglomerativeClustering(n_clusters=10,linkage='ward').fit_predict(centres)
    memberships=np.zeros((n,10),dtype=np.float32)
    raw=np.nan_to_num(hdbscan.all_points_membership_vectors(model),nan=0.,posinf=0.,neginf=0.)
    for j,m in enumerate(macro):memberships[:,m]+=raw[:,j]
    del raw
    memberships=np.maximum(memberships,0);memberships/=np.maximum(1,memberships.sum(axis=1,keepdims=True))
    mapped=np.argmax(memberships,axis=1).astype(np.int16);noise=labels<0
    prototypes=[]
    for g in range(10):
        native=groups[macro==g];members=np.flatnonzero(np.isin(labels,native))
        prototypes.append(medoid_position(geometry[members],members))
    result=(mapped,noise,prototypes,attempts,len(groups),memberships)
    return result+(model,macro) if return_model else result

def build(source,root):
    started=time.time();root=Path(root);folder=root/'data/taste-garage';folder.mkdir(parents=True,exist_ok=True)
    database=folder/'cars.sqlite';database.unlink(missing_ok=True);conn=sqlite3.connect(database)
    conn.execute('CREATE TABLE cars(position INTEGER PRIMARY KEY,payload BLOB NOT NULL)')
    parts=[];ids=[];prices=[];source_days=[];count=0
    for chunk in pd.read_json(source,lines=True,chunksize=5000):
        if chunk.row_id.duplicated().any():raise ValueError('Duplicate canonical IDs')
        prices.extend(chunk.price.astype(float).tolist());source_days.extend((pd.to_datetime(chunk.date,utc=True).astype('int64')//86400000000000).tolist())
        parts.append(chunk.reindex(columns=list(CATEGORICAL)+['year','odometer']).copy());ids.extend(chunk.row_id.astype(str).tolist())
        for r in chunk.to_dict('records'):
            car={k:r.get(k) for k in list(CATEGORICAL)+['year','odometer']};car={k:None if pd.isna(v) else v for k,v in car.items()}
            text=str(r.get('text') or '')
            text=re.sub(r'(?i)\b[A-HJ-NPR-Z0-9]{17}\b','[VIN removed]',text)
            text=re.sub(r'(?<!\d)(?:\+?1[ .-]*)?(?:\(\d{3}\)|\d{3})[ .-]*\d{3}[ .-]*\d{4}','[phone removed]',text)
            text=re.sub(r'\S+@\S+','[email removed]',text)
            car.update(id=str(r['row_id']),price=float(r['price']),observed_date=str(r['date']),description=text[:100])
            conn.execute('INSERT INTO cars VALUES(?,?)',(count,zlib.compress(json.dumps({k:v for k,v in car.items() if v is not None},allow_nan=False,separators=(',',':')).encode(),6)));count+=1
        conn.commit();print('Prepared',count,'original asking cars',flush=True)
    conn.execute('VACUUM');conn.close()
    print('Compact catalogue MiB:',database.stat().st_size/1024**2,flush=True)
    if database.stat().st_size>90*1024*1024:raise ValueError('Serving catalogue exceeds 90 MiB release budget before clustering')
    if len(set(ids))!=count:raise ValueError('Canonical vehicle duplication detected')
    frame=pd.concat(parts,ignore_index=True);del parts;gc.collect()
    codes,numeric,levels=encode(frame)
    geometry,_,encoder=one_hot_geometry(frame,return_state=True)
    gc.collect()
    print('Fitting scaled-feature HDBSCAN and soft memberships on',count,'cars',flush=True)
    clusters,noise,prototypes,attempts,density_count,memberships,model,macro=density_profiles(geometry,return_model=True)
    import joblib
    offline=root/'training-cache/taste-incremental';offline.mkdir(parents=True,exist_ok=True)
    joblib.dump({'encoder':encoder,'hdbscan':model,'macro':macro,'medoids':geometry[prototypes],'source_rows':count},offline/'frozen-profiles.joblib',compress=3)
    del geometry;gc.collect()
    valid=np.array([int(hashlib.sha256(('taste-validation:'+rid).encode()).hexdigest()[:8],16)%5==0 for rid in ids]);valid[prototypes]=False
    np.savez_compressed(folder/'features.npz',codes=codes,numeric=numeric,memberships=memberships,clusters=clusters,noise=noise,validation=valid,prices=np.array(prices,dtype=np.float32),source_days=np.array(source_days,dtype=np.int32))
    profiles=[]
    from pricing.taste_ranking import records
    # Reading the ten prototypes is bounded; no serving-time full catalogue decode.
    cars=records(root,prototypes)
    for i,pos in enumerate(prototypes):
        subset=frame.loc[clusters==i]
        bodies=subset.type.map(clean).value_counts();makes=subset.manufacturer.map(clean).value_counts()
        body=next((v for v in bodies.index if v!='__missing__'),'Mixed cars')
        make=next((v for v in makes.index if v!='__missing__'),'Mixed makes')
        profiles.append({'position':pos,'cluster':i,'name':str(body).title()+' · '+str(make).title(),'cars':int(len(subset)),'prototype':cars[i]})
    manifest={'version':4,'encoding_kind':'common-onehot-v1','clustering_target_free':True,'category_min_count':100,'category_min_fraction':.001,'category_count_threshold':encoder['threshold'],'description_excerpt_characters':100,'category_encoding':'Common-category one-hot indicators; rare/unseen pooled; threshold max(100, ceil(0.001 * rows)); only numeric features standardized and clipped; no price targets, UMAP or SVD','rows':count,'source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),'source':'Original canonical Craigslist asking cohort; all eligible deduplicated partitions, transductive catalogue exploration','levels':levels,'profiles':profiles,'density_clusters':density_count,'noise_fraction':float(noise.mean()),'clustering_attempts':attempts,'elapsed_seconds':time.time()-started,'features':encoder['feature_names'],'sha256':{name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in ['features.npz','cars.sqlite']},'representatives':'Exact sum-distance medoid for groups up to 2000; deterministic 256-candidate medoid evaluated against every member for larger groups; always an actual non-noise advert','limits':'Ten medoid ratings propagate via a soft membership matrix, retaining unassigned mass. HDBSCAN EOM groups are merged to ten macroprofiles when necessary; noise retains soft memberships and unassigned mass, and is flagged. No user features, duplicate rows or personal-satisfaction benchmark.'}
    if database.stat().st_size>90*1024*1024:raise ValueError('Serving catalogue exceeds 90 MiB release budget')
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in manifest.items() if k not in ['levels','profiles','sha256']},indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--root',default='.');a=p.parse_args();build(a.source,a.root)

