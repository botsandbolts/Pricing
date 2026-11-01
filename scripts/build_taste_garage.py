"""Full canonical asking-cohort UMAP/HDBSCAN, ten actual prototypes, portable assets."""
import argparse,gc,hashlib,json,re,sqlite3,time,zlib
from pathlib import Path
import numpy as np
import pandas as pd
from pricing.taste_ranking import CATEGORICAL,encode,clean,leave_one_out_geometry

def density_profiles(embedding):
    import hdbscan
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import pairwise_distances_argmin
    n=len(embedding);attempts=[];chosen=None
    for divisor in [100,250,500,1000]:
        minimum=max(30,n//divisor)
        model=hdbscan.HDBSCAN(min_cluster_size=minimum,min_samples=10,cluster_selection_method='leaf',metric='euclidean',core_dist_n_jobs=2).fit(embedding)
        labels=model.labels_;groups=np.unique(labels[labels>=0]);attempts.append({'min_cluster_size':minimum,'density_clusters':len(groups),'noise_fraction':float(np.mean(labels<0))})
        if len(groups)>=10:chosen=(labels,groups);break
    if chosen is None:raise ValueError('HDBSCAN did not support ten dense groups; no fabricated clusters are released')
    labels,groups=chosen;centres=np.array([embedding[labels==g].mean(axis=0) for g in groups])
    macro=np.arange(10) if len(groups)==10 else AgglomerativeClustering(n_clusters=10,linkage='ward').fit_predict(centres)
    mapped=np.full(n,-1,dtype=np.int16)
    for g,m in zip(groups,macro):mapped[labels==g]=m
    centres=np.array([embedding[mapped==g].mean(axis=0) for g in range(10)])
    noise=mapped<0
    if noise.any():mapped[noise]=pairwise_distances_argmin(embedding[noise],centres).astype(np.int16)
    prototypes=[]
    for g in range(10):
        members=np.flatnonzero((mapped==g)&~noise)
        prototypes.append(int(members[np.argmin(np.sum((embedding[members]-centres[g])**2,axis=1))]))
    return mapped,noise,prototypes,attempts,len(groups)

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
    geometry,_=leave_one_out_geometry(frame,np.array(prices,dtype=float),smoothing=20.)
    gc.collect()
    import umap
    print('Fitting full-cohort UMAP on',count,'cars',flush=True)
    embedding=umap.UMAP(n_components=5,n_neighbors=30,min_dist=.05,metric='euclidean',n_epochs=100,low_memory=True,random_state=42,n_jobs=1).fit_transform(geometry)
    del geometry;gc.collect();print('Fitting HDBSCAN',flush=True)
    clusters,noise,prototypes,attempts,density_count=density_profiles(embedding)
    valid=np.array([int(hashlib.sha256(('taste-validation:'+rid).encode()).hexdigest()[:8],16)%5==0 for rid in ids]);valid[prototypes]=False
    np.savez_compressed(folder/'features.npz',codes=codes,numeric=numeric,clusters=clusters,noise=noise,validation=valid,prices=np.array(prices,dtype=np.float32),source_days=np.array(source_days,dtype=np.int32))
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
    manifest={'version':1,'description_excerpt_characters':100,'category_encoding':'Smoothed leave-one-out log advertised price; alpha=20; own row excluded from group and global prior; dense standardized features; no one-hot or SVD before UMAP','rows':count,'source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),'source':'Original canonical Craigslist asking cohort; all eligible deduplicated partitions, transductive catalogue exploration','levels':levels,'profiles':profiles,'density_clusters':density_count,'noise_fraction':float(noise.mean()),'clustering_attempts':attempts,'elapsed_seconds':time.time()-started,'features':list(CATEGORICAL)+['model_year','log_mileage','mileage_missing'],'sha256':{name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in ['features.npz','cars.sqlite']},'limits':'Ten prototype ratings propagate as weak labels. HDBSCAN leaf groups are merged to ten macroprofiles when necessary; density noise is assigned to its nearest profile and flagged. No user features, duplicate rows or personal-satisfaction benchmark.'}
    if database.stat().st_size>90*1024*1024:raise ValueError('Serving catalogue exceeds 90 MiB release budget')
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in manifest.items() if k not in ['levels','profiles','sha256']},indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--root',default='.');a=p.parse_args();build(a.source,a.root)
