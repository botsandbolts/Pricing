"""Session-specific vehicle-only LambdaMART; inherited labels are not user truth."""
import hashlib,json,sqlite3,zlib
from pathlib import Path
import numpy as np
import pandas as pd

CATEGORICAL=('manufacturer','model','type','fuel','transmission','drive','title_status','condition','cylinders','paint_color','state')
NUMERIC=('model_year','log_mileage','mileage_missing')

def clean(value):
    if value is None or pd.isna(value):return '__missing__'
    text=' '.join(str(value).strip().lower().split())
    return text if text not in ('','unknown','nan','none') else '__missing__'

def encode(frame):
    levels={};codes=[]
    for field in CATEGORICAL:
        values=frame[field].map(clean) if field in frame else pd.Series('__missing__',index=frame.index)
        names=sorted(set(values));levels[field]=names;lookup={v:i for i,v in enumerate(names)}
        codes.append(values.map(lookup).to_numpy(dtype=np.int32))
    year=pd.to_numeric(frame.year,errors='coerce').to_numpy(float)
    miles=pd.to_numeric(frame.odometer,errors='coerce').to_numpy(float)
    valid=np.isfinite(miles)&(miles>=0)&(miles<=2e6)
    year=np.where(np.isfinite(year),year,np.nanmedian(year));median=np.median(miles[valid]) if valid.any() else 95000.
    numeric=np.column_stack([year,np.log1p(np.where(valid,miles,median)),~valid]).astype(np.float32)
    return np.column_stack(codes),numeric,levels

def features(codes,numeric,levels):
    # Base vehicle attributes use native categorical partitioning; price features are appended by train_rank. No user or cluster predictors.
    data={field:pd.Categorical.from_codes(codes[:,j],categories=levels[field]) for j,field in enumerate(CATEGORICAL)}
    data.update({name:numeric[:,j] for j,name in enumerate(NUMERIC)})
    return pd.DataFrame(data)

def ndcg10(labels,scores,linear=False):
    labels=np.asarray(labels,dtype=float);scores=np.asarray(scores,dtype=float)
    if len(labels)!=len(scores) or not len(labels):raise ValueError('Nonempty aligned labels/scores required')
    k=min(10,len(labels));order=np.argsort(-scores,kind='stable')[:k];ideal=np.sort(labels)[::-1][:k]
    discount=np.log2(np.arange(k)+2);den=np.sum((ideal if linear else np.expm1(np.log(2)*ideal))/discount)
    return float(np.sum((labels[order] if linear else np.expm1(np.log(2)*labels[order]))/discount)/den) if den>0 else None

def train_rank(codes,numeric,levels,clusters,ratings,prototype_positions,validation_mask,rounds=96,prices=None,adjusted_prices=None,memberships=None,medoid_distances=None):
    from xgboost import XGBRanker
    ratings=np.asarray(ratings,dtype=int)
    k=len(ratings)
    if k not in (10,25) or ratings.shape!=(k,) or not np.isin(ratings,[1,2,3,4]).all():raise ValueError('Rate every representative from 1 to 4')
    if not np.any(ratings>=3):raise ValueError('Rate at least one representative Good or Very good')
    if len(set(ratings))<2:raise ValueError('At least two different ratings are needed to learn an ordering')
    clusters=np.asarray(clusters,dtype=int)
    if clusters.shape!=(len(codes),) or clusters.min()<0 or clusters.max()>=k:raise ValueError('Invalid cluster map')
    target=ratings[clusters]-1
    soft=memberships is not None
    if soft:
        m=np.asarray(memberships,dtype=float)
        if m.shape!=(len(codes),k) or not np.isfinite(m).all() or (m<0).any() or (m.sum(axis=1)>1.0001).any():raise ValueError('Invalid soft memberships')
        evidence=preference_evidence(m,ratings,medoid_distances)
        relevance=(evidence['signed']+1)*1.5
        target=np.rint(relevance*10).astype(np.int32)
    else:relevance=target.astype(float)
    valid=np.asarray(validation_mask,dtype=bool).copy();valid[np.asarray(prototype_positions,dtype=int)]=False
    train=~valid
    if train.sum()<30 or valid.sum()<10:raise ValueError('Insufficient distinct vehicles for training and validation')
    x=features(codes,numeric,levels)
    if prices is not None:
        original=np.asarray(prices,dtype=float)
        if original.shape!=(len(x),) or not np.isfinite(original).all() or (original<=0).any():raise ValueError('Invalid advertised price features')
        x['log_advertised_price']=np.log(original).astype(np.float32)
        current=np.asarray(adjusted_prices if adjusted_prices is not None else np.full(len(x),np.nan),dtype=float)
        if current.shape!=(len(x),):raise ValueError('Invalid adjusted price features')
        supported=np.isfinite(current)&(current>0)
        x['log_date_adjusted_price']=np.where(supported,np.log(np.maximum(current,1)),np.nan).astype(np.float32)
        x['price_adjustment_missing']=(~supported).astype(np.float32)
    params=dict(objective='rank:ndcg',eval_metric='ndcg@10',tree_method='hist',enable_categorical=True,max_cat_to_onehot=1,max_cat_threshold=32,max_bin=64,max_depth=3,min_child_weight=.001,reg_lambda=1.,reg_alpha=0.,learning_rate=.07,lambdarank_pair_method='topk',lambdarank_num_pair_per_sample=10,n_jobs=2,random_state=42)
    if soft:params['ndcg_exp_gain']=False
    probe=XGBRanker(**params,n_estimators=rounds,early_stopping_rounds=20)
    probe.fit(x.loc[train],target[train],qid=np.zeros(train.sum(),dtype=np.uint32),eval_set=[(x.loc[valid],target[valid])],eval_qid=[np.zeros(valid.sum(),dtype=np.uint32)],verbose=False)
    iterations=min(probe.get_booster().num_boosted_rounds(),max(16,int(probe.best_iteration)+1))
    held=probe.predict(x.loc[valid],iteration_range=(0,iterations));metric=ndcg10(relevance[valid],held,linear=soft)
    rng=np.random.default_rng(42);random_metric=ndcg10(relevance[valid],rng.random(valid.sum()),linear=soft)
    del probe
    # Final fit uses each original advert exactly once. No Cartesian expansion.
    model=XGBRanker(**params,n_estimators=iterations)
    model.fit(x,target,qid=np.zeros(len(x),dtype=np.uint32),verbose=False)
    scores=model.predict(x)
    score_range=float(np.ptp(scores))
    if score_range<1e-7:raise ValueError('Ranking model learned no ordering; a constant shortlist is not released')
    eligible=evidence['eligible'].copy() if soft else np.ones(len(scores),dtype=bool)
    eligible[np.asarray(prototype_positions,dtype=int)]=False
    candidates=np.flatnonzero(eligible)
    if not len(candidates):raise ValueError('No cars pass your positive and negative preference checks; revise your ratings')
    # Soft preference breaks tree-leaf ties; source row order never decides the shortlist.
    order=candidates[np.lexsort((-relevance[candidates],-scores[candidates]))[:10]]
    explanations=[]
    if soft:
        for i in order:
            item={key:float(evidence[key][i]) for key in ['signed','positive','negative']}
            if medoid_distances is not None:
                good=np.flatnonzero(ratings>=3);bad=np.flatnonzero(ratings<=2);d=np.asarray(medoid_distances)[i]
                item['liked_profile']=int(good[np.argmin(d[good])]);item['liked_distance']=float(d[item['liked_profile']])
                if len(bad):item['disliked_profile']=int(bad[np.argmin(d[bad])]);item['disliked_distance']=float(d[item['disliked_profile']])
            explanations.append(item)
    return {'eligible_cars':int(len(candidates)),'evidence':explanations,'preference_method':'signed memberships plus medoid-distance evidence; nearest-dislike exclusion' if medoid_distances is not None else 'signed memberships','score_range':score_range,'positions':order.tolist(),'scores':scores[order].tolist(),'ndcg10':metric,'random_ndcg10':random_metric,'training_rows':int(train.sum()),'validation_rows':int(valid.sum()),'final_rows':len(x),'rounds':iterations,'direct_judgments':k,'soft_memberships':soft,'gain':'linear soft relevance; 0.1 ordinal training resolution' if soft else 'exponential ordinal','feature_importance':dict(zip(x.columns,model.feature_importances_.astype(float).tolist()))}

def active_catalogue_folder(root):
    root=Path(root);candidate=root/'data/taste-garage-umap'
    if all((candidate/f).exists() for f in ['manifest.json','features.npz','profiles-25.json','features-25.npz']):return candidate
    return root/'data/taste-garage'

def load(root,profile_count=10,catalogue_path=None):
    folder=Path(catalogue_path) if catalogue_path is not None else active_catalogue_folder(root);file=folder/'manifest.json' 
    if not file.exists():return None
    manifest=json.loads(file.read_text())
    for name,digest in manifest['sha256'].items():
        p=folder/name
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('Recommendation asset integrity mismatch')
    with np.load(folder/'features.npz',allow_pickle=False) as data:arrays={key:data[key] for key in data.files}
    if profile_count==25:
        variant=folder/'profiles-25.json'
        if not variant.exists():return None
        extra=json.loads(variant.read_text())
        if extra['source_sha256']!=manifest['source_sha256']:raise ValueError('Profile source mismatch')
        for name,digest in extra['sha256'].items():
            if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=digest:raise ValueError('Long-test asset integrity mismatch')
        with np.load(folder/'features-25.npz',allow_pickle=False) as data:arrays.update({key:data[key] for key in data.files})
        arrays.pop('medoid_rmse',None);manifest.update(extra)
    elif profile_count!=10:raise ValueError('Choose the 10- or 25-car test')
    k=len(manifest['profiles'])
    n=len(arrays['clusters'])
    if n!=manifest['rows'] or arrays['codes'].shape!=(n,len(CATEGORICAL)) or arrays['numeric'].shape!=(n,len(NUMERIC)) or arrays['prices'].shape!=(n,) or arrays['source_days'].shape!=(n,):raise ValueError('Invalid feature matrix')
    if manifest.get('version',1)>=2:
        m=arrays.get('memberships')
        if m is None or m.shape!=(n,k) or not np.isfinite(m).all() or (m<0).any() or (m.sum(axis=1)>1.0001).any():raise ValueError('Invalid soft membership asset')
    return manifest,arrays

def records(root,positions):
    path=(Path(root)/'data/taste-garage/cars.sqlite').resolve()
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as conn:
        output=[]
        for pos in positions:
            found=conn.execute('SELECT payload FROM cars WHERE position=?',(int(pos),)).fetchone()
            if found is None:raise ValueError('Reference car missing')
            output.append(json.loads(zlib.decompress(found[0])))
    return output


def adjusted_price_features(codes,numeric,levels,prices,source_days,when,transport):
    """Bounded vectorized price transport, preserving unsupported values as NaN."""
    from marketbridge.time_engine import today
    frame=features(codes,numeric,levels)
    for field in CATEGORICAL:frame[field]=frame[field].astype(object)
    frame['year']=numeric[:,0];frame['odometer']=np.expm1(numeric[:,1])
    sources=pd.DatetimeIndex(np.asarray(source_days).astype('datetime64[D]'))
    target=pd.Timestamp(when)
    old_age=np.maximum(0,sources.year+(sources.dayofyear-1)/365.25-numeric[:,0])
    new_age=target.year+(target.dayofyear-1)/365.25-numeric[:,0]
    valid=(old_age<=30)&(new_age>=0)&(new_age<=30)&(numeric[:,2]==0)
    out=np.full(len(frame),np.nan,dtype=np.float32);ids=np.flatnonzero(valid)
    if not len(ids):return out
    factors=transport.curves['ask'].ratio(frame.iloc[ids],old_age[ids],new_age[ids],from_miles=frame.odometer.to_numpy()[ids],to_miles=frame.odometer.to_numpy()[ids])
    current,_=transport.clock.value(when,as_of=today())
    months=sources.strftime('%Y-%m');lookup={m:transport.clock.reference(m+'-01')[0] for m in set(months[ids])}
    values=np.asarray(prices)[ids]*factors*current/np.array([lookup[months[i]] for i in ids])
    good=np.isfinite(values)&(values>0)&(values<=1e7);out[ids[good]]=values[good]
    return out


def leave_one_out_geometry(frame,prices,smoothing=20.,return_state=False):
    """Dense category price profiles, excluding each row from group AND prior."""
    from sklearn.preprocessing import StandardScaler
    target=np.log(np.asarray(prices,dtype=float));n=len(target)
    if n<2 or not np.isfinite(target).all():raise ValueError('Finite positive advertised prices required for leave-one-out encoding')
    if smoothing<0:raise ValueError('Smoothing must be nonnegative')
    prior=(target.sum()-target)/(n-1)
    columns=[];category_counts=[];category_stats={}
    for field in CATEGORICAL:
        values=frame[field].map(clean) if field in frame else pd.Series('__missing__',index=frame.index)
        category_counts.append(values.nunique())
        stats=pd.DataFrame({'category':values.to_numpy(),'target':target}).groupby('category').target.agg(['sum','count'])
        category_stats[field]={str(k):(float(r['sum']),int(r['count'])) for k,r in stats.iterrows()}
        groups=pd.DataFrame({'category':values.to_numpy(),'target':target}).groupby('category',sort=False).target
        sums=groups.transform('sum').to_numpy();counts=groups.transform('count').to_numpy()
        denominator=counts-1+smoothing
        encoded=np.divide(sums-target+smoothing*prior,denominator,out=prior.copy(),where=denominator>0)
        columns.append(encoded)
    _,numeric,_=encode(frame)
    encoded_matrix=np.column_stack(columns)
    # A constant or all-unique category otherwise has tiny leave-out variation
    # that StandardScaler could amplify into a spurious price-order signal.
    informative=(encoded_matrix.std(axis=0)>=max(.001,.001*float(target.std()))) & (np.asarray(category_counts)>1) & (np.asarray(category_counts)<n)
    stable=encoded_matrix.copy();stable[:,~informative]=0.
    matrix=np.column_stack([stable,numeric])
    scaler=StandardScaler().fit(matrix)
    scaled=np.clip(scaler.transform(matrix),-6,6).astype(np.float32)
    if return_state:
        state={'stats':category_stats,'prior':float(target.mean()),'smoothing':smoothing,'informative':informative,'scaler':scaler,'year_median':float(np.median(numeric[:,0])),'log_mileage_median':float(np.median(numeric[:,1]))}
        return scaled,encoded_matrix,state
    return scaled,encoded_matrix



def transform_new_geometry(frame,state):
    """Frozen inference encoding: reference cohort statistics, never new targets."""
    cols=[]
    for field in CATEGORICAL:
        values=frame[field].map(clean) if field in frame else pd.Series('__missing__',index=frame.index)
        lookup=state['stats'][field];a=state['smoothing'];prior=state['prior']
        cols.append(np.array([(lookup[v][0]+a*prior)/(lookup[v][1]+a) if v in lookup else prior for v in values]))
    encoded=np.column_stack(cols);encoded[:,~np.asarray(state['informative'])]=0
    year=pd.to_numeric(frame.get('year',pd.Series(np.nan,index=frame.index)),errors='coerce').to_numpy(float)
    miles=pd.to_numeric(frame.get('odometer',pd.Series(np.nan,index=frame.index)),errors='coerce').to_numpy(float)
    valid=np.isfinite(miles)&(miles>=0)&(miles<=2e6)
    numeric=np.column_stack([np.where(np.isfinite(year),year,state['year_median']),np.where(valid,np.log1p(np.maximum(miles,0)),state['log_mileage_median']),~valid])
    return np.clip(state['scaler'].transform(np.column_stack([encoded,numeric])),-6,6).astype(np.float32)

def predict_new_profiles(frame,bundle):
    """No refit: frozen encoding/scaling, HDBSCAN affinities, medoid distances."""
    import hdbscan
    geometry=transform_new_geometry(frame,bundle['encoder'])
    cluster_geometry=bundle['umap'].transform(geometry) if 'umap' in bundle else geometry
    if bundle.get('prediction_method')=='flat':
        from hdbscan.flat import membership_vector_flat
        raw=membership_vector_flat(bundle['hdbscan'],cluster_geometry)
    else:raw=hdbscan.membership_vector(bundle['hdbscan'],cluster_geometry)
    raw=np.nan_to_num(raw,nan=0.,posinf=0.,neginf=0.)
    m=np.zeros((len(frame),len(bundle['medoids'])),dtype=np.float32)
    for j,k in enumerate(bundle['macro']):m[:,k]+=raw[:,j]
    m=np.maximum(m,0);m/=np.maximum(1,m.sum(axis=1,keepdims=True))
    return m,(faiss_cosine if bundle.get('neighbour_metric')=='cosine' else faiss_l2)(geometry,bundle['medoids'])[0]


def faiss_l2(queries,references):
    """Exact FAISS L2 search; Euclidean distance in scaled feature units, column-aligned distances."""
    import faiss
    q=np.ascontiguousarray(queries,dtype=np.float32);r=np.ascontiguousarray(references,dtype=np.float32)
    if q.ndim!=2 or r.ndim!=2 or q.shape[1]!=r.shape[1] or not len(r) or not q.shape[1] or not np.isfinite(q).all() or not np.isfinite(r).all():raise ValueError('Finite aligned feature matrices required')
    index=faiss.IndexFlatL2(r.shape[1]);index.add(r)
    distances=np.empty((len(q),len(r)),dtype=np.float32);orders=np.empty((len(q),len(r)),dtype=np.int32)
    for start in range(0,len(q),2048):
        squared,order=index.search(q[start:start+2048],len(r))
        l2=np.sqrt(np.maximum(squared,0))
        distances[start:start+len(order)]=np.take_along_axis(l2,np.argsort(order,axis=1),axis=1)
        orders[start:start+len(order)]=order
    return distances,orders


def faiss_cosine(queries,references):
    """Exact cosine distance (1 - normalized inner product); zero vectors are neutral."""
    import faiss
    q=np.array(queries,dtype=np.float32,order='C',copy=True)
    r=np.array(references,dtype=np.float32,order='C',copy=True)
    if q.ndim!=2 or r.ndim!=2 or q.shape[1]!=r.shape[1] or not len(r) or not q.shape[1] or not np.isfinite(q).all() or not np.isfinite(r).all():raise ValueError('Finite aligned feature matrices required')
    faiss.normalize_L2(q);faiss.normalize_L2(r)
    index=faiss.IndexFlatIP(r.shape[1]);index.add(r)
    distances=np.empty((len(q),len(r)),dtype=np.float32)
    orders=np.empty((len(q),len(r)),dtype=np.int32)
    for start in range(0,len(q),2048):
        similarity,order=index.search(q[start:start+2048],len(r))
        distance=1-np.clip(similarity,-1,1)
        distances[start:start+len(order)]=np.take_along_axis(distance,np.argsort(order,axis=1),axis=1)
        orders[start:start+len(order)]=order
    return distances,orders


def preference_evidence(memberships,ratings,distances=None):
    """Signed evidence: negative reactions lower preference; no approval from noise."""
    m=np.asarray(memberships,dtype=float);r=np.asarray(ratings,dtype=int)
    signed=(2*r-5)/3.;positive=m@np.maximum(signed,0);negative=m@np.maximum(-signed,0)
    allowed=np.ones(len(m),dtype=bool)
    if distances is not None:
        d=np.asarray(distances,dtype=float)
        if d.shape!=m.shape or not np.isfinite(d).all() or (d<0).any():raise ValueError('Invalid medoid distances')
        affinity=1/(1+d*d);confidence=affinity.max(axis=1);weights=affinity/np.maximum(affinity.sum(axis=1,keepdims=True),1e-12)
        positive=.75*positive+.25*(weights@np.maximum(signed,0))*confidence
        negative=.75*negative+.25*(weights@np.maximum(-signed,0))*confidence
        good=r>=3;bad=r<=2
        if not good.any():allowed[:]=False
        elif bad.any():allowed=d[:,good].min(axis=1)<d[:,bad].min(axis=1)
    score=positive-negative
    return {'signed':score,'positive':positive,'negative':negative,'eligible':allowed&(score>1e-8)}
