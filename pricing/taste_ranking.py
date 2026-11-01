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
    # Native categorical partitioning in XGBoost, no one-hot/user/cluster/price features.
    data={field:pd.Categorical.from_codes(codes[:,j],categories=levels[field]) for j,field in enumerate(CATEGORICAL)}
    data.update({name:numeric[:,j] for j,name in enumerate(NUMERIC)})
    return pd.DataFrame(data)

def ndcg10(labels,scores):
    labels=np.asarray(labels,dtype=int);scores=np.asarray(scores,dtype=float)
    if len(labels)!=len(scores) or not len(labels):raise ValueError('Nonempty aligned labels/scores required')
    k=min(10,len(labels));order=np.argsort(-scores,kind='stable')[:k];ideal=np.sort(labels)[::-1][:k]
    discount=np.log2(np.arange(k)+2);den=np.sum(np.expm1(np.log(2)*ideal)/discount)
    return float(np.sum(np.expm1(np.log(2)*labels[order])/discount)/den) if den>0 else None

def train_rank(codes,numeric,levels,clusters,ratings,prototype_positions,validation_mask,rounds=96,prices=None,adjusted_prices=None):
    from xgboost import XGBRanker
    ratings=np.asarray(ratings,dtype=int)
    if ratings.shape!=(10,) or not np.isin(ratings,[1,2,3,4]).all():raise ValueError('Rate all ten prototypes from 1 to 4')
    if len(set(ratings))<2:raise ValueError('At least two different ratings are needed to learn an ordering')
    clusters=np.asarray(clusters,dtype=int)
    if clusters.shape!=(len(codes),) or clusters.min()<0 or clusters.max()>9:raise ValueError('Invalid cluster map')
    target=ratings[clusters]-1
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
    params=dict(objective='rank:ndcg',eval_metric='ndcg@10',tree_method='hist',enable_categorical=True,max_cat_to_onehot=1,max_cat_threshold=32,max_bin=64,max_depth=3,min_child_weight=20,reg_lambda=20.,reg_alpha=1.,learning_rate=.07,lambdarank_pair_method='topk',lambdarank_num_pair_per_sample=10,n_jobs=2,random_state=42)
    probe=XGBRanker(**params,n_estimators=rounds,early_stopping_rounds=12)
    probe.fit(x.loc[train],target[train],qid=np.zeros(train.sum(),dtype=np.uint32),eval_set=[(x.loc[valid],target[valid])],eval_qid=[np.zeros(valid.sum(),dtype=np.uint32)],verbose=False)
    held=probe.predict(x.loc[valid]);metric=ndcg10(target[valid],held)
    rng=np.random.default_rng(42);random_metric=ndcg10(target[valid],rng.random(valid.sum()))
    iterations=int(probe.best_iteration)+1
    del probe
    # Final fit uses each original advert exactly once. No Cartesian expansion.
    model=XGBRanker(**params,n_estimators=iterations)
    model.fit(x,target,qid=np.zeros(len(x),dtype=np.uint32),verbose=False)
    scores=model.predict(x)
    scores[np.asarray(prototype_positions,dtype=int)]=-np.inf
    order=np.argsort(-scores,kind='stable')[:10]
    return {'positions':order.tolist(),'scores':scores[order].tolist(),'ndcg10':metric,'random_ndcg10':random_metric,'training_rows':int(train.sum()),'validation_rows':int(valid.sum()),'final_rows':len(x),'rounds':iterations,'direct_judgments':10,'feature_importance':dict(zip(x.columns,model.feature_importances_.astype(float).tolist()))}

def load(root):
    folder=Path(root)/'data/taste-garage';file=folder/'manifest.json'
    if not file.exists():return None
    manifest=json.loads(file.read_text())
    for name,digest in manifest['sha256'].items():
        p=folder/name
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('Recommendation asset integrity mismatch')
    with np.load(folder/'features.npz',allow_pickle=False) as data:arrays={key:data[key] for key in data.files}
    n=len(arrays['clusters'])
    if n!=manifest['rows'] or arrays['codes'].shape!=(n,len(CATEGORICAL)) or arrays['numeric'].shape!=(n,len(NUMERIC)):raise ValueError('Invalid feature matrix')
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
