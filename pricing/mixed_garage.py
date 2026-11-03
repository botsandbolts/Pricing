"""Mixed car geometry: common-category Dice and bounded numeric differences."""
from datetime import date
import numpy as np,pandas as pd
from numba import njit

AS_OF=date(2026,11,1)
KIND='lemon-textprice-onehot-umap-v1'


@njit(cache=True)
def mixed_distance(x,y,n_cat,n_num):
    intersection=0.;size=0.
    for i in range(n_cat):
        intersection+=x[i]*y[i];size+=x[i]+y[i]
    categorical=1.-2.*intersection/size if size>0 else 0.
    numeric=0.
    for i in range(n_cat,n_cat+n_num):numeric+=abs(x[i]-y[i])
    return .5*categorical+.5*numeric/n_num


def normalized(x):
    x=np.asarray(x,dtype=np.float32)
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)


def comparison_vectors(matrix,state):
    c=state['onehot_dimensions']
    return np.ascontiguousarray(normalized(np.column_stack([normalized(matrix[:,:c]),normalized(matrix[:,c:])])/np.sqrt(2)),dtype=np.float32)


def numeric_view(frame,transport):
    from pricing.taste_ranking import encode,adjusted_price_features
    codes,numeric,levels=encode(frame)
    prices=pd.to_numeric(frame.get('price',pd.Series(np.nan,index=frame.index)),errors='coerce').to_numpy(float)
    source=frame.get('date',frame.get('observed_date',pd.Series(None,index=frame.index)))
    dates=pd.to_datetime(source,utc=True,errors='coerce')
    supported=dates.notna().to_numpy()&np.isfinite(prices)&(prices>0)
    adjusted=np.full(len(frame),np.nan)
    if supported.any():
        subset=np.flatnonzero(supported);days=(dates.iloc[subset].astype('int64')//86400000000000).to_numpy()
        adjusted[subset]=adjusted_price_features(codes[subset],numeric[subset],levels,prices[subset],days,AS_OF,transport)
    years=pd.to_numeric(frame.get('year',pd.Series(np.nan,index=frame.index)),errors='coerce').to_numpy(float)
    miles=pd.to_numeric(frame.get('odometer',pd.Series(np.nan,index=frame.index)),errors='coerce').to_numpy(float)
    valid=np.isfinite(miles)&(miles>=0)&(miles<=2e6)
    logs=np.where(valid,np.log1p(np.maximum(miles,0)),np.nan)
    good_price=np.isfinite(adjusted)&(adjusted>0)
    log_price=np.full(len(frame),np.nan);log_price[good_price]=np.log(adjusted[good_price])
    text_price=pd.to_numeric(frame.get('predicted_text_price',pd.Series(np.nan,index=frame.index)),errors='coerce').to_numpy(float)
    good_text=np.isfinite(text_price)&(text_price>0)
    log_text=np.full(len(frame),np.nan);log_text[good_text]=np.log(text_price[good_text])
    return np.column_stack([years,logs,~valid,log_price,~good_price,log_text,~good_text]),adjusted


def fit_geometry(frame,transport):
    from pricing.taste_ranking import one_hot_geometry
    if 'predicted_text_price' not in frame:raise ValueError('Stored Lemon predicted_text_price column is required; do not substitute XGBoost predicted_price')
    prices=pd.to_numeric(frame.predicted_text_price,errors='coerce').to_numpy(float)
    if not (np.isfinite(prices)&(prices>0)).all():raise ValueError('Every retained training row needs its existing text-price value')
    base,hot,cat=one_hot_geometry(frame,return_state=True)
    raw,adjusted=numeric_view(frame,transport)
    if np.isfinite(adjusted).mean()<.5:raise ValueError('Insufficient supported November 2026 price scenarios')
    medians=np.nanmedian(raw,axis=0);filled=np.where(np.isfinite(raw),raw,medians)
    low=np.quantile(filled,.01,axis=0);high=np.quantile(filled,.99,axis=0)
    low[[2,4,6]]=0.;high[[2,4,6]]=1.
    spans=np.maximum(high-low,1e-6)
    state={'kind':KIND,'categorical':cat,'onehot_dimensions':hot.shape[1],'numeric_dimensions':7,
           'numeric_names':['vehicle_year','log_mileage','mileage_missing','log_asking_price_nov2026','asking_price_nov2026_missing','log_predicted_text_price','predicted_text_price_missing'],
           'medians':medians,'low':low,'spans':spans,'time_transport':transport,
           'price_date':AS_OF.isoformat(),'text_price_source':'existing cross-fitted Lemon price column; not recomputed',
           'metric':'0.5 * Dice(one-hot) + 0.5 * mean absolute difference(robust range-scaled numeric)',
           'weights':{'categorical':.5,'numeric':.5}}
    numeric=np.clip((filled-low)/spans,0,1).astype(np.float32)
    geometry=np.column_stack([hot,numeric]).astype(np.float32)
    if not np.isfinite(geometry).all():raise ValueError('Nonfinite mixed geometry')
    return geometry,state,adjusted


def transform_geometry(frame,state):
    from pricing.taste_ranking import transform_new_geometry
    base=transform_new_geometry(frame,state['categorical']);hot=base[:,:state['onehot_dimensions']]
    raw,_=numeric_view(frame,state['time_transport']);filled=np.where(np.isfinite(raw),raw,state['medians'])
    numeric=np.clip((filled-state['low'])/state['spans'],0,1)
    return np.column_stack([hot,numeric]).astype(np.float32)


def direct_hdbscan_geometry(matrix,state):
    """Weighted L1 equals 50% nominal mismatch rate + 50% mean numeric difference."""
    c=int(state['onehot_dimensions']);n=int(state['numeric_dimensions'])
    fields=len(state['categorical']['categories'])
    if fields<1 or n<1:raise ValueError('Categorical and numeric feature blocks are required')
    weights=np.r_[np.full(c,.25/fields),np.full(n,.5/n)].astype(np.float64)
    x=np.asarray(matrix,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=len(weights) or not np.isfinite(x).all():
        raise ValueError('Finite aligned mixed feature matrix required')
    return np.ascontiguousarray(x*weights),weights


def euclidean_hdbscan_geometry(matrix,state):
    """Balance nominal mismatches and numeric squared differences before Euclidean HDBSCAN."""
    c=int(state['onehot_dimensions']);n=int(state['numeric_dimensions'])
    fields=len(state['categorical']['categories'])
    if fields<1 or n<1:raise ValueError('Both feature blocks are required')
    weights=np.r_[np.full(c,np.sqrt(.25/fields)),np.full(n,np.sqrt(.5/n))]
    x=np.asarray(matrix,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=len(weights) or not np.isfinite(x).all():raise ValueError('Invalid mixed features')
    return np.ascontiguousarray(x*weights),weights
