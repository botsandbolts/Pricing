"""Safe auxiliary auction features and weighted conformal building blocks.
No retail-sale conversion is claimed without matched target-domain outcomes.
"""
import argparse,gzip,hashlib,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from carbids.data import load_cohort,valid_vin
from carbids.inference import PricePipeline

def positions(value,bits,k):
    h=hashlib.sha256(value.encode()).digest();a=int.from_bytes(h[:16],'little');b=int.from_bytes(h[16:],'little')|1
    return [(a+i*b)%bits for i in range(k)]

def index_training_vehicles(cohort,model_dir):
    d=load_cohort(cohort)
    manifest=json.loads((Path(model_dir)/'manifest.json').read_text())
    cohort_hash=hashlib.sha256(Path(cohort).read_bytes()).hexdigest()
    if manifest.get('training_cohort_sha256')!=cohort_hash or manifest.get('training_candidates')!=len(d):raise ValueError('Overlap index must use the exact fitted training cohort')
    values=d.vehicle_group.astype(str).tolist();bits=max(1024,len(values)*24);k=12
    bitmap=bytearray((bits+7)//8)
    for value in values:
        for position in positions(value,bits,k):bitmap[position//8]|=1<<(position%8)
    path=Path(model_dir)
    with gzip.open(path/'training-vehicles.bloom.gz','wb') as f:f.write(bitmap)
    report={'bits':bits,'k':k,'rows':len(values),'cohort_sha256':hashlib.sha256(Path(cohort).read_bytes()).hexdigest(),
        'purpose':'Conservative training-VIN overlap exclusion for auxiliary feature export. False positives cause exclusions; no false negatives for indexed VINs.'}
    (path/'bridge-index.json').write_text(json.dumps(report,indent=2));return report

def seen_vehicle(vin,index,bitmap):
    value='vin:'+vin
    return all(bitmap[p//8]&(1<<(p%8)) for p in positions(value,index['bits'],index['k']))

def auxiliary_features(frame,model_dir,include_asking_gap=False):
    path=Path(model_dir);model=PricePipeline(path)
    if not (path/'bridge-index.json').exists():raise ValueError('Build the training-VIN overlap index before exporting features')
    index=json.loads((path/'bridge-index.json').read_text())
    if index.get('cohort_sha256')!=model.manifest.get('training_cohort_sha256'):raise ValueError('Training-VIN index does not match this fitted model')
    with gzip.open(path/'training-vehicles.bloom.gz','rb') as f:bitmap=f.read()
    rows=[]
    for row in frame.to_dict('records'):
        vin=valid_vin(row.get('vin',row.get('VIN')))
        if not vin:raise ValueError('Valid VINs required to prove training-vehicle disjointness; missing VINs cannot be silently treated as independent')
        if seen_vehicle(vin,index,bitmap):raise ValueError('A requested vehicle may overlap the CarBids training cohort. Use a disjoint model/fold; no in-sample auxiliary prediction is exported.')
        if model.text_model is not None and len(str(row.get('text','')).strip())<80:raise ValueError('Enhanced bridge requires meaningful pre-sale text')
        fields={**row};fields.setdefault('manufacturer',row.get('make'))
        if fields.get('source') not in model.manifest['sources']:fields['source']=model.manifest['sources'][0]
        p=model.predict(str(row.get('text','')),fields)
        result={'auction_prediction_usd':p['price'],'auction_log_price':float(np.log1p(p['price'])),
            'auction_band_low_usd':p['range'][0],'auction_band_high_usd':p['range'][1],
            'auction_estimate_domain':','.join(model.manifest['sources']),
            'requested_source_supported':row.get('source') in model.manifest['sources'],
            'auction_feature_role':'auxiliary_auction_signal_not_retail_sale_conversion',
            'model_year_supported':model.manifest.get('model_year_range',[1886,2027])[0]<=float(row['year'])<=model.manifest.get('model_year_range',[1886,2027])[1]}
        if include_asking_gap and row.get('asking_price_usd') is not None:
            ask=float(row['asking_price_usd'])
            if not np.isfinite(ask) or ask<=0:raise ValueError('Asking price must be positive')
            result['asking_minus_auction_log_gap']=float(np.log1p(ask)-np.log1p(p['price']))
        if 'row_id' in row:result['row_id']=row['row_id']
        rows.append(result)
    return pd.DataFrame(rows)

def effective_sample_size(weights):
    w=np.asarray(weights,float)
    if w.ndim!=1 or not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:raise ValueError('Finite nonnegative weights with positive total required')
    return float(w.sum()**2/(w@w))

def weighted_conformal_radius(scores,weights,test_weight,alpha=.2):
    """Includes the weighted test-point mass at infinity. Returns inf to abstain.
    Coverage requires the weighted exchangeability / covariate-shift assumptions;
    this numerical quantile does not establish those assumptions.
    """
    scores=np.asarray(scores,float);weights=np.asarray(weights,float)
    if scores.ndim!=1 or scores.shape!=weights.shape or len(scores)==0 or not np.isfinite(scores).all() or (scores<0).any():raise ValueError('Finite nonnegative residual scores required')
    effective_sample_size(weights)
    if not 0<alpha<1 or not np.isfinite(test_weight) or test_weight<=0:raise ValueError('Invalid alpha or test weight')
    order=np.argsort(scores);threshold=(1-alpha)*(weights.sum()+test_weight)
    if threshold>weights.sum():return math.inf
    index=np.searchsorted(np.cumsum(weights[order]),threshold,side='left')
    return float(scores[order[min(index,len(scores)-1)]])

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('index');a.add_argument('--cohort',default='data/cohort.jsonl');a.add_argument('--model',default='models/tabular')
    a=sub.add_parser('export');a.add_argument('--include-pre-sale-asking-feature',action='store_true',help='Retail-sale calibration only. Never use when asking price itself is the target.');a.add_argument('--input',required=True);a.add_argument('--output',required=True);a.add_argument('--model',default='models/tabular')
    args=p.parse_args()
    if args.command=='index':print(json.dumps(index_training_vehicles(args.cohort,args.model),indent=2))
    else:auxiliary_features(pd.read_csv(args.input),args.model,include_asking_gap=args.include_pre_sale_asking_feature).to_csv(args.output,index=False)
