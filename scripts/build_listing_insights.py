"""Publish small reports from original canonical historical data, never raw adverts."""
import argparse,hashlib,json
from pathlib import Path
import pandas as pd
from pricing.listing_statistics import experiment

def build(path,side,output,max_train=40000,max_other=10000,bootstrap=40):
    # Deterministic group-independent hash sampling within original time partitions.
    held=[];counts={};offset=0
    for chunk in pd.read_json(path,lines=True,chunksize=5000):
        for k,v in chunk.phase.value_counts().items():counts[k]=counts.get(k,0)+int(v)
        chunk['_rank']=chunk.row_id.astype(str).map(lambda s:hashlib.sha256(('listing-insights-v1:'+s).encode()).hexdigest())
        held.append(chunk)
    d=pd.concat(held,ignore_index=True);rows=[]
    for phase,part in d.groupby('phase'):
        limit=max_train if phase=='train' else max_other
        rows.extend(part.sort_values('_rank').head(limit).drop(columns='_rank').to_dict('records'))
    report=experiment(rows,side,bootstrap=bootstrap)
    report['provenance']={'canonical_input_sha256':hashlib.file_digest(open(path,'rb'),'sha256').hexdigest(),
        'original_partition_counts':counts,'sampling':'Deterministic SHA-256 rank within original train/validation/test partitions. Up to 40,000 train, 10,000 validation and 10,000 test records. No residual trimming.',
        'source':'https://zenodo.org/records/10457828' if side=='ask' else 'https://www.kaggle.com/datasets/tunguz/used-car-auction-prices',
        'source_license':'CC-BY-4.0' if side=='ask' else 'CC0-1.0'}
    out=Path(output);out.parent.mkdir(exist_ok=True,parents=True);out.write_text(json.dumps(report,indent=2,allow_nan=False));print(side,report['rows'],report['test_mae_usd'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ask');p.add_argument('--bid');p.add_argument('--output',default='data/listing-insights');p.add_argument('--bootstrap',type=int,default=40);a=p.parse_args()
    for side,path in [('ask',a.ask),('bid',a.bid)]:
        if path:build(path,side,Path(a.output)/(side+'.json'),bootstrap=a.bootstrap)
