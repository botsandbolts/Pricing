"""Rank actual observed make/model labels, preserving source spelling."""
import gzip,json
from pathlib import Path

def build(root):
    root=Path(root)
    source=root/'models/xgboost-hierarchy/ask/bayesian/bayesian.json.gz'
    counts=json.load(gzip.open(source,'rt'))['raw_model_counts']
    rows=[]
    for name,n in sorted(counts.items(),key=lambda x:(-x[1],x[0])):
        if '::' not in name:continue
        make,model=name.split('::',1)
        if make in ('unknown','') or model in ('unknown','') or len(model)>80:continue
        rows.append({'make':make,'model':model,'count':n})
        if len(rows)==1000:break
    if len(rows)!=1000:raise ValueError('Fewer than 1,000 supported combinations')
    target=root/'data/popular-vehicles.json'
    target.write_text(json.dumps({'ranking_basis':'Retained original Craigslist calibration-vehicle counts from the installed Bayesian release; historical 2021 listings, including trim/model label variants. Not current US sales popularity.','vehicles':rows},separators=(',',':')))
if __name__=='__main__':build(Path(__file__).resolve().parents[1])
