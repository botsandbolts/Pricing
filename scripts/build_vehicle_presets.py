"""Exactly 1,000 specific historical vehicles, not make/model popularity."""
import hashlib,json
from pathlib import Path
from pricing.comparable_retrieval import reference_rows

def examples(root):
    ask=reference_rows(root,'ask');bid=reference_rows(root,'bid')
    rank=lambda r:hashlib.sha256(('specific-vehicles-v1:'+str(r['id'])).encode()).hexdigest()
    rows=sorted(ask,key=rank)[:1000]
    if len(rows)<1000:rows+=sorted(bid,key=rank)[:1000-len(rows)]
    if len(rows)!=1000 or len({r['id'] for r in rows})!=1000:raise ValueError('Need 1,000 distinct source vehicles')
    rows=sorted(rows,key=lambda r:(str(r['manufacturer']).lower(),str(r['model']).lower(),r['year'],r['odometer'] if r.get('odometer') is not None else -1,str(r['id'])))
    return {'selection':'1,000 distinct historical vehicle records selected deterministically from the original reference data; no popularity ranking. Asking examples include available seller-description text; auction examples have no seller description. Edit mileage and condition for your valuation date.','vehicles':rows}

def build(root):
    root=Path(root);data=examples(root);out=root/'data/vehicle-examples.json';out.write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')));print('Prepared 1,000 specific vehicle examples')
if __name__=='__main__':build(Path(__file__).resolve().parents[1])
