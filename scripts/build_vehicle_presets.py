"""Exactly 1,000 specific historical vehicles, not make/model popularity."""
import hashlib,json
from pathlib import Path
from pricing.comparable_retrieval import reference_rows

def examples(root):
    ask=reference_rows(root,'ask');bid=reference_rows(root,'bid')
    rank=lambda r:hashlib.sha256(('specific-vehicles-v1:'+str(r['id'])).encode()).hexdigest()
    rows=sorted(ask,key=rank)[:500]+sorted(bid,key=rank)[:500]
    if len(rows)<1000:
        used={r['id'] for r in rows}
        rows+=sorted([r for r in ask+bid if r['id'] not in used],key=rank)[:1000-len(rows)]
    if len(rows)!=1000 or len({r['id'] for r in rows})!=1000:raise ValueError('Need 1,000 distinct source vehicles')
    rows=sorted(rows,key=rank)
    return {'selection':'1,000 individual historical cars, mixing asking adverts and auction records in shuffled order. Asking examples include available seller-description text; auction examples have no seller description. Edit mileage and condition for your valuation date.','vehicles':rows}

def build(root):
    root=Path(root);data=examples(root);out=root/'data/vehicle-examples.json';out.write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')));print('Prepared 1,000 specific vehicle examples')
if __name__=='__main__':build(Path(__file__).resolve().parents[1])
