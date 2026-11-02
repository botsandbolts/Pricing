"""Join the existing text-price scalar to the identity-checked retained Lemon rows."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd


def prepare(source,evidence,text_prices,output):
    source=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    expected='746bc9f8e02e1e708ea5e1d9479408effb131eb6b1d1ea7710380701e62dcf12'
    actual=hashlib.file_digest(source.open('rb'),'sha256').hexdigest()
    if actual!=expected:raise ValueError('Original source identity mismatch')
    pred=pd.read_csv(evidence,dtype={'row_id':str});kept=pred.loc[pred.split.eq('calibration')].copy()
    assert kept.row_id.is_unique and len(kept)==110832
    stored=pd.read_csv(text_prices,dtype={'row_id':str})
    assert stored.row_id.is_unique and set(stored.row_id)==set(kept.row_id)
    assert np.isfinite(stored.predicted_text_price).all() and (stored.predicted_text_price>0).all()
    keep=set(kept.row_id);pieces=[]
    for chunk in pd.read_json(source,lines=True,chunksize=5000):
        chunk['row_id']=chunk.row_id.astype(str);selected=chunk.loc[chunk.row_id.isin(keep)].copy()
        if len(selected):pieces.append(selected)
    cohort=pd.concat(pieces,ignore_index=True);assert len(cohort)==len(keep)
    cohort=cohort.merge(stored[['row_id','predicted_text_price']],on='row_id',validate='one_to_one',sort=False)
    check=cohort[['row_id','price']].merge(kept[['row_id','observed_price']],on='row_id',validate='one_to_one')
    assert np.allclose(check.price,check.observed_price)
    cohort.to_json(out/'cohort.jsonl',orient='records',lines=True,date_format='iso')
    ids=cohort.row_id.astype(str).tolist();(out/'row-ids.json').write_text(json.dumps(ids))
    selection={'source_rows':211844,'retained_rows':len(cohort),'scope':'Only retained calibration/training rows; validation/test excluded',
               'source_sha256':actual,'selection_evidence_sha256':hashlib.file_digest(open(evidence,'rb'),'sha256').hexdigest(),
               'retained_id_sha256':hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest(),
               'text_price_file_sha256':hashlib.file_digest(open(text_prices,'rb'),'sha256').hexdigest(),
               'text_price_source':'Stored minilm-cleaned/ask/text-price-oof.csv.gz cross-fitted dollar means; no embeddings or fresh MiniLM inference'}
    (out/'selection.json').write_text(json.dumps(selection,indent=2));print(json.dumps(selection,indent=2),flush=True)


def from_catalogue(root,output):
    import gzip,sqlite3,zlib
    from pricing.taste_ranking import load
    root=Path(root);folder=root/'data/taste-garage-mixed';meta,_=load(root,catalogue_path=folder)
    with sqlite3.connect(folder/'cars.sqlite') as conn:
        rows=[json.loads(zlib.decompress(r[0])) for r in conn.execute('SELECT payload FROM cars ORDER BY position')]
    frame=pd.DataFrame(rows).rename(columns={'id':'row_id','observed_date':'date'})
    ids=frame.row_id.astype(str).tolist();selection=json.loads((folder/'selection.json').read_text())
    assert len(ids)==len(set(ids))==selection['retained_rows']==110832
    assert hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()==selection['retained_id_sha256']
    assert np.isfinite(frame.predicted_text_price).all() and (frame.predicted_text_price>0).all()
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    frame.to_json(out/'cohort.jsonl',orient='records',lines=True,date_format='iso',double_precision=15)
    (out/'row-ids.json').write_text(json.dumps(ids));(out/'selection.json').write_text(json.dumps(selection,indent=2))
    print('Restored',len(frame),'verified Lemon rows and their existing text-price scalar')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='training-cache/full-cohort.jsonl');p.add_argument('--evidence',default='training-cache/lemon-evidence/ask-minilm-cleaned.csv.gz');p.add_argument('--text-prices',default='training-cache/lemon-text-prices/retained-text-prices.csv.gz');p.add_argument('--output',default='training-cache/lemon-mixed');p.add_argument('--from-catalogue',action='store_true');a=p.parse_args()
    if a.from_catalogue:from_catalogue('.',a.output)
    else:prepare(a.source,a.evidence,a.text_prices,a.output)
