"""Join the original merged network output to the serving catalogue by row ID."""
import hashlib,json,sqlite3,zlib
from pathlib import Path
from pricing.dealer_signals import candidates

def main():
    folder=Path('data/merchant-discovery-minilm')
    source=Path('training-cache/gliner-merchants/merged.json')
    extracted=json.loads(source.read_text())
    lookup={str(r['row_id']):candidates(r['merchant_candidates']) for r in extracted}
    assert len(lookup)==len(extracted)==110832
    rows=[]
    with sqlite3.connect(str(folder/'cars.sqlite')) as db:
        for expected,(position,payload) in enumerate(db.execute('SELECT position,payload FROM cars ORDER BY position')):
            assert position==expected
            car=json.loads(zlib.decompress(payload)); key=str(car['row_id'])
            assert key in lookup, f'Missing GLiNER source for {key}'
            rows.append(lookup[key])
    assert len(rows)==110832
    report={'rows':len(rows),'rows_with_candidates':sum(bool(r) for r in rows),
            'unique_candidate_names':len({e['name'].casefold() for row in rows for e in row}),
            'source_run':41602344245,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'method':'Original GLiNER candidates; repeated mentions use maximum confidence; exact case/whitespace normalization only; no manual filtering or fuzzy merges',
            'feature_weight':'0.5 * confidence / sqrt(candidate count), sparse one-hot dealer block; L2 regularized with other ranking features',
            'identity_status':'Inferred merchant candidates, not verified identities; original network errors remain possible'}
    (folder/'dealers.json').write_text(json.dumps({'rows':rows,'report':report},separators=(',',':')))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
