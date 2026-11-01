"""Train and audit small retrieval-only mixed-type imputation artifacts."""
import gzip,hashlib,json
from pathlib import Path
import numpy as np
from pricing.comparable_retrieval import reference_rows,valid
from pricing.retrieval_imputation import RetrievalImputer,FIELDS,missing,export,restore,fingerprint

def build(root,side):
    rows=[r for r in reference_rows(root,side) if valid(r,allow_missing_miles=True)]
    held=[i for i,r in enumerate(rows) if int(hashlib.sha256(str(r.get('id',i)).encode()).hexdigest()[:8],16)%5==0]
    held_set=set(held);train=[r for i,r in enumerate(rows) if i not in held_set]
    probe=RetrievalImputer().fit(train);masked=[dict(rows[i]) for i in held];rng=np.random.default_rng(42);cells=[]
    for i,row in enumerate(masked):
        for field in FIELDS[2:]:
            if not missing(row.get(field)) and rng.random()<.2:
                cells.append((i,field,str(row[field]).strip().lower()));row[field]=None
    cats,nums,_=probe.transform(masked);audit=[]
    for field in FIELDS[2:]:
        examples=[(i,true) for i,f,true in cells if f==field]
        if not examples:continue
        modes=[str(r[field]).strip().lower() for r in train if not missing(r.get(field))]
        mode=max(set(modes),key=modes.count) if modes else '__missing__'
        audit.append({'field':field,'masked_cells':len(examples),'held_out_accuracy':float(np.mean([cats[i][FIELDS.index(field)]==true for i,true in examples])),
                      'mode_baseline_accuracy':float(np.mean([true==mode for _,true in examples]))})
    # Serving does not deserialize arbitrary pickle code: portable text models and schedule.
    final=RetrievalImputer().fit(rows);data=export(final);portable=restore(data)
    test=[dict(rows[0],odometer=None,fuel=None,transmission=None)]
    a=final.transform(test);b=portable.transform(test)
    assert a[0]==b[0] and np.allclose(a[1],b[1])
    folder=Path(root)/'data/retrieval-imputation';folder.mkdir(exist_ok=True)
    artifact=folder/(side+'.json.gz')
    with gzip.open(artifact,'wt') as f:json.dump(data,f,allow_nan=False)
    if artifact.stat().st_size>8*1024*1024:raise ValueError('Imputation serving artifact exceeds 8 MiB')
    report={'market':side,'method':'sklearn IterativeImputer with native-categorical LightGBM classifiers for optional categories; LightGBM regression for year/log-mileage. CPU, small trees, observed values restored.',
            'reference_sha256':fingerprint(rows),'model_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),'artifact_bytes':artifact.stat().st_size,
            'rows':len(rows),'held_out_audit_rows':len(held),'masked_category_reconstruction':audit,
            'limits':'Randomly masked cells in held-out vehicles measure reconstruction only, not real missing-not-at-random performance. Inferred values are used only in retrieval, never verified facts or production valuation inputs. Entirely unobserved categorical fields remain explicit unknown categories.'}
    (folder/(side+'.json')).write_text(json.dumps(report,indent=2,allow_nan=False));print(side,report['rows'],report['artifact_bytes'],flush=True)
if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    for side in ('ask','bid','hybrid'):build(root,side)
