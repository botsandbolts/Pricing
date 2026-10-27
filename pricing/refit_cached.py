"""Cleanlab/default-Huber refit reusing verified, separately cross-fitted MiniLM features."""
import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from pricing.label_quality import cleanlab_inliers
from pricing.suite_text import validated_features
from pricing.tabular import NUMERIC,CATEGORICAL,features,fit_schema
from pricing.model_suite import load_cohort
from carbids.common import metrics


def run(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise ValueError('Choose new model output')
    text=validated_features(a.data,a.text_features);d=load_cohort(a.data)
    tr=np.flatnonzero(d.phase=='train');va=np.flatnonzero(d.phase=='validation');te=np.flatnonzero(d.phase=='test')
    keep,audit=cleanlab_inliers(d.iloc[tr][NUMERIC+CATEGORICAL],d.price.iloc[tr],out/'assessment-cleanlab',d.row_id.iloc[tr])
    schema=fit_schema(d.iloc[tr]);x=features(d,schema,text.loc[d.row_id].to_numpy())
    def fit(x,y):
        m=lgb.LGBMRegressor(objective='huber',n_jobs=4,random_state=42,verbosity=-1)
        m.fit(x,np.log1p(y),categorical_feature=CATEGORICAL);return m.booster_
    model=fit(x.iloc[tr[keep]],d.price.iloc[tr[keep]])
    vp=np.expm1(np.clip(model.predict(x.iloc[va],num_threads=4),0,np.log1p(1e7)))
    tp=np.expm1(np.clip(model.predict(x.iloc[te],num_threads=4),0,np.log1p(1e7)))
    band=float(np.quantile(abs(np.log1p(d.price.iloc[va])-np.log1p(vp)),.8))
    evaluation={'status':'evaluated','rows':{'train':len(tr),'validation':len(va),'test':len(te),'eligible':len(d),
        'retained_train':len(keep),'outlier_removed':len(tr)-len(keep)},'outlier_filter':audit,
        'stacked':metrics(d.price.iloc[te],tp),'validation':metrics(d.price.iloc[va],vp),
        'description_only':metrics(d.price.iloc[te],text.loc[d.row_id.iloc[te]].to_numpy()),
        'selected':{'objective':'huber','target':'log1p','statistical_parameters':'LightGBM defaults, 100 trees'},
        'note':'Separate original chronological held-out assessment. Train-only text head for held predictions and training-only OOF for fitting; no full-cohort text feature used in assessment.',
        'band80_test_coverage':float(np.mean(abs(np.log1p(d.price.iloc[te])-np.log1p(tp))<=band))}
    (out/'evaluation.json').write_text(json.dumps(evaluation,indent=2));print(json.dumps(evaluation),flush=True)
    keep,finalaudit=cleanlab_inliers(d[NUMERIC+CATEGORICAL],d.price,out/'deployment-cleanlab',d.row_id)
    full=text.attrs['full_oof'];schema=fit_schema(d);x=features(d,schema,full.loc[d.row_id].to_numpy())
    model=fit(x.iloc[keep],d.price.iloc[keep])
    with gzip.open(out/'lightgbm.txt.gz','wt') as f:f.write(model.model_to_string())
    (out/'schema.json').write_text(json.dumps(schema,indent=2))
    source=Path(a.full_text_model);p=json.loads((source/'provenance.json').read_text())
    if p['fitting_rows']!=len(d):raise ValueError('Final text model must cover full eligible cohort')
    shutil.copytree(source,out/'text-model',dirs_exist_ok=True)
    fingerprint=hashlib.sha256(Path(a.data).read_bytes()+Path(__file__).read_bytes()+json.dumps(evaluation,sort_keys=True).encode()).hexdigest()
    (out/'manifest.json').write_text(json.dumps({'status':'ready','pipeline':'fine-tuned-minilm-l6+lightgbm',
        'trained_rows':len(keep),'training_candidates':len(d),'outlier_filter':finalaudit,
        'feature_names':model.feature_name(),'target_transform':'log1p','target_mean':0.,'target_std':1.,
        'band80_log_radius':band,'fingerprint':fingerprint,'assessment_note':evaluation['note'],
        'deployment_scope':'full-eligible-data-refit-after-held-out-assessment',
        'text_reuse':'Verified original three-fold MiniLM OOF predictions and original full-cohort fine-tuned text head; LightGBM newly fitted after Cleanlab.',
        'text_provenance':p},indent=2));print('READY',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--text-features',required=True)
    p.add_argument('--full-text-model',required=True);p.add_argument('--output',default='models/minilm-cleanlab')
    run(p.parse_args())
