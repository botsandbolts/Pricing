"""Package only the calibration refit needed to score a new vehicle."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def package(source,output,side):
    source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    report=json.loads((source/'evaluation.json').read_text())
    if report['market']!=side or report['folds']!=5:raise ValueError('Wrong source assessment')
    names=['calibration-refit.cbm','evaluation.json']
    if side=='ask':names.append('calibration-refit-tfidf-umap.joblib')
    for name in names:
        src=source/name
        if not src.is_file():raise FileNotFoundError(src)
        shutil.copy2(src,output/name)
    # A posterior is included only if it was fitted from this same OOF assessment.
    bayes=source/'bayesian'
    if bayes.exists():
        for name in ['posterior.npz','bayesian.json.gz','bayesian-evaluation.json']:
            if (bayes/name).exists():
                (output/'bayesian').mkdir(exist_ok=True)
                shutil.copy2(bayes/name,output/'bayesian'/name)
    manifest={'market':side,'base_kind':'fivefold-calibration-only-catboost-refit',
        'description':'fold-fitted TF-IDF and UMAP in ask only',
        'features':['age','log_miles','auction_grade','manufacturer','model','type','transmission','state','fuel','drive','title_status']+([f'description_umap_{j}' for j in range(1,9)] if side=='ask' else []),
        'test_mae_usd':report['scores']['test']['mae_usd'],
        'files':{str(p.relative_to(output)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in output.rglob('*') if p.is_file()}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'market':side,'bytes':sum(v['bytes'] for v in manifest['files'].values()),'files':manifest['files']},indent=2))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--source',required=True);a.add_argument('--output',required=True);a.add_argument('--market',choices=['ask','bid'],required=True)
    x=a.parse_args();package(x.source,x.output,x.market)
