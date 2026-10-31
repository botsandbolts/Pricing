"""Fetch only original Kaggle cohort and completed OOF files; verify known identities."""
import hashlib,json,shutil,urllib.parse,zipfile
from pathlib import Path
import requests
ROOT='https://www.kaggle.com/api/v1/datasets/download/botsandbolts/pricing/'

def download(name,dest):
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix(dest.suffix+'.download')
    with requests.get(ROOT+urllib.parse.quote(name,safe=''),stream=True,timeout=(30,300)) as r:
        r.raise_for_status()
        with tmp.open('wb') as f:
            for chunk in r.iter_content(1024*1024):f.write(chunk)
    # NPZ is itself a ZIP container; it is the requested file, not an outer archive.
    if dest.suffix=='.npz':
        tmp.replace(dest)
    elif zipfile.is_zipfile(tmp):
        with zipfile.ZipFile(tmp) as z:
            matches=[n for n in z.namelist() if n==name or n.rsplit('/',1)[-1]==dest.name]
            if len(matches)!=1:raise ValueError('Ambiguous Kaggle download archive')
            with z.open(matches[0]) as src,dest.open('wb') as out:shutil.copyfileobj(src,out)
        tmp.unlink()
    else:tmp.replace(dest)
    print('DOWNLOADED',name,dest.stat().st_size,flush=True)

def main():
    cohort=Path('training-cache/full-cohort.jsonl')
    download('pricing/training-cache/full-cohort.jsonl',cohort)
    expected=json.loads(Path('models/xgboost-sparse-minilm/ask/minilm/provenance.json').read_text())['cohort_sha256']
    actual=hashlib.file_digest(cohort.open('rb'),'sha256').hexdigest()
    if actual!=expected:raise ValueError('Kaggle cohort differs from the verified original cohort: '+actual)
    for fold in range(3):
        name=f'fold-{fold}.npz';dest=Path('training-cache/kaggle-original')/name
        download('pricing/artifacts/car-price-studio-kaggle/assessment/description-oof/'+name,dest)
        original=Path('data/kaggle-minilm-assessment')/name
        if hashlib.sha256(dest.read_bytes()).digest()!=hashlib.sha256(original.read_bytes()).digest():raise ValueError('Original MiniLM OOF archive mismatch')
    (cohort.parent/'kaggle-source.json').write_text(json.dumps({'dataset':'botsandbolts/pricing','cohort_sha256':actual,'original_text_oof':'three original supervised folds verified; not nested in the new five-fold tree CV','new_features':'pinned label-free MiniLM encoder; no reuse of supervised features across outer folds'},indent=2))
if __name__=='__main__':main()
