"""Freeze the CatBoost recipe into a portable Kaggle notebook (no credentials)."""
import base64,hashlib,io,json,zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
b=io.BytesIO()
with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
    for folder in ['pricing','marketbridge','carbids','scripts']:
        for p in sorted((root/folder).glob('*.py')):z.write(p,str(p.relative_to(root)))
    for p in [root/'models/structural-transfer/used-car-index.csv',root/'models/minilm-cleanlab/text-model/provenance.json']:
        z.write(p,str(p.relative_to(root)))
blob=b.getvalue();encoded=base64.b64encode(blob).decode();sha=hashlib.sha256(blob).hexdigest()
def cell(kind,text):return {'cell_type':kind,'metadata':{},'source':text.splitlines(True),**({'outputs':[],'execution_count':None} if kind=='code' else {})}
cells=[cell('markdown','''# Car Price Studio — Five-stage CatBoost release
Full eligible asking and auction cohorts. Five rounds remove ceil(5% of remaining rows) per original train/validation/test partition. The same cleaned cohort trains both AFT and median quantile models; asking retains verified MiniLM stacking.

GPU: cleaning + quantile. CPU: SurvivalAft (CatBoost does not support it on GPU). Tests include unchanged original test cohorts and unfiltered training baselines. Filtered test scores are label-selected and optimistic. This is an experimental new formula, not a promised accuracy gain.

Attach `botsandbolts/notebookf1cefc6227` (verified features) and `botsandbolts/pricing` (canonical cohort). Internet ON, GPU T4. Save & Run All persists artifacts, then the batch GPU stops automatically.
'''),cell('code',f'''import sys,subprocess,os,json,io,zipfile,base64,hashlib,shutil
from pathlib import Path
ROOT=Path('/kaggle/working/pricing');ROOT.mkdir(exist_ok=True)
blob=base64.b64decode({encoded!r})
assert hashlib.sha256(blob).hexdigest()=={sha!r}
zipfile.ZipFile(io.BytesIO(blob)).extractall(ROOT)
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
subprocess.run([sys.executable,'-m','pip','install','-q','catboost==1.2.10','lightgbm>=4.5,<5','streamlit>=1.51,<2','pytest'],check=True)
from catboost import CatBoostRegressor
import numpy as np
CatBoostRegressor(loss_function='Quantile:alpha=0.5',iterations=2,depth=2,task_type='GPU',devices='0',allow_writing_files=False).fit(np.arange(60).reshape(30,2),np.arange(30.),verbose=False)
print('GPU quantile smoke passed; source SHA:',{sha!r})
'''),cell('code','''# Rebuild canonical cohorts, preserving row identities and chronological phases.
from carbids.data import download,prepare
cache=ROOT/'training-cache';cache.mkdir(exist_ok=True)
inputs=Path('/kaggle/input')
ASK=cache/'full-cohort.jsonl';BID=cache/'bid-cohort.jsonl'
canonical=list(inputs.rglob('training-cache/full-cohort.jsonl'))
if not ASK.exists() and len(canonical)==1:
    shutil.copy2(canonical[0],ASK)
if not ASK.exists():
    vehicles=list(inputs.rglob('vehicles.csv'))
    assert len(vehicles)==1, 'Attach the Pricing dataset with canonical cohort or vehicles.csv'
    source=cache/'vehicles.zip'
    if not source.exists():
        with zipfile.ZipFile(source,'w',zipfile.ZIP_STORED) as z:z.write(vehicles[0],'vehicles.csv')
    subprocess.run([sys.executable,'-m','scripts.prepare_full',str(source),'--output',str(ASK),'--workers','2'],check=True)
if not BID.exists():prepare(download(cache/'auctions.zip'),BID)
features=list(inputs.rglob('minilm-features/provenance.json'))
assert len(features)==1, 'Attach the verified-feature notebook output exactly once'
FEATURES=features[0].parent
from pricing.suite_text import validated_features
f=validated_features(ASK,FEATURES)
assert len(f)==211844
print('Verified MiniLM assessment and full OOF:',len(f))
'''),cell('code','''# Save each completed stage and assessment; no reusing old quartile masks.
command=[sys.executable,'-u','-m','scripts.train_iterative_catboost','--ask',str(ASK),'--bid',str(BID),
         '--text-features',str(FEATURES),'--device','GPU','--threads','4',
         '--cleaner-iterations','400','--iterations','1500','--output','models/iterative-catboost']
log=ROOT/'iterative-training.log'
with log.open('w') as f:
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in process.stdout:print(line,end='');f.write(line);f.flush()
    code=process.wait()
if code:raise RuntimeError('Training failed; see preserved iterative-training.log')
report=json.loads((ROOT/'models/iterative-catboost/summary.json').read_text())
assert report['status']=='complete' and len(report['models'])==4
print('ALL FOUR FULL-COHORT MODELS COMPLETE')
'''),cell('code','''# Package serving bundles + reproducible evidence, without source listing descriptions.
import gzip
from catboost import CatBoostRegressor
for name in ['ask','ask-quantile','bid','bid-aft']:
    folder=ROOT/'models/iterative-catboost'/name
    m=json.loads((folder/'manifest.json').read_text());model=CatBoostRegressor()
    model.load_model(blob=gzip.decompress((folder/m['model_file']).read_bytes()))
    assert model.feature_names_==m['feature_names']
    e=json.loads((folder/'evaluation.json').read_text())
    print(name,'original test MAE:',round(e['assessment']['original_unfiltered_test']['mae_usd'],2),
          'change vs unfiltered fit:',round(e['original_test_mae_change_usd'],2))
out=Path('/kaggle/working/Car_Price_Studio_Iterative_CatBoost.zip')
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted((ROOT/'models/iterative-catboost').rglob('*')):
        if p.is_file():z.write(p,str(p.relative_to(ROOT)))
    z.write(ROOT/'scripts/train_iterative_catboost.py','scripts/train_iterative_catboost.py')
print('READY — download',out,'; GPU batch ends after this cell.')
''')]
notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11.0'}},'nbformat':4,'nbformat_minor':5}
p=root/'train_iterative_catboost_kaggle.ipynb';p.write_text(json.dumps(notebook,indent=1))
Path('/workspace/scratch/train_iterative_catboost_kaggle.ipynb').write_bytes(p.read_bytes())
print(p,'source SHA',sha)
