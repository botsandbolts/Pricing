"""Recover complete canonical descriptions by identity, never use GUI excerpts."""
import hashlib,json,subprocess,sys,urllib.request
from pathlib import Path
import pandas as pd
root=Path('training-cache');root.mkdir(exist_ok=True)
archive=root/'original-ask.zip'
if not archive.exists():urllib.request.urlretrieve('https://zenodo.org/records/10457828/files/Used%20Cars%20Dataset.zip?download=1',archive)
assert hashlib.file_digest(open(archive,'rb'),'sha256').hexdigest()=='eb2ad6c7dedd3c73ffed4bf47929ab66e55d7da186ed240230caa1fad85d2e8f'
source=root/'full-text-cohort.jsonl'
subprocess.run([sys.executable,'-m','scripts.prepare_full',str(archive),'--output',str(source),'--workers','2'],check=True)
folder=root/'lemon-mixed';frame=pd.read_json(folder/'cohort.jsonl',lines=True);frame['row_id']=frame.row_id.astype(str)
needed=set(frame.row_id);mapping={}
for chunk in pd.read_json(source,lines=True,chunksize=5000):
    for row in chunk.loc[chunk.row_id.astype(str).isin(needed)].to_dict('records'):
        mapping[str(row['row_id'])]=(row['text'],row['price'])
assert set(mapping)==needed
assert all(abs(float(row.price)-float(mapping[row.row_id][1]))<.01 for row in frame.itertuples())
frame['text']=frame.row_id.map(lambda rid:mapping[rid][0])
frame.to_json(folder/'cohort.jsonl',orient='records',lines=True,date_format='iso',double_precision=15)
print('Recovered complete canonical descriptions for',len(frame),'identity-matched cars')
