"""Prepare checkpoint 0 and encode every source row with that SAME checkpoint."""
import argparse,json,hashlib,shutil
from pathlib import Path
import numpy as np,pandas as pd
from pricing.adapted_bge import AdaptedEncoder
from pricing.electra_embeddings import file_sha256,shard_positions
p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--shard',type=int);a=p.parse_args()
if a.prepare:
    if file_sha256('training-cache/full-cohort.jsonl')!='746bc9f8e02e1e708ea5e1d9479408effb131eb6b1d1ea7710380701e62dcf12':raise ValueError('Wrong canonical cohort')
    files=list(Path('adapted-source').rglob('adaptation.json'))
    if len(files)!=1:raise ValueError('Expected exactly one adapted checkpoint')
    src=files[0].parent
    receipt=json.loads(files[0].read_text())
    if receipt['shard']!=0:raise ValueError('Checkpoint 0 is preselected')
    dest=Path('adapted-bge');shutil.copytree(src,dest,dirs_exist_ok=True)
    hashes={f.name:file_sha256(f) for f in dest.iterdir() if f.is_file()}
    identity=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    (dest/'bundle.json').write_text(json.dumps({'identity':identity,'files':hashes,'adaptation':receipt},indent=2))
    encoder=AdaptedEncoder(dest)
    vectors=encoder.encode(['A red sedan.','A red sedan.'])
    if vectors.shape!=(2,384) or not np.allclose(vectors[0],vectors[1]):raise ValueError('Adapted BGE smoke failed')
    from pricing.minilm_signal import Encoder as MiniLMEncoder
    mini=MiniLMEncoder('minilm').encode(['A red sedan.'])
    if mini.shape!=(1,384) or not np.allclose(np.linalg.norm(mini,axis=1),1,atol=1e-3):raise ValueError('MiniLM smoke failed')
    print('Pinned adapted checkpoint 0',identity,flush=True)
else:
    if a.shard is None or not 0<=a.shard<25:raise ValueError('Shard 0..24 required')
    from pricing.minilm_signal import Encoder as MiniLMEncoder
    from scripts.common import clean
    encoder=AdaptedEncoder('adapted-bge');mini=MiniLMEncoder('minilm');ids=[];vectors=[];minis=[];offset=0
    for chunk in pd.read_json('training-cache/full-cohort.jsonl',lines=True,chunksize=1000,dtype={'row_id':str}):
        selected=chunk.loc[shard_positions(offset,len(chunk),a.shard)];offset+=len(chunk)
        if selected.empty:continue
        texts=selected.text.fillna('').astype(str).tolist()
        vectors.append(encoder.encode(texts))
        from pricing.advert_text import scrub
        minis.append(mini.encode([clean(scrub(t)) for t in texts]))
        ids.extend(selected.row_id.astype(str).tolist());print(a.shard,len(ids),flush=True)
    out=Path('adapted-vectors');out.mkdir(exist_ok=True)
    np.savez_compressed(out/f'{a.shard:02d}.npz',ids=np.asarray(ids),embeddings=np.concatenate(vectors),minilm=np.concatenate(minis))
    (out/f'{a.shard:02d}.json').write_text(json.dumps({'shard':a.shard,'rows':len(ids),'source_rows':offset,
      'source_sha256':file_sha256('training-cache/full-cohort.jsonl'),
      'encoder_identity':json.loads(Path('adapted-bge/bundle.json').read_text())['identity']}))
