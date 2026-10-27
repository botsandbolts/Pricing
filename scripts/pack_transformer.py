"""Lossless, checksum-verified Git-friendly storage for a Hugging Face backbone."""
import argparse,gzip,hashlib,json
from pathlib import Path


def pack(path):
    path=Path(path);source=path/'model.safetensors'
    raw=source.read_bytes();payload=gzip.compress(raw,compresslevel=6,mtime=0);parts=[]
    for i,start in enumerate(range(0,len(payload),8*1024*1024)):
        data=payload[start:start+8*1024*1024];name=f'weights-{i:03d}.gz.part'
        (path/name).write_bytes(data);parts.append({'file':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    spec={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'compressed_sha256':hashlib.sha256(payload).hexdigest(),
        'config_sha256':hashlib.sha256((path/'config.json').read_bytes()).hexdigest(),'parts':parts}
    (path/'weights.json').write_text(json.dumps(spec,indent=2))
    assert gzip.decompress(b''.join((path/p['file']).read_bytes() for p in parts))==raw
    source.unlink();return spec

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('backbone');print(json.dumps(pack(p.parse_args().backbone),indent=2))
