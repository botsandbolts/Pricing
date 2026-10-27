"""Losslessly split large fitted AFT files into checksum-verified Git blobs."""
import gzip
import hashlib
import json
from pathlib import Path


def read_model_file(path):
    path=Path(path);registry=Path(str(path)+'.parts.json')
    if not registry.exists():return path.read_bytes()
    spec=json.loads(registry.read_text());chunks=[]
    for part in spec['parts']:
        if Path(part['file']).name!=part['file']:raise ValueError('Invalid model part name')
        raw=(path.parent/part['file']).read_bytes()
        if len(raw)!=part['bytes'] or hashlib.sha256(raw).hexdigest()!=part['sha256']:raise ValueError('Model part checksum mismatch')
        chunks.append(raw)
    raw=b''.join(chunks)
    if len(raw)!=spec['bytes'] or hashlib.sha256(raw).hexdigest()!=spec['sha256']:raise ValueError('Model checksum mismatch')
    return raw


def split_model(path,part_bytes=8*1024*1024):
    path=Path(path);raw=path.read_bytes();parts=[]
    for i,start in enumerate(range(0,len(raw),part_bytes)):
        piece=raw[start:start+part_bytes];name=path.name+f'.{i:03d}.part'
        (path.parent/name).write_bytes(piece)
        parts.append({'file':name,'bytes':len(piece),'sha256':hashlib.sha256(piece).hexdigest()})
    spec={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'parts':parts}
    Path(str(path)+'.parts.json').write_text(json.dumps(spec,indent=2))
    assert read_model_file(path)==raw
    path.unlink();return spec
