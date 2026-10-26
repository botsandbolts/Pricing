"""Lossless multipart storage for large gzip LightGBM files.
The connector has a 16 MiB request limit; 8 MiB binary parts fit when base64 encoded.
"""
import argparse,gzip,hashlib,io,json
from pathlib import Path

def payload_files(member):return member.get('parts',[member['file']])

def read_model_text(folder,member):
    folder=Path(folder)
    if 'parts' not in member:
        with gzip.open(folder/member['file'],'rt') as stream:return stream.read()
    payload=b''.join((folder/name).read_bytes() for name in member['parts'])
    if hashlib.sha256(payload).hexdigest()!=member['compressed_sha256']:raise ValueError('Ensemble model part checksum mismatch')
    return gzip.decompress(payload).decode('utf-8')

def pack_bundle(folder,part_bytes=8*1024*1024):
    folder=Path(folder);file=folder/'manifest.json';manifest=json.loads(file.read_text())
    if not 0<part_bytes<=8*1024*1024:raise ValueError('Parts must be at most 8 MiB')
    for member in manifest['members']:
        if 'parts' in member:
            read_model_text(folder,member);continue
        original=folder/member['file'];digest=hashlib.sha256();parts=[]
        with original.open('rb') as stream:
            i=0
            while data:=stream.read(part_bytes):
                name=member['file']+f'.part{i:02d}';(folder/name).write_bytes(data);parts.append(name);digest.update(data);i+=1
        member['parts']=parts;member['compressed_sha256']=digest.hexdigest()
        # Validate exact gzip payload and decoded contents before removing this generated whole-file copy.
        assert read_model_text(folder,member)==gzip.decompress(original.read_bytes()).decode('utf-8')
    manifest['storage']='lossless gzip payloads split into verified parts of at most 8 MiB'
    temporary=file.with_suffix('.tmp');temporary.write_text(json.dumps(manifest,indent=2));temporary.replace(file)
    for member in manifest['members']:(folder/member['file']).unlink(missing_ok=True)
    return {'members':len(manifest['members']),'parts':sum(len(m['parts']) for m in manifest['members']),'max_part_bytes':part_bytes}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args();print(json.dumps(pack_bundle(a.folder)))
