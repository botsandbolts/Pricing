"""Split compressed research artifacts into checksum-verified release blobs."""
import argparse,gzip,hashlib,json,shutil
from pathlib import Path
from pricing.suite_storage import split_model


def package(root):
    root=Path(root)
    for side in ['ask','bid']:
        folder=root/side
        if not (folder/'evaluation.json').exists():continue
        raw=folder/'base-assessment.ubj'
        if raw.exists():
            with raw.open('rb') as src,gzip.open(str(raw)+'.gz','wb',compresslevel=6) as dest:shutil.copyfileobj(src,dest)
            raw.unlink()
        for path in list(folder.iterdir()):
            if path.suffix in {'.gz','.joblib'} and path.stat().st_size>500000:split_model(path,part_bytes=500000)
        files={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in folder.iterdir() if p.name!='release-files.json'}
        (folder/'release-files.json').write_text(json.dumps(files,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',default='models/hierarchical-original',nargs='?');package(parser.parse_args().root)
