"""Fetch pinned browser tokenizer/runtime assets; model weights stream on first inference."""
from pathlib import Path
import urllib.request,json,hashlib
root=Path(__file__).resolve().parents[1]/'dist';manifest=json.loads((Path(__file__).parent/'assets-sha256.json').read_text())
for dest,asset in manifest.items():
 p=root/dest
 if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==asset['sha256']:continue
 p.parent.mkdir(parents=True,exist_ok=True);print('Downloading',dest,flush=True);tmp=p.with_suffix(p.suffix+'.partial');urllib.request.urlretrieve(asset['url'],tmp)
 if hashlib.sha256(tmp.read_bytes()).hexdigest()!=asset['sha256']:raise ValueError('Checksum mismatch: '+dest)
 tmp.replace(p)
print('Ready. Run: python -m http.server 8501 --bind 0.0.0.0 --directory dist')
