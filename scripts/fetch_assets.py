"""Download the public model/runtime assets; no user data or training required."""
from pathlib import Path
import urllib.request,hashlib,json
root=Path(__file__).resolve().parents[1]/'dist'
expected=json.loads((Path(__file__).parent/'assets-sha256.json').read_text())
files={'encoder/onnx/model_quantized.onnx':'https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_quantized.onnx'}
for name in ['ort.wasm.min.mjs','ort-wasm-simd-threaded.wasm','ort-wasm-simd-threaded.mjs']:
 files['vendor/'+name]='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/'+name
for dest,url in files.items():
 p=root/dest
 if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==expected[dest]:continue
 p.parent.mkdir(parents=True,exist_ok=True);print('Downloading',dest,flush=True)
 tmp=p.with_suffix(p.suffix+'.partial');urllib.request.urlretrieve(url,tmp)
 if hashlib.sha256(tmp.read_bytes()).hexdigest()!=expected[dest]:raise ValueError('Asset checksum mismatch: '+dest)
 tmp.replace(p)
print('Ready. Run: python -m http.server 8501 --bind 0.0.0.0 --directory dist')
