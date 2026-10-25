"""Fetch public runtime/tokenizer assets. --training also fetches the large XLM-R encoder."""
from pathlib import Path
import urllib.request,hashlib,json,sys
root=Path(__file__).resolve().parents[1]/'dist';expected=json.loads((Path(__file__).parent/'assets-sha256.json').read_text())
revision='e5d116277351513fd260955ece953ecddde7046e'
base='https://huggingface.co/Xenova/paraphrase-multilingual-mpnet-base-v2/resolve/'+revision+'/'
files={}
for name in ['ort.wasm.min.mjs','ort-wasm-simd-threaded.wasm','ort-wasm-simd-threaded.mjs']:
 files['vendor/'+name]='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/'+name
files['vendor/tokenizers.LICENSE']='https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/LICENSE'
files['vendor/tokenizers.mjs']='https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/dist/tokenizers.mjs'
for name in ['tokenizer.json','tokenizer_config.json','config.json','special_tokens_map.json','sentencepiece.bpe.model']:
 files['encoder/'+name]=base+name
if '--training' in sys.argv:files['encoder/onnx/model_quantized.onnx']=base+'onnx/model_quantized.onnx'
for dest,url in files.items():
 p=root/dest
 if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==expected[dest]:continue
 p.parent.mkdir(parents=True,exist_ok=True);print('Downloading',dest,flush=True)
 tmp=p.with_suffix(p.suffix+'.partial');urllib.request.urlretrieve(url,tmp)
 if hashlib.sha256(tmp.read_bytes()).hexdigest()!=expected[dest]:raise ValueError('Asset checksum mismatch: '+dest)
 tmp.replace(p)
print('Ready. Run: python -m http.server 8501 --bind 0.0.0.0 --directory dist')
