"""Download pinned public encoder assets and the checkpoints used for verification."""
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
repos={'embaas':('embaas/sentence-transformers-multilingual-e5-large','f5ce37891bfe42176585f3e74b256456ca1119a3'),'reference':('intfloat/multilingual-e5-large','3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3'),'onnx':('Xenova/multilingual-e5-large','00fc3aeb3dbb95842de2ac1961d33c6319acf57b')}
jobs=[]
for name in ['embaas','onnx']:
 for file in ['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json']:jobs.append((name,file))
for name in ['embaas','onnx']:jobs.append((name,'sentencepiece.bpe.model'))
for name in ['embaas','reference']:jobs.append((name,'pytorch_model.bin'))
for name in ['onnx']:jobs.append((name,'onnx/model_quantized.onnx'))
for f in ['modules.json','sentence_bert_config.json','1_Pooling/config.json']:jobs.append(('embaas',f))
def get(job):
 name,file=job;repo,revision=repos[name];return hf_hub_download(repo,file,revision=revision,local_dir=str(root/name))
with ThreadPoolExecutor(max_workers=4) as pool:
 for path in pool.map(get,jobs):print(path,flush=True)
