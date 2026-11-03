"""Export the pinned summarization checkpoint, quantize offline, save small graphs."""
import hashlib,json,shutil
from pathlib import Path
from optimum.exporters.onnx import main_export
from onnxruntime.quantization import quantize_dynamic,QuantType
from transformers import AutoTokenizer
from pricing.bart_summary import MODEL_ID,REVISION
out=Path('models/bart-cpu');out.mkdir(parents=True,exist_ok=True)
raw=Path('training-cache/bart-onnx')
main_export(MODEL_ID,output=raw,task='text2text-generation-with-past',
            revision=REVISION,device='cpu',dtype='fp32',batch_size=1,sequence_length=32)
AutoTokenizer.from_pretrained(MODEL_ID,revision=REVISION).save_pretrained(out)
shutil.copy2(raw/'config.json',out/'config.json')
for name in ['encoder_model.onnx','decoder_model_merged.onnx']:
    if not (raw/name).exists():raise FileNotFoundError(name)
    quantize_dynamic(str(raw/name),str(out/name),weight_type=QuantType.QInt8,
                     per_channel=False,op_types_to_quantize=['MatMul','Gemm','Gather'],extra_options={'EnableSubgraph':True})
    print('QUANTIZED',name,(out/name).stat().st_size,flush=True)
manifest={'model':MODEL_ID,'revision':REVISION,'dtype':'dynamic-int8','runtime':'ONNX Runtime CPU; no PyTorch import','graphs':{}}
from pricing.suite_storage import split_model
for name in ['encoder_model.onnx','decoder_model_merged.onnx']:
    path=out/name
    manifest['graphs'][name]={'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),'bytes':path.stat().st_size}
    if path.stat().st_size>8*1024*1024:split_model(path,part_bytes=8*1024*1024)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('BART INT8 exported',flush=True)
