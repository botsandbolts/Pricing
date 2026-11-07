"""Offline parity diagnosis: original ST vs original ONNX vs INT8 ONNX.

No user data, only deterministic public price-redacted historical examples.
This script is never imported by Streamlit.
"""
import json
import os
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/"reports"/"qa"/"bge-int8-release"


def inspect():
    from huggingface_hub import hf_hub_download
    import onnxruntime as ort
    from tokenizers import Tokenizer
    from pricing.quantized_bge_runtime import SOURCE,REVISION,WEIGHTS,TOKENIZER
    from pricing.neural_ask_embedding_live import BGEDeepCrossAsk
    from scripts.common import clean
    from scripts.release_bge_int8 import PACKAGE,examples
    import torch
    torch.set_num_threads(2)
    REPORT.mkdir(parents=True,exist_ok=True)
    os.environ["PRICING_BGE_BACKEND"]="native"
    original=BGEDeepCrossAsk(ROOT)
    unquant=hf_hub_download(repo_id=SOURCE,revision=REVISION,
                            filename="onnx/model.onnx")
    settings=ort.SessionOptions()
    settings.intra_op_num_threads=2
    settings.inter_op_num_threads=1
    settings.enable_cpu_mem_arena=False
    opts={"providers":["CPUExecutionProvider"],"sess_options":settings}
    fp32=ort.InferenceSession(unquant,**opts)
    int8=ort.InferenceSession(str(PACKAGE/WEIGHTS),**opts)
    tk=Tokenizer.from_file(str(PACKAGE/TOKENIZER))
    tk.enable_truncation(max_length=256)
    reports=[]
    for fields,description in examples()[:10]:
        text=clean(description)
        expected=original.prepare_text(description)["embedding"][0]
        st_token=original.encoder.tokenize([text])
        encoding=tk.encode(text)
        native_ids=st_token["input_ids"][0].detach().cpu().numpy().tolist()
        raw_tokens=encoding.ids
        tokens_match=(native_ids==raw_tokens)
        same_length=len(native_ids)==len(raw_tokens)
        if not same_length:
            common=sum(a==b for a,b in zip(native_ids,raw_tokens))
        else:
            common=sum(a==b for a,b in zip(native_ids,raw_tokens))
        result={"native_token_count":len(native_ids),
                "onnx_token_count":len(raw_tokens),
                "token_ids_equal":tokens_match,
                "token_id_matches_at_start":common,
                "first_native_ids":native_ids[:12],
                "first_onnx_ids":raw_tokens[:12]}
        payload={
            "input_ids":np.asarray([encoding.ids],dtype=np.int64),
            "attention_mask":np.asarray([encoding.attention_mask],dtype=np.int64),
            "token_type_ids":np.asarray([encoding.type_ids],dtype=np.int64)
        }
        for name,sess in (("fp32",fp32),("int8",int8)):
            input_names={q.name for q in sess.get_inputs()}
            prediction=sess.run(None,{k:v for k,v in payload.items()
                                      if k in input_names})
            candidates=[np.asarray(x) for x in prediction if
                        getattr(x,"ndim",0)==3 and x.shape[0]==1 and x.shape[-1]==384]
            if not candidates:raise ValueError("No 384D token sequence output for "+name)
            hs=candidates[0][0]
            for mode in ("cls","mean_masked"):
                if mode=="cls":
                    vector=hs[0]
                else:
                    mask=np.asarray(encoding.attention_mask,dtype=np.float32)
                    vector=np.sum(hs*mask.reshape(-1,1),axis=0)/max(mask.sum(),1)
                vector=np.asarray(vector,dtype=np.float32)
                vector/=max(float(np.linalg.norm(vector)),1e-12)
                cosine=float(np.dot(vector,expected)/
                    max(float(np.linalg.norm(expected)),1e-12))
                result[name+"_"+mode+"_cosine"]=cosine
            result[name+"_output_names"]=[x.name for x in sess.get_outputs()]
        reports.append(result)
    metrics={
        "examples":len(reports),
        "native_tokenizer_matches_all":all(r["token_ids_equal"] for r in reports),
        "native_tokenizer_matches_n":sum(r["token_ids_equal"] for r in reports),
        "fp32_cls_cos_min":min(r["fp32_cls_cosine"] for r in reports),
        "fp32_cls_cos_median":float(np.median([r["fp32_cls_cosine"] for r in reports])),
        "fp32_mean_cos_median":float(np.median([r["fp32_mean_masked_cosine"] for r in reports])),
        "int8_cls_cos_min":min(r["int8_cls_cosine"] for r in reports),
        "int8_cls_cos_median":float(np.median([r["int8_cls_cosine"] for r in reports])),
        "int8_mean_cos_median":float(np.median([r["int8_mean_masked_cosine"] for r in reports])),
        "records":reports[:3],
    }
    (REPORT/"backend-diagnostics.json").write_text(json.dumps(metrics,indent=2))
    print(json.dumps({"stage":"encoder_backend_diagnostics",**{
        k:v for k,v in metrics.items() if k!="records"}}),flush=True)
    if metrics["fp32_cls_cos_min"]<.995:
        raise ValueError("Original ONNX graph is not encoder-parity compatible; quantization is unsafe")
    if metrics["int8_cls_cos_min"]<.990:
        raise ValueError("INT8 ONNX output differs too much from original FP32 encoder")


if __name__=="__main__":
    inspect()
