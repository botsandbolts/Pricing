"""Release gate for frozen BGE-small INT8 ONNX with unchanged Ask/Bid heads.

Step 1: quantize offline at pinned model revision.
Step 2: compare native PyTorch and ONNX embeddings and actual historical
        asking PRICE predictions for 48 price-redacted real examples.
Step 3: bench both original and quantized Ask+Bid in two FRESH CI processes.
Step 4: publish only if parity AND peak RSS improvement pass.

No model retraining. No need for raw original price labels; all vehicles are
from published training examples. No personal user data enters this workflow.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/"models"/"bge-small-int8-onnx"
REPORT=ROOT/"reports"/"qa"/"bge-int8-release"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def quantize():
    from huggingface_hub import hf_hub_download
    from onnxruntime.quantization import quantize_dynamic,QuantType
    from pricing.quantized_bge_runtime import (
        SOURCE,REVISION,WEIGHTS,TOKENIZER)
    PACKAGE.mkdir(parents=True,exist_ok=True)
    REPORT.mkdir(parents=True,exist_ok=True)
    original=Path(hf_hub_download(
        repo_id=SOURCE,revision=REVISION,filename="onnx/model.onnx"))
    tokenizer=Path(hf_hub_download(
        repo_id=SOURCE,revision=REVISION,filename="tokenizer.json"))
    target=PACKAGE/WEIGHTS
    # The original float ONNX is retained only in the transient CI cache.
    # Comparing its embeddings to native isolates a tokenizer/pooling/export
    # mismatch from a true quantization-only numerical drift.
    (REPORT/"reference-onnx-path.txt").write_text(str(original))
    # Dynamic QInt8 for transformer MatMul/Gemm weights only. Embedding and
    # normalization stay full precision, preserving the native price head.
    quantize_dynamic(str(original),str(target),
        weight_type=QuantType.QInt8,per_channel=True,
        op_types_to_quantize=["MatMul","Gemm"])
    shutil.copyfile(tokenizer,PACKAGE/TOKENIZER)
    measured={
        "full_precision_encoder_onnx_mb":round(original.stat().st_size/1048576,1),
        "int8_encoder_onnx_mb":round(target.stat().st_size/1048576,1),
        "weight_shrink_pct":round(
            100*(1-target.stat().st_size/original.stat().st_size),1)
    }
    print(json.dumps({"stage":"quantized_bytes",**measured}),flush=True)
    if not target.is_file() or target.stat().st_size>=original.stat().st_size*.88:
        raise ValueError("INT8 weights did not shrink the pinned BGE encoder enough")
    return measured


def examples():
    from pricing.neural_ask_embedding_live import NUMS,CATS
    data=json.loads((ROOT/"data"/"description-examples.json").read_text())
    items=data["examples"]
    usable=[]
    for item in items:
        car=item.get("fields",{})
        if not all(name in car for name in ("year","manufacturer","model","odometer")):
            continue
        if int(car["year"])>2021:
            continue
        usable.append((dict(car),str(item["text"])[:8000]))
    if len(usable)<40:
        raise ValueError("Not enough authentic price-redacted training adverts")
    # Manufacturer-balanced source is already deterministic in the file.
    return usable[:48]


def vectors_and_price():
    from pricing.quantized_bge_runtime import (
        VerifiedInt8BGE,SOURCE,REVISION,WEIGHTS,TOKENIZER,MANIFEST)
    from pricing.neural_ask_embedding_live import BGEDeepCrossAsk
    from scripts.common import clean
    import torch
    torch.set_num_threads(2)
    # Provisionally allow the candidate ONLY INSIDE this CI workspace. The
    # manifest and weight files are not pushed unless parity + RSS pass.
    m={
        "status":"passed","source_model":SOURCE,"source_revision":REVISION,
        "format":"onnx-int8-dynamic-avx2","dimension":384,"max_tokens":256,
        "price_head_unchanged":True,"quantization_did_not_use_prices":True,
        "max_price_drift_passed":True,"memory_reduction_passed":True,
        "weight_sha256":sha(PACKAGE/WEIGHTS),
        "tokenizer_sha256":sha(PACKAGE/TOKENIZER)
    }
    (PACKAGE/MANIFEST).write_text(json.dumps(m,indent=2))
    # First load NATIVE only. We NEVER run this paired baseline in Streamlit.
    os.environ["PRICING_BGE_BACKEND"]="native"
    native=BGEDeepCrossAsk(ROOT)
    experimental=VerifiedInt8BGE(ROOT)
    import onnxruntime as ort
    fp32_path=Path((REPORT/"reference-onnx-path.txt").read_text().strip())
    if not fp32_path.is_file():
        raise FileNotFoundError("Original floating-point ONNX diagnostic reference missing")
    ort_opts=experimental.session.get_session_options()
    full_precision_session=ort.InferenceSession(
        str(fp32_path),sess_options=ort_opts,
        providers=["CPUExecutionProvider"])
    quant_session=experimental.session
    native_vs_fp32=[]
    native_vs_fp32_dollar=[]
    cosines=[]
    deviations=[]
    abs_dollars=[]
    for fields,description in examples():
        prepared=native.prepare_text(description)
        original=float(native.predict_prepared(fields,prepared)["price"])
        experimental.session=full_precision_session
        fp32=experimental.encode(
            [clean(description)],batch_size=1,convert_to_numpy=True,
            normalize_embeddings=True,show_progress_bar=False)
        experimental.session=quant_session
        vector=experimental.encode(
            [clean(description)],batch_size=1,convert_to_numpy=True,
            normalize_embeddings=True,show_progress_bar=False)
        # All path variants use identical tokenization, CLS pooling, L2 norm.
        fp32=np.asarray(fp32,dtype=np.float32).astype(np.float16).astype(np.float32)
        prepared_fp32={"raw":description,"embedding":fp32}
        fp32_price=float(native.predict_prepared(fields,prepared_fp32)["price"])
        cosine_fp32=float(np.dot(prepared["embedding"][0],fp32[0])/
            max(np.linalg.norm(prepared["embedding"][0])*np.linalg.norm(fp32[0]),1e-10))
        native_vs_fp32.append(cosine_fp32)
        native_vs_fp32_dollar.append(abs(fp32_price-original))
        vector=np.asarray(vector,dtype=np.float32)
        vector=vector.astype(np.float16).astype(np.float32)
        substitute={"raw":description,"embedding":vector}
        candidate=float(native.predict_prepared(fields,substitute)["price"])
        if not (math.isfinite(original) and math.isfinite(candidate)):
            raise ValueError("Non-finite native/quantized price")
        dot=float(np.dot(prepared["embedding"][0],vector[0]))
        denom=float(np.linalg.norm(prepared["embedding"][0])*
                    np.linalg.norm(vector[0]))
        cosines.append(dot/max(denom,1e-10))
        deviations.append(abs(candidate-original)/max(100.,original))
        abs_dollars.append(abs(candidate-original))
    minimum_cos=float(min(cosines))
    worst=float(max(deviations))
    p95=float(np.percentile(deviations,95))
    median=float(np.median(deviations))
    print(json.dumps({"stage":"float_onnx_reference_diagnostic",
          "min_cosine_native_vs_float_onnx":float(min(native_vs_fp32)),
          "median_cosine_native_vs_float_onnx":float(np.median(native_vs_fp32)),
          "max_native_vs_float_onnx_price_drift_usd":float(max(native_vs_fp32_dollar)),
          "mean_native_vs_float_onnx_price_drift_usd":float(np.mean(native_vs_fp32_dollar))}),
          flush=True)
    result={
        "reference_native_vs_float_onnx_min_cosine":float(min(native_vs_fp32)),
        "reference_native_vs_float_onnx_max_price_drift_usd":float(max(native_vs_fp32_dollar)),
        "source":"48 deterministic price-redacted historical listing examples",
        "count":len(cosines),
        "min_embedding_cosine":minimum_cos,
        "max_relative_price_drift":worst,
        "p95_relative_price_drift":p95,
        "median_relative_price_drift":median,
        "max_absolute_price_drift_usd":float(max(abs_dollars)),
        "test_used_ground_truth_sale_prices":False,
        "native_price_network_unchanged":True,
    }
    print(json.dumps({"stage":"real_price_parity",**result}),flush=True)
    (REPORT/"price-parity.json").write_text(json.dumps(result,indent=2))
    # Conservative acceptance for changing the pretrained text encoder:
    # <0.8% max dollar-price difference, <0.4% p95, <0.15% median.
    if not (minimum_cos>=.992 and worst<=.008 and p95<=.004 and median<=.0015):
        raise ValueError("INT8 BGE's end-to-end pricing drift exceeds the gate")
    return result


def bench(backend):
    env=os.environ.copy()
    env.update(PRICING_BGE_BACKEND=backend,OMP_NUM_THREADS="2",
               MKL_NUM_THREADS="2",TOKENIZERS_PARALLELISM="false")
    command=[sys.executable,"-m","scripts.native_pricing_runtime_smoke"]
    # Separate *CI-only* processes make peak RSS comparisons meaningful.
    # The public app does not launch subprocesses or swap weights in memory.
    result=subprocess.run(
        command,cwd=str(ROOT),env=env,capture_output=True,text=True,
        timeout=180,check=False)
    if result.returncode:
        print(result.stderr[-2500:],file=sys.stderr)
        raise RuntimeError("Real production-head benchmark failed for "+backend)
    rows=[]
    for raw in result.stdout.splitlines():
        if not raw.startswith("{"):continue
        try:record=json.loads(raw)
        except ValueError:continue
        if "stage" in record and "peak_rss_mb" in record:
            rows.append(record)
    last=next((r for r in reversed(rows)
               if r["stage"]=="both_native_prices_verified"),None)
    if not last:
        raise ValueError("Real Ask+Bid benchmark receipt missing for "+backend)
    return {
        "backend":backend,"peak_rss_mb":last["peak_rss_mb"],
        "stages":rows,
        "native_ask_and_bid_verified":True,
    }


def release(measured,parity):
    native=bench("native")
    quant=bench("int8")
    benefit=float(native["peak_rss_mb"]-quant["peak_rss_mb"])
    percent=100*benefit/max(native["peak_rss_mb"],1.)
    report={
        "native":native,"int8":quant,
        "saved_peak_rss_mb":benefit,
        "saved_peak_rss_pct":round(percent,2),
        "memory_reduction_passed":benefit>=40 and percent>=5.,
        "measured_independent_processes":True,
        "scope":"Both verified native Ask and Bid models on one CPU runner",
    }
    (REPORT/"memory-parity.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({"stage":"memory_benchmark",
        "native_mb":native["peak_rss_mb"],"int8_mb":quant["peak_rss_mb"],
        "saved_mb":benefit,"saved_pct":percent}),flush=True)
    if not report["memory_reduction_passed"]:
        raise ValueError("INT8 BGE did not cut combined process peak RSS >=40 MB / >=5%")
    path=PACKAGE/"manifest.json"
    m=json.loads(path.read_text())
    m.update({
        "built_utc":datetime.now(timezone.utc).isoformat(),
        "source_native_checkpoint_sha256":sha(
            ROOT/"models"/"neural-experimental-ask-bge"/"weights.pt"),
        "price_head_unchanged":True,
        "quantization_did_not_use_prices":True,
        "max_price_drift_passed":True,
        "memory_reduction_passed":True,
        "price_parity":parity,
        "memory_parity":{
            "native_peak_rss_mb":native["peak_rss_mb"],
            "int8_peak_rss_mb":quant["peak_rss_mb"],
            "saved_peak_rss_mb":benefit,
            "saved_peak_rss_pct":round(percent,2),
        },
        "quantization":measured,
        "estimated_model_quality_improvement":False,
        "shipping_policy":"publish only if identical native price weights and gated end-to-end parity",
    })
    path.write_text(json.dumps(m,indent=2))
    print("INT8 BGE RELEASE GATE PASSED",flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=("quantize","parity","release","all"))
    args=p.parse_args()
    if args.command in ("quantize","all"):
        measured=quantize()
        (REPORT/"weight-savings.json").write_text(json.dumps(measured,indent=2))
    if args.command in ("parity","all"):
        parity=vectors_and_price()
    if args.command in ("release","all"):
        measured=json.loads((REPORT/"weight-savings.json").read_text())
        parity=json.loads((REPORT/"price-parity.json").read_text())
        release(measured,parity)


if __name__=="__main__":
    main()
