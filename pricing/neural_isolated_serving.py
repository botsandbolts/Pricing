"""Run the *existing native* Ask and Bid checkpoints in short-lived processes.

Streamlit Community Cloud's tight RAM budget must never hold BGE, a second
DeepFM and an XGBoost booster together just to render a page. Each request
loads exactly one native engine in a subprocess, returns tiny JSON, then exits.

No pickle, ONNX conversion, quantization, retraining, Captum, or surrogate.
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path


class IsolatedModelError(RuntimeError):
    """Public-safe error, excluding detailed model paths or seller text."""


def _finite_price(value):
    try:
        p=float(value)
    except (ValueError,TypeError) as exc:
        raise IsolatedModelError("The neural engine returned an invalid price.") from exc
    if not math.isfinite(p) or not 100 <= p <= 1_000_000:
        raise IsolatedModelError("The neural engine returned an out-of-range price.")
    return p


def run(root,market,fields,description="",operation="predict",expected=None,
        timeout=105):
    """Bounded JSON IPC; the parent never imports Torch or SentenceTransformer."""
    if market not in ("ask","bid") or operation not in ("predict","sensitivity"):
        raise ValueError("Unsupported prediction request")
    if not isinstance(fields,dict):
        raise ValueError("Vehicle fields must be a dictionary")
    if len(str(description))>8000:
        raise ValueError("Advert description exceeds 8000 characters")
    if expected is not None:_finite_price(expected)
    root=Path(root).resolve()
    if not (root/"pricing"/"neural_bid_blend_live.py").is_file():
        raise IsolatedModelError("Verified model installation is incomplete.")
    data={"market":market,"fields":fields,
          "description":str(description),"operation":operation,
          "expected":expected}
    env=os.environ.copy()
    env.update({"OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2",
                "OPENBLAS_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"2",
                "TOKENIZERS_PARALLELISM":"false","MALLOC_ARENA_MAX":"2",
                "HF_HUB_DISABLE_TELEMETRY":"1"})
    try:
        proc=subprocess.run(
            [sys.executable,"-m","pricing.neural_isolated_worker"],
            cwd=str(root),env=env,input=json.dumps(data,default=str),
            capture_output=True,text=True,
            timeout=min(max(float(timeout),5.),120.),
            check=False)
    except subprocess.TimeoutExpired as exc:
        raise IsolatedModelError(
            "The model took too long. Your page is safe; try another vehicle or retry.") from exc
    except (OSError,ValueError) as exc:
        raise IsolatedModelError(
            "The isolated neural model could not start on this server.") from exc
    if proc.returncode!=0:
        if proc.returncode in (-9,137,-11):
            raise IsolatedModelError(
                "The model worker ran out of its memory allowance. "
                "The website stays open; the native model needs more capacity.")
        raise IsolatedModelError(
            "The isolated model process could not complete. "
            "Your other Car Price Studio pages are unaffected.")
    # Libraries may print to stdout. Accept ONLY our final tagged JSON line.
    lines=[line.partition("CPS_NATIVE_RESULT:")[2]
           for line in proc.stdout.splitlines()
           if line.startswith("CPS_NATIVE_RESULT:")]
    if len(lines)!=1:
        raise IsolatedModelError("The model worker did not provide a valid receipt.")
    try:
        reply=json.loads(lines[0])
    except (ValueError,TypeError) as exc:
        raise IsolatedModelError("The native model receipt was corrupted.") from exc
    if not isinstance(reply,dict) or reply.get("ok") is not True:
        raise IsolatedModelError(
            reply.get("message","The native model could not evaluate this car.")
            if isinstance(reply,dict) else "The native model request failed.")
    if reply.get("market")!=market or reply.get("operation")!=operation:
        raise IsolatedModelError("The model worker returned the wrong market.")
    if operation=="predict":
        result=reply.get("prediction",{})
        if not isinstance(result,dict) or result.get("market")!=market:
            raise IsolatedModelError("The native model receipt is incomplete.")
        _finite_price(result.get("price"))
        return result
    response=reply.get("sensitivity",{})
    if not isinstance(response,dict) or response.get("market")!=market:
        raise IsolatedModelError("The explanation receipt is incomplete.")
    _finite_price(response.get("baseline_usd"))
    for row in response.get("changes",[]):
        _finite_price(row.get("historical_price_usd"))
    if len(response.get("changes",[]))>4:
        raise IsolatedModelError("The explanation worker exceeded its evaluation budget.")
    return response
