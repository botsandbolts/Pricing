# Car Price Studio · Memory observability and smaller BGE inference

## Production diagnostics (privacy-safe)

We now use [psutil](https://psutil.io/api/) on the Streamlit server. A single daemon thread takes a cheap process sample **every 10 seconds**, and additional snapshots are recorded at each step of the real Ask/Bid valuation and the 24-step Captum explanation.

All events are JSON dictionaries with **process and container information only**:

| Data | Frequency and purpose |
|---|---|
| Resident set size (RSS), virtual size (VMS) | 10 seconds, plus each valuation/Captum stage |
| CPU percentage, thread count | 10 seconds, exposes runaway thread pools |
| Linux container `memory.current`, `memory.max` | 10 seconds, accounts for actual hosting limits |
| cgroup `oom`, `oom_kill`, `high`, `max` | Every sample, indicates whether kernel killed the process |
| Unique/proportional set size (USS/PSS) | 60 seconds, and selected model/Captum events; more expensive to measure |
| Actual model phase and elapsed milliseconds | On Submit, BGE load, BGE encode, Ask forward, Bid load/forward, Captum start/finish/error |
| Python allocator bytes | **Only** when `PRICING_MEMORY_TRACE_PYTHON=1`; optional standard-library `tracemalloc` |

To read the logs: **Streamlit Cloud → Manage app → Logs** and filter for `car_price_studio.memory`, `"component":"memory"`, `valuation stage=`, or `captum_24step_complete`. There is a bounded secondary file at `/tmp/car_price_studio_memory.jsonl` (2 MB plus two rotating backups, private file permissions); this is ephemeral and is **not a durable analytics database**.

The watcher intentionally **never logs seller descriptions, VINs, vehicle inputs, session identifiers, client IPs, passwords, environment variables, stack traces, or Hugging Face tokens**. Uncaught tracebacks continue to go to the separate internal server error logger.

A kernel OOM kill or a host reboot can terminate Python before its final callback. The periodic JSON records and cgroup OOM counters are therefore more useful than relying solely on exceptions.

### Why not run a full native heap profiler on Streamlit Cloud?

Continuous full allocation tracing creates more allocations, CPU overhead and large recordings. We install [Memray](https://bloomberg.github.io/memray/) **only in the CI benchmark**; it follows native PyTorch/ONNX C/C++ allocations and produces an artifact recording. `tracemalloc` is useful for Python allocation leaks but does not explain native PyTorch memory by itself. Both deeper modes are off in production by default.

## Candidate quantized encoder

The untouched historical Ask and Bid pricing checkpoints use PyTorch and must continue to support Captum. Most of the apparent overhead comes from the frozen **BAAI/bge-small-en-v1.5** pretrained text encoder, not the small pricing heads. Its 133 MB original ONNX graph can be weight-quantized to INT8 offline.

The optional alternative is a **single real ONNX Runtime model**, generated from the exact pinned encoder revision. We do not run a second encoder or an API call on each car. The original 384-dim L2-normalized BGE vector is preserved as the feature consumed by the verified ResMLP.

The [release workflow](../.github/workflows/bge-int8-release.yml) must successfully complete:

1. Weight size reduction after dynamic MatMul/Gemm INT8 conversion, with no model retraining and no price labels.
2. Parity on **48 deterministic, real, price-redacted historical car descriptions**. The source original BGE and quantized BGE vectors are fed into the **same unchanged trained Ask pricing head**.
3. Guardrails for price drift, not just cosine similarity; a smaller encoder is not acceptable if it changes dollar prices too much.
4. Two **fresh separate benchmark processes in CI** running the **same original Ask/Bid price heads**: an original encoder run and an INT8 encoder run. The CI benchmark is allowed to use processes for fair measurement; Streamlit production inference does not.
5. At least **40 MB / 5% less peak RSS** across both markets, based on the measured process peak.
6. On success, and only on success, the CI runner commits the ~35 MB INT8 ONNX artifact, tokenizer and checksummed release manifest. Only then does the public application select `auto` INT8. Before that it retains the original pinned BGE PyTorch implementation.

This is **serving quantization**, not a new trained pricing model. Any changes to errors are differences relative to the original predictions, not a measured improvement on independent transaction labels.

### Controls

- `PRICING_BGE_BACKEND=auto` (default): only use an artifact with the verified release manifest.
- `PRICING_BGE_BACKEND=native`: explicitly force the original BGE if the INT8 serving environment causes problems.
- `PRICING_BGE_BACKEND=int8`: require the verified INT8 artifact or fail visibly (no unsupported substitute).
- `PRICING_MEMORY_MONITOR=0`: disable the periodic sampler (not recommended during diagnosis).
- `PRICING_MEMORY_TRACE_PYTHON=1`: enable detailed Python allocation tracking for temporary debugging, at increased memory overhead.

Neither quantization nor monitoring changes the historical CVaR95 winner checkpoints, ensemble weights, or the 24-step Captum calculation.
