"""Bounded, privacy-safe process memory telemetry for Car Price Studio.

Production: psutil RSS/VMS/USS, cgroup v1/v2 memory, oom_kill counters,
process threads and CPU, recorded every 10s and on tagged model stages.
Diagnostics go to stderr and a rotating /tmp JSONL; NEVER record vehicle
text, filenames, stack traces, IPs, VINs, credentials or user/session IDs.

Optional deep Python-only tracking: PRICING_MEMORY_TRACE_PYTHON=1 starts
tracemalloc with 1 frame. Disabled by default: tracemalloc misses native
PyTorch/ONNX allocations and itself consumes memory.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG=logging.getLogger("car_price_studio.memory")
_GUARD=threading.Lock()
_SAMPLER=None
_PROC=None
_LOG_INITIALIZED=False
_MIB=1024*1024
_INTERVAL_SEC=10
_EVENTS={"oom","oom_kill","oom_group_kill","high","max"}


def _read_int(path):
    try:
        value=Path(path).read_text().strip()
        if not value or value=="max":return None
        return int(value)
    except (OSError,ValueError,TypeError):
        return None


def _cgroup():
    """Actual container allocation; host-wide free memory is misleading."""
    current=_read_int("/sys/fs/cgroup/memory.current")
    maximum=_read_int("/sys/fs/cgroup/memory.max")
    events_path="/sys/fs/cgroup/memory.events"
    if current is None:
        current=_read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        maximum=_read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        events_path="/sys/fs/cgroup/memory/memory.failcnt"
    result={}
    if current is not None:result["container_used_mb"]=round(current/_MIB,1)
    if maximum is not None and 0<maximum<1<<50:
        result["container_limit_mb"]=round(maximum/_MIB,1)
        if current is not None:
            result["container_used_pct"]=round(100*current/maximum,1)
    try:
        for line in Path(events_path).read_text().splitlines():
            parts=line.split()
            if len(parts)==2 and parts[0] in _EVENTS:
                result["cgroup_"+parts[0]]=int(parts[1])
    except (OSError,ValueError):
        pass
    return result


def _memory_process(include_uss=False):
    global _PROC
    import psutil
    if _PROC is None:_PROC=psutil.Process(os.getpid())
    p=_PROC
    info=p.memory_info()
    data={
        "rss_mb":round(info.rss/_MIB,1),
        "vms_mb":round(info.vms/_MIB,1),
        "threads":p.num_threads(),
    }
    try:
        data["cpu_pct"]=round(p.cpu_percent(interval=None),1)
    except (psutil.Error,ValueError):
        pass
    if include_uss:
        try:
            if hasattr(p,"memory_footprint"):
                full=p.memory_footprint()
            else:full=p.memory_full_info()
            for key in ("uss","pss","swap"):
                value=getattr(full,key,None)
                if value is not None:data[key+"_mb"]=round(value/_MIB,1)
        except (psutil.Error,OSError,ValueError):
            pass
    return data


def snapshot(stage,*,elapsed_ms=None,detailed=False):
    """Return structured metrics only. Stage names are controlled by source."""
    data={
        "component":"memory","schema":1,
        "stage":str(stage)[:72],
        "unix_s":round(time.time(),3),
        "pid":os.getpid(),
    }
    if elapsed_ms is not None:
        data["elapsed_ms"]=round(float(elapsed_ms),1)
    try:
        data.update(_memory_process(include_uss=detailed))
    except (ImportError,OSError,RuntimeError):
        data["process_stats_unavailable"]=True
    data.update(_cgroup())
    if os.environ.get("PRICING_MEMORY_TRACE_PYTHON")=="1":
        import tracemalloc
        if tracemalloc.is_tracing():
            current,peak=tracemalloc.get_traced_memory()
            data.update(python_alloc_mb=round(current/_MIB,1),
                        python_alloc_peak_mb=round(peak/_MIB,1))
    return data


def _configure():
    global _LOG_INITIALIZED
    if _LOG_INITIALIZED:return
    # Root stderr handler is managed by Streamlit; add a small bounded file
    # for process-level debugging. No customer inputs or file paths emitted.
    try:
        log_path=Path("/tmp/car_price_studio_memory.jsonl")
        handler=RotatingFileHandler(log_path,maxBytes=2*1024*1024,
                                    backupCount=2,encoding="utf-8")
        os.chmod(log_path,0o600)
        handler.setFormatter(logging.Formatter("%(message)s"))
        LOG.addHandler(handler)
    except (OSError,PermissionError):
        pass
    LOG.setLevel(logging.WARNING)
    _LOG_INITIALIZED=True


def record(stage,*,elapsed_ms=None,detailed=False):
    try:
        datum=snapshot(stage,elapsed_ms=elapsed_ms,detailed=detailed)
        pressure=datum.get("container_used_pct",0)
        datum["pressure_level"]=("critical" if pressure>=90 else
                                 "high" if pressure>=80 else "normal")
        LOG.warning(json.dumps(datum,separators=(",",":"),sort_keys=True))
        return datum
    except Exception:
        # Diagnostics must never interrupt a valuation or explanation.
        return {"component":"memory","stage":"metrics_unavailable"}


def _sampling_loop():
    cycle=0
    while True:
        time.sleep(_INTERVAL_SEC)
        cycle+=1
        record("periodic",detailed=cycle%6==0)


def ensure_monitor():
    """Start one daemon PER serving process; Streamlit reruns reuse it."""
    global _SAMPLER
    if os.environ.get("PRICING_MEMORY_MONITOR","1")=="0":return False
    with _GUARD:
        if _SAMPLER and _SAMPLER.is_alive():return True
        _configure()
        if os.environ.get("PRICING_MEMORY_TRACE_PYTHON")=="1":
            import tracemalloc
            if not tracemalloc.is_tracing():
                tracemalloc.start(1)
        record("streamlit_boot",detailed=True)
        _SAMPLER=threading.Thread(target=_sampling_loop,
             name="cps-memory-watch",daemon=True)
        _SAMPLER.start()
    return True
