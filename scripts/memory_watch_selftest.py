"""No-network sanity check for low-overhead production memory telemetry."""
import json
import os
import time
from pricing import memory_watch


def main():
    before=memory_watch.snapshot("selftest_before",detailed=True)
    assert before["stage"]=="selftest_before"
    assert before["rss_mb"]>0
    assert before["vms_mb"]>0
    assert before["threads"]>=1
    assert before.get("container_used_mb",0)>=0
    assert all(key not in before for key in (
        "vehicle_description","vin","ip_address","user_id","email",
        "input_text","token","secret"))
    assert memory_watch.ensure_monitor() is True
    assert memory_watch.ensure_monitor() is True
    assert memory_watch._SAMPLER.is_alive()
    afterwards=memory_watch.record("selftest_after",elapsed_ms=42.,detailed=True)
    assert afterwards["stage"]=="selftest_after"
    assert afterwards["elapsed_ms"]==42.
    assert afterwards["pressure_level"] in ("normal","high","critical")
    assert memory_watch._SAMPLER.name=="cps-memory-watch"
    assert not os.environ.get("PRICING_MEMORY_TRACE_PYTHON")
    print(json.dumps({"test":"pass","rss_mb":afterwards["rss_mb"],
                      "container_limit_mb":afterwards.get("container_limit_mb"),
                      "log_rotation":"2 MB + 2 backups",
                      "cadence_seconds":10,
                      "sensitive_input_logged":False}),flush=True)


if __name__=="__main__":
    main()
