"""Read-only, research-only make/model ASK price differences between real months.

Never treat an insufficient two-month panel as a validated model-level index.
Never silently apply its factors to production dated car valuations.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

PATH=Path(__file__).resolve().parents[1]/"models"/"europe-ask"/"model_pulse.json"


def read_pulse(path=PATH):
    try:
        data=json.loads(Path(path).read_text(encoding="utf-8"))
        if (data.get("schema_version")!=1 or
                data.get("status")!="research_only" or
                not isinstance(data.get("research_pairs"),list)):
            raise ValueError("Invalid model-level evidence contract")
        return data
    except (OSError,ValueError,TypeError,KeyError):
        return {"schema_version":1,"status":"unavailable","research_pairs":[],
                "message":"Independent make/model monthly evidence is not packaged."}


def lookup_pulse(data,country,make,model):
    if data.get("status")!="research_only":
        return None
    county=str(country or "").strip().upper()
    brand=str(make or "").strip().lower()
    name=str(model or "").strip().lower()
    for pair in data.get("research_pairs",[]):
        if str(pair.get("country","")).upper()!=county:
            continue
        for item in pair.get("models",[]):
            if (item.get("make")==brand and item.get("model")==name):
                try:
                    effect=float(item["relative_pct"])
                    lo=float(item["interval_low_pct"])
                    hi=float(item["interval_high_pct"])
                    if (not all(map(math.isfinite,(effect,lo,hi))) or
                            lo>effect or effect>hi):
                        continue
                    return {"country":county,
                            "early_month":pair["early_month"],
                            "late_month":pair["late_month"],
                            **item,
                            "status":"research_only",
                            "applied_to_quote":False}
                except (KeyError,TypeError,ValueError):
                    continue
    return None
