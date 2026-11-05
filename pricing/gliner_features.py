"""Normalize extracted seller claims without treating silence as denial."""
import json
import re
import sqlite3
from pathlib import Path


FEATURE_GROUPS = {
    "maintenance": ("brake_service", "tire_service", "timing_belt_service"),
    "faults": ("engine_fault", "transmission_fault", "air_conditioning_fault"),
    "history": ("accident_history", "salvage_title"),
    "equipment": ("heated_seats", "backup_camera", "sunroof"),
}
MONEY = re.compile(r"[$€£]|\b(?:usd|dollars?|asking price)\b", re.I)
NEGATED = re.compile(r"\b(?:no|never|without|free of)\b", re.I)
UNCERTAIN = re.compile(r"\b(?:maybe|possibly|unknown|suspected|might)\b", re.I)


def records(extraction, section):
    values = extraction.get(section, [])
    if isinstance(values, dict):
        values = [values]
    return [item for item in values if isinstance(item, dict)] if isinstance(values, list) else []


def phrases(value):
    values = value if isinstance(value, list) else [value]
    for item in values:
        if isinstance(item, dict):
            item = item.get("text")
        if isinstance(item, str) and item.strip() and not MONEY.search(item):
            yield item.strip()


def normalize(extraction, description):
    """Return numeric values (None is missing) and source evidence."""
    claims = {}

    def add(name, evidence, value):
        if evidence.casefold() in description.casefold() and not UNCERTAIN.search(evidence):
            claims.setdefault(name, []).append((value, evidence))

    for vehicle in records(extraction, "vehicle"):
        for phrase in phrases(vehicle.get("accident_history")):
            if re.search(r"\b(?:no accidents?|accident[- ]free|never (?:been )?in an accident)\b", phrase, re.I):
                add("accident_history", phrase, 0)
            elif re.search(r"\b(?:had an accident|was in an accident|accident damage|previous accident)\b", phrase, re.I) and not NEGATED.search(phrase):
                add("accident_history", phrase, 1)
        for phrase in phrases(vehicle.get("title_status")):
            if re.fullmatch(r"clean title", phrase, re.I):
                add("salvage_title", phrase, 0)
            elif re.fullmatch(r"(?:salvage|rebuilt) title", phrase, re.I):
                add("salvage_title", phrase, 1)
        for phrase in phrases(vehicle.get("equipment")):
            for name in FEATURE_GROUPS["equipment"]:
                if re.fullmatch(name.replace("_", " "), phrase, re.I):
                    # Require the source clause, since the extracted span can omit a negation.
                    for match in re.finditer(r"[^.!?\n]*\b" + re.escape(phrase) + r"\b[^.!?\n]*", description, re.I):
                        clause = match.group().strip()
                        if not MONEY.search(clause) and not re.search(r"\b(?:broken|needs|missing|doesn.t work)\b", clause, re.I):
                            add(name, clause, int(not NEGATED.search(clause)))

    for record in records(extraction, "repair"):
        for phrase in phrases(record.get("evidence")):
            if NEGATED.search(phrase) or re.search(r"\b(?:needs?|due|recommend)\b", phrase, re.I):
                continue
            if re.search(r"\b(?:new|replaced|changed|installed|serviced)\b", phrase, re.I):
                for pattern, name in [(r"brakes?|brake pads", "brake_service"),
                                      (r"tires?|tyres?", "tire_service"),
                                      (r"timing belt", "timing_belt_service")]:
                    if re.search(r"\b(?:" + pattern + r")\b", phrase, re.I):
                        add(name, phrase, 1)
    for record in records(extraction, "fault"):
        for phrase in phrases(record.get("evidence")):
            if re.search(r"\b(?:fixed|resolved|repaired|replaced)\b", phrase, re.I):
                continue
            for pattern, name in [(r"engine", "engine_fault"), (r"transmission", "transmission_fault"),
                                  (r"air conditioning|a/c|ac", "air_conditioning_fault")]:
                if not re.search(r"\b(?:" + pattern + r")\b", phrase, re.I):
                    continue
                if re.search(r"\bno\b.*\b(?:issues?|faults?|problems?)\b", phrase, re.I):
                    add(name, phrase, 0)
                elif not NEGATED.search(phrase) and re.search(r"\b(?:fault|broken|slipping|knocking|doesn.t work|not working|needs repair)\b", phrase, re.I):
                    add(name, phrase, 1)

    result = {}
    for names in FEATURE_GROUPS.values():
        for name in names:
            observations = claims.get(name, [])
            values = {value for value, _ in observations}
            result[name] = {
                "value": next(iter(values)) if len(values) == 1 else None,
                "evidence": list(dict.fromkeys(phrase for _, phrase in observations)),
            }
    return result


def iter_database(path):
    """Yield listing IDs and the shared normalized records from the GLiNER database."""
    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True) as db:
        for row_id, description, payload in db.execute(
            "SELECT row_id, description, extraction_json FROM descriptions"
        ):
            yield row_id, normalize(json.loads(payload)["extraction"], description)
