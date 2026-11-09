"""Opt-in European EUR feedback isolated from legacy US dollar feedback."""
from __future__ import annotations
import json
import uuid
from datetime import datetime,timezone

def store_feedback(quote, score, *, explicit_consent=False):
    if explicit_consent is not True or type(score) is not int or score not in (1,2,3,4):
        return False
    from pricing.analytics import configured,_store
    if not configured():
        return False
    vehicle=quote.get("vehicle") or {}
    # Never include seller city, full description, VIN, registration number,
    # contact details, IP, analytics identifier, or browser fingerprint.
    safe={key:vehicle.get(key) for key in (
        "seller_country","make","model","reg_year","mileage_km","power_hp",
        "fuel","transmission") }
    price=float(quote["price_eur"])
    if not 100<=price<=1_000_000:
        return False
    storage=_store()
    storage.initialize()
    storage.execute("""CREATE TABLE IF NOT EXISTS european_ask_feedback (
        feedback_id TEXT PRIMARY KEY,submitted_at TEXT NOT NULL,
        rating INTEGER NOT NULL,estimated_ask_eur REAL NOT NULL,
        country TEXT,vehicle_json TEXT NOT NULL,
        explicit_consent INTEGER NOT NULL
    )""",[])
    # Retention cleanup is best-effort at every new submission; it is not a
    # guaranteed background deletion job and must not be described as one.
    try:
        storage.execute(
            "DELETE FROM european_ask_feedback WHERE submitted_at < datetime('now','-90 days')",[])
    except (OSError,RuntimeError):
        pass
    storage.execute("""INSERT INTO european_ask_feedback
        (feedback_id,submitted_at,rating,estimated_ask_eur,country,
         vehicle_json,explicit_consent) VALUES (?,?,?,?,?,?,1)""",
        [str(uuid.uuid4()),datetime.now(timezone.utc).isoformat(),
         score,price,safe.get("seller_country"),json.dumps(safe,ensure_ascii=False)])
    return True
