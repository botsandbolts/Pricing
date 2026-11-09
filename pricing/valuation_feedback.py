"""Database schema for optional accuracy ratings.

Visitors must opt into saving each rating explicitly. Save only the chosen
1–4 score and allowlisted vehicle details and the two displayed dollar prices,
never seller text, VIN, address, IP, or personal identifiers.
"""
SCHEMA = [
    """CREATE TABLE IF NOT EXISTS valuation_feedback (
      feedback_id TEXT PRIMARY KEY,
      request_id TEXT NOT NULL,
      submitted_at TEXT NOT NULL,
      model_key TEXT NOT NULL CHECK(model_key IN ('bge', 'granite')),
      score INTEGER NOT NULL CHECK(score BETWEEN 1 AND 4),
      ask_price_usd REAL NOT NULL,
      bid_price_usd REAL,
      valuation_date TEXT,
      vehicle_json TEXT NOT NULL,
      explicit_consent INTEGER NOT NULL CHECK(explicit_consent = 1)
    )""",
    "CREATE UNIQUE INDEX IF NOT EXISTS feedback_request_model ON valuation_feedback(request_id,model_key)",
    "CREATE INDEX IF NOT EXISTS valuation_feedback_timestamp ON valuation_feedback(submitted_at)",
]
