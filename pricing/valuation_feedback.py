"""Consent-only, privacy-minimal subjective valuation quality feedback.

A rating is an opinion, not evidence that the model's true dollar error equals
that rating. One rating per quote and market (Ask or Bid), with a stable quote ID.
The same price, chosen model, market date and allowlisted vehicle features are
stored in pricing_queries for diagnosis; NEVER raw adverts, VINs or IPs.
"""
import math
import time
import uuid

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS valuation_feedback (
      query_id TEXT PRIMARY KEY REFERENCES pricing_queries(query_id) ON DELETE CASCADE,
      recorded_at TEXT NOT NULL,
      market TEXT NOT NULL CHECK(market IN ('ask','bid')),
      rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 4)
    )""",
    "CREATE INDEX IF NOT EXISTS valuation_feedback_timestamp ON valuation_feedback(recorded_at)",
]


def save(side, price, fields, model, source, valuation_date, quote_key, rating):
    """Return success; telemetry outages never interrupt a user valuation."""
    from pricing.analytics import (
        _store, _location, configured, valid_consent, utcnow, _failure,
    )
    import streamlit as st
    if side not in {"ask", "bid"} or type(rating) is not int or rating not in (1,2,3,4):
        return False
    consent = st.session_state.get("_analytics_consent")
    if not configured() or not valid_consent(consent):
        return False
    try:
        amount = float(price)
        if not math.isfinite(amount) or amount <= 0:
            return False
        # Must be an explicit, per-valuation random UUID (not a car fingerprint).
        key = str(uuid.UUID(str(quote_key)))
        store = _store()
        if time.monotonic() < store.retry_after:
            return False
        store.initialize()
        location = _location()
        query_id = str(uuid.uuid5(uuid.UUID(consent["session_id"]),
                                  source + ":" + side + ":" + key))
        store.visit(consent, location)
        store.query(consent, location, query_id, side, amount, model,
                    source, valuation_date, fields)
        store.execute("""INSERT INTO valuation_feedback
          (query_id,recorded_at,market,rating) VALUES (?,?,?,?)
          ON CONFLICT(query_id) DO UPDATE SET
            recorded_at=excluded.recorded_at,rating=excluded.rating""",
            [query_id, utcnow(), side, rating])
        st.session_state["_analytics_status"] = "feedback_saved"
        return True
    except Exception as error:
        _failure("valuation_feedback_write", error)
        if "store" in locals():
            store.retry_after = time.monotonic() + 60
        return False


def render(receipt, *, adjusted=None):
    """One accessible feedback panel beneath Ask/Bid receipts; no modal or rerun trap."""
    import streamlit as st
    from pricing.analytics import valid_consent
    if not any(receipt.get(side) for side in ("ask","bid")):
        return
    quote_key = receipt.get("quote_id")
    if not quote_key:
        return
    st.markdown("### 🎯 How close was our estimate?")
    st.caption("You know your car better than our neural networks. "
               "Rate the price, not the website. This helps improve future releases.")
    sides=[side for side in ("ask","bid") if receipt.get(side)]
    with st.form("valuation_feedback_form", clear_on_submit=False):
        if len(sides)>1:
            side=st.radio("Which estimate?",sides,format_func=lambda s:
                          "✍ Seller asking" if s=="ask" else "🔨 Wholesale auction",
                          horizontal=True)
        else:
            side=sides[0]
            st.caption("Rating: "+("Seller asking" if side=="ask" else "Wholesale auction"))
        ratings=((1,"1 · Catastrophic 💥"),(2,"2 · Needs work 🔧"),
                 (3,"3 · Good 👍"),(4,"4 · Excellent 🏆"))
        rating=st.radio("How accurate does this price seem?",
                        [v for v,_ in ratings],index=2,
                        format_func=lambda v:dict(ratings)[v],horizontal=False)
        submit=st.form_submit_button("📝 Send my accuracy rating",
                                     use_container_width=True)
    st.caption("Optional. Ratings are opinions, not measured accuracy. "
               "We save this with the valuation details only if you accept analytics. "
               "No advert text, VIN or address is saved.")
    if submit:
        if not valid_consent(st.session_state.get("_analytics_consent")):
            st.info("Your rating was not stored because analytics consent is off. "
                    "You can turn it on under Privacy & cookies.")
            return
        result=receipt[side]
        price=float(result["price"])
        if adjusted and side in adjusted:
            price=float(adjusted[side]["price"])
        success=save(side,price,receipt["fields"],result.get("model","unknown"),
                     "dual-ask-quote",receipt.get("date"),quote_key,int(rating))
        if success:
            st.success("✅ Thank you! Feedback recorded with this valuation. "
                       "No registration, no salesman with a clipboard.")
        else:
            st.warning("We couldn't save your rating right now. Your price is unaffected.")
