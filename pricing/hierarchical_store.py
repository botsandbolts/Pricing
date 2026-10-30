"""Private, consented audit table for explicitly requested hierarchical comparisons."""
import json,math,time,uuid

SCHEMA=[
'''CREATE TABLE IF NOT EXISTS hierarchical_estimations (
 estimate_id TEXT PRIMARY KEY, visitor_id TEXT NOT NULL REFERENCES visitors(visitor_id),
 session_id TEXT NOT NULL, timestamp TEXT NOT NULL, country TEXT, city TEXT,
 side TEXT NOT NULL CHECK(side IN ('ask','bid')),
 estimation_kind TEXT NOT NULL CHECK(estimation_kind='historical_heldout_check'),
 base_price_usd REAL NOT NULL CHECK(base_price_usd>0),
 corrected_price_usd REAL NOT NULL CHECK(corrected_price_usd>0),
 correction_log REAL NOT NULL, correction_sd_log REAL NOT NULL,
 make_support INTEGER NOT NULL, model_support INTEGER NOT NULL,
 model_fingerprint TEXT NOT NULL, method TEXT NOT NULL)''',
'CREATE INDEX IF NOT EXISTS hierarchy_visitor ON hierarchical_estimations(visitor_id)',
'CREATE INDEX IF NOT EXISTS hierarchy_timestamp ON hierarchical_estimations(timestamp)'
]


def record(side,result,bundle,event_key):
    import streamlit as st
    from pricing.analytics import configured,valid_consent,_store,_location,utcnow,_failure
    consent=st.session_state.get('_analytics_consent')
    if not configured() or not valid_consent(consent):return False
    if side not in {'ask','bid'} or side!=bundle.get('side'):return False
    if len(str(bundle.get('fingerprint','')))!=64:return False
    try:
        values=[float(result[k]) for k in ['base_price','corrected_price','delta_log','correction_sd_log']]
        if not all(math.isfinite(v) for v in values) or not all(0<v<=1e7 for v in values[:2]) or values[3]<0:return False
        key=str(uuid.uuid5(uuid.UUID(consent['session_id']),'hierarchy:'+str(event_key)))
        store=_store()
        if time.monotonic()<store.retry_after:return False
        store.initialize();location=_location();store.visit(consent,location)
        store.execute('''INSERT OR IGNORE INTO hierarchical_estimations
          (estimate_id,visitor_id,session_id,timestamp,country,city,side,estimation_kind,
           base_price_usd,corrected_price_usd,correction_log,correction_sd_log,
           make_support,model_support,model_fingerprint,method)
          VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
          [key,consent['visitor_id'],consent['session_id'],utcnow(),location.get('country'),location.get('city'),
           side,'historical_heldout_check',*values,int(result['make_n']),int(result['model_n']),
           bundle['fingerprint'],'sequential_empirical_bayes_log_residuals'])
        return True
    except Exception as error:
        _failure('hierarchical_write',error)
        if 'store' in locals():store.retry_after=time.monotonic()+60
        return False
