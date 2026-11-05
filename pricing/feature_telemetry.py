"""First-party, consented functionality telemetry for Turso/libSQL.

No IP, coordinates, user-agent strings, seller text, VINs or form values are stored.
All browser-provided values are untrusted and validated against bounded schemas.
"""
import json
import re
import time
import uuid

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS analytics_sessions (
       session_id TEXT PRIMARY KEY,
       visitor_id TEXT NOT NULL REFERENCES visitors(visitor_id),
       started_at TEXT NOT NULL, last_seen TEXT NOT NULL,
       country_code TEXT, country_name TEXT, region TEXT, city TEXT,
       geo_provider TEXT, geo_precision TEXT NOT NULL DEFAULT 'unknown',
       browser_language TEXT, browser_languages_json TEXT, timezone_name TEXT,
       browser TEXT, device TEXT, os_family TEXT, viewport_bucket TEXT,
       consent_version TEXT NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS analytics_sessions_visitor ON analytics_sessions(visitor_id)",
    "CREATE INDEX IF NOT EXISTS analytics_sessions_last_seen ON analytics_sessions(last_seen)",
    """CREATE TABLE IF NOT EXISTS feature_events (
       event_id TEXT PRIMARY KEY,
       visitor_id TEXT NOT NULL REFERENCES visitors(visitor_id),
       session_id TEXT NOT NULL REFERENCES analytics_sessions(session_id),
       recorded_at TEXT NOT NULL,
       feature_name TEXT NOT NULL, subpanel_name TEXT,
       event_type TEXT NOT NULL, action_name TEXT,
       duration_seconds INTEGER, outcome TEXT)""",
    "CREATE INDEX IF NOT EXISTS feature_events_time ON feature_events(recorded_at)",
    "CREATE INDEX IF NOT EXISTS feature_events_feature ON feature_events(feature_name, event_type, recorded_at)",
    "CREATE INDEX IF NOT EXISTS feature_events_session ON feature_events(session_id)",
    """CREATE TABLE IF NOT EXISTS feature_runs (
       run_id TEXT PRIMARY KEY,
       visitor_id TEXT NOT NULL REFERENCES visitors(visitor_id),
       session_id TEXT NOT NULL REFERENCES analytics_sessions(session_id),
       recorded_at TEXT NOT NULL, feature_name TEXT NOT NULL,
       model_family TEXT, status TEXT NOT NULL,
       input_count INTEGER, result_count INTEGER, duration_ms INTEGER)""",
    "CREATE INDEX IF NOT EXISTS feature_runs_time ON feature_runs(recorded_at)",
    "CREATE INDEX IF NOT EXISTS feature_runs_feature ON feature_runs(feature_name, recorded_at)",
    """CREATE TABLE IF NOT EXISTS feature_errors (
       error_id TEXT PRIMARY KEY,
       visitor_id TEXT NOT NULL REFERENCES visitors(visitor_id),
       session_id TEXT NOT NULL REFERENCES analytics_sessions(session_id),
       recorded_at TEXT NOT NULL, feature_name TEXT NOT NULL,
       error_code TEXT NOT NULL, severity TEXT NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS feature_errors_time ON feature_errors(recorded_at)",
]

# Only names of tools and controls, not their values, are permitted.
FEATURES = {
    'calculator', 'listing_inspector', 'taste_garage', 'two_market',
    'value_over_time', 'science', 'evaluation', 'model_lab',
    'deal_detective', 'outlier_lab', 'merchant_lens', 'studio',
}
EVENTS = {
    'panel_view', 'page_view', 'panel_engagement', 'navigation_action',
    'estimate_requested', 'estimate_completed', 'explanation_requested',
    'scenario_requested', 'summary_requested', 'import_requested',
    'import_applied', 'import_action', 'inspector_action',
    'download_requested', 'share_opened', 'share', 'shortlist_action',
    'game_action', 'feedback_action', 'hierarchical_action',
    'chart_interaction', 'section_toggled', 'setting_changed',
    'model_selected',
}
WORD = re.compile(r'^[a-z][a-z0-9_]{0,63}$')
LANGUAGE = re.compile(r'^[a-zA-Z]{2,3}(?:-[A-Za-z0-9]{2,8}){0,4}$')
ZONE = re.compile(r'^[A-Za-z_+\-/]{1,64}$')


def text_field(value, limit=100):
    """Bounded human-readable geography. No emails, addresses or URLs."""
    if not isinstance(value, str):
        return None
    value = ' '.join(value.strip().split())
    if not value or len(value) > limit or re.search(r'[@<>\\/\d]|https?:|www\.', value, re.I):
        return None
    return value


def language_field(value):
    return value if isinstance(value, str) and LANGUAGE.fullmatch(value) else None


def safe_profile(consent):
    if not isinstance(consent, dict):
        consent = {}
    langs = consent.get('browser_languages')
    if not isinstance(langs, list):
        langs = []
    languages = list(dict.fromkeys(x for x in langs[:5] if language_field(x)))
    first = language_field(consent.get('browser_language'))
    if first and first not in languages:
        languages.insert(0, first)
    zone = consent.get('timezone_name')
    return {
        'browser_language': first,
        'browser_languages_json': json.dumps(languages[:5]),
        'timezone_name': zone if isinstance(zone, str) and ZONE.fullmatch(zone) else None,
        'os_family': consent.get('os_family') if consent.get('os_family') in {
            'Windows', 'macOS', 'Linux', 'Android', 'iOS', 'Other'} else 'Other',
        'viewport_bucket': consent.get('viewport_bucket') if consent.get('viewport_bucket') in {
            'small', 'medium', 'large', 'extra_large'} else None,
    }


def _country_name(location):
    return text_field(location.get('country_name'), 80)


def sync_session(store, consent, location):
    """Called after the parent visitor upsert. Existing sessions keep their first timestamp."""
    from pricing.analytics import utcnow, CONSENT_VERSION
    now = utcnow()
    profile = safe_profile(consent)
    code = location.get('country')
    if not isinstance(code, str) or not re.fullmatch(r'[A-Z]{2}', code):
        code = None
    provider = location.get('geo_provider')
    precision = 'self_reported_city' if provider == 'Visitor correction' else (
        'ip_approximate' if provider else 'unknown')
    store.execute(
        """INSERT INTO analytics_sessions (
          session_id, visitor_id, started_at, last_seen,
          country_code, country_name, region, city, geo_provider, geo_precision,
          browser_language, browser_languages_json, timezone_name,
          browser, device, os_family, viewport_bucket, consent_version
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(session_id) DO UPDATE SET
          last_seen=excluded.last_seen, country_code=excluded.country_code,
          country_name=excluded.country_name, region=excluded.region,
          city=excluded.city, geo_provider=excluded.geo_provider,
          geo_precision=excluded.geo_precision,
          browser_language=excluded.browser_language,
          browser_languages_json=excluded.browser_languages_json,
          timezone_name=excluded.timezone_name, browser=excluded.browser,
          device=excluded.device, os_family=excluded.os_family,
          viewport_bucket=excluded.viewport_bucket,
          consent_version=excluded.consent_version""",
        [consent['session_id'], consent['visitor_id'], now, now, code,
         _country_name(location), text_field(location.get('region')),
         text_field(location.get('city')), provider if provider in {
             'GeoJS / MaxMind GeoLite', 'MaxMind GeoLite2', 'DB-IP Lite',
             'Visitor correction'} else None, precision,
         profile['browser_language'], profile['browser_languages_json'],
         profile['timezone_name'], consent.get('browser', 'Other'),
         consent.get('device', 'unknown'), profile['os_family'],
         profile['viewport_bucket'], CONSENT_VERSION])


def safe_event(raw):
    if not isinstance(raw, dict) or raw.get('event_type') not in EVENTS:
        return None
    if raw.get('feature_name') not in FEATURES:
        return None
    try:
        event_id = str(uuid.UUID(str(raw.get('event_id'))))
    except (ValueError, TypeError, AttributeError):
        return None
    subpanel = raw.get('subpanel_name')
    action = raw.get('action_name')
    if subpanel is not None and (not isinstance(subpanel, str) or not WORD.fullmatch(subpanel)):
        subpanel = None
    if action is not None and (not isinstance(action, str) or not WORD.fullmatch(action)):
        action = None
    try:
        duration = int(raw.get('duration_seconds')) if raw.get('duration_seconds') is not None else None
    except (ValueError, TypeError, OverflowError):
        duration = None
    if duration is not None and not 0 <= duration <= 1800:
        duration = None
    return (event_id, raw['feature_name'], subpanel, raw['event_type'],
            action, duration, None)


def record_browser_events(events):
    """A best-effort, idempotent, allowlisted component-to-Turso bridge."""
    import streamlit as st
    from pricing.analytics import configured, valid_consent, _store, _location, utcnow, _failure
    consent = st.session_state.get('_analytics_consent')
    if not configured() or not valid_consent(consent) or not isinstance(events, list):
        return False
    rows = [r for r in (safe_event(x) for x in events[:30]) if r is not None]
    if not rows:
        return False
    try:
        store = _store()
        if time.monotonic() < store.retry_after:
            return False
        store.initialize()
        store.visit(consent, _location())
        now = utcnow()
        store.atomic([
            ("""INSERT OR IGNORE INTO feature_events
               (event_id,visitor_id,session_id,recorded_at,feature_name,
                subpanel_name,event_type,action_name,duration_seconds,outcome)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
             (event_id, consent['visitor_id'], consent['session_id'], now,
              feature, subpanel, event_type, action, duration, outcome))
            for event_id, feature, subpanel, event_type, action, duration, outcome in rows
        ])
        return True
    except Exception as error:
        _failure('feature_events_write', error)
        if 'store' in locals():
            store.retry_after = time.monotonic() + 60
        return False


def record_server_error(feature, code, severity='error'):
    """Explicitly coded failures only; never an exception message or stack trace."""
    import streamlit as st
    from pricing.analytics import configured, valid_consent, _store, _location, utcnow, _failure
    consent = st.session_state.get('_analytics_consent')
    if (not configured() or not valid_consent(consent) or
            feature not in FEATURES or not isinstance(code, str) or
            not WORD.fullmatch(code) or severity not in {'warning', 'error'}):
        return False
    try:
        store = _store()
        if time.monotonic() < store.retry_after:
            return False
        store.initialize()
        store.visit(consent, _location())
        store.execute(
            """INSERT INTO feature_errors
               (error_id,visitor_id,session_id,recorded_at,feature_name,error_code,severity)
               VALUES (?,?,?,?,?,?,?)""",
            (str(uuid.uuid4()), consent['visitor_id'], consent['session_id'],
             utcnow(), feature, code, severity))
        return True
    except Exception as error:
        _failure('feature_error_write', error)
        if 'store' in locals():
            store.retry_after = time.monotonic() + 60
        return False
