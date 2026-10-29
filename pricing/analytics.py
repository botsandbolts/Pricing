"""Private, opt-in Turso analytics. Never imported by model training/inference."""
import json
import logging
import math
import os
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

import requests

CONSENT_VERSION = '2026-10-28-v1'
RETENTION_DAYS = 90
SCHEMA = [
    '''CREATE TABLE IF NOT EXISTS visitors (
       visitor_id TEXT PRIMARY KEY, first_seen TEXT NOT NULL, last_seen TEXT NOT NULL,
       country TEXT, city TEXT, region TEXT, geo_provider TEXT, browser TEXT NOT NULL, device TEXT NOT NULL,
       consent_version TEXT NOT NULL, consent_at TEXT NOT NULL)''',
    '''CREATE TABLE IF NOT EXISTS pricing_queries (
       query_id TEXT PRIMARY KEY, visitor_id TEXT NOT NULL REFERENCES visitors(visitor_id),
       session_id TEXT NOT NULL, timestamp TEXT NOT NULL, country TEXT, city TEXT, region TEXT, geo_provider TEXT,
       side TEXT NOT NULL CHECK(side IN ('ask','bid')), price_usd REAL NOT NULL,
       model TEXT NOT NULL, source TEXT NOT NULL, valuation_date TEXT,
       vehicle_json TEXT NOT NULL)''',
    'CREATE INDEX IF NOT EXISTS queries_visitor ON pricing_queries(visitor_id)',
    'CREATE INDEX IF NOT EXISTS queries_timestamp ON pricing_queries(timestamp)',
]
VEHICLE_FIELDS = {'year', 'manufacturer', 'model', 'odometer', 'type', 'fuel',
                  'transmission', 'drive', 'condition', 'title_status', 'cylinders',
                  'paint_color', 'state', 'auction_condition'}


def failure_code(error):
    """Return only allowlisted categories; provider messages never leave this function."""
    code = 'internal'
    if isinstance(error, requests.HTTPError):
        status = error.response.status_code if error.response is not None else 0
        code = 'authentication_' + str(status) if status in (401, 403) else 'http_' + str(status)
        if status in (400, 401, 403) and error.response is not None:
            # Classify locally; never expose the provider's raw error message.
            body = error.response.text[:4096].lower()
            auth_error = any(word in body for word in ('jwt', 'authorization', 'authentication',
                                                      'auth token', 'invalid token', 'token expired'))
            if auth_error:
                if 'expired' in body or 'expiration' in body:
                    code = 'token_expired'
                elif any(word in body for word in ('not yet valid', 'issued in the future', 'immature')):
                    code = 'token_not_yet_valid'
                elif 'signature' in body:
                    code = 'token_signature_rejected'
                elif any(word in body for word in ('base64', 'decode', 'parse', 'malformed',
                                                  'segment', 'format', 'deserialize')):
                    code = 'token_format_rejected'
                elif any(word in body for word in ('scope', 'permission', 'database mismatch',
                                                  'audience', 'namespace')):
                    code = 'token_scope_rejected'
                else:
                    code = 'authentication_' + str(status)
            elif status == 400 and 'want_rows' in body:
                code = 'protocol_want_rows'
            elif status == 400 and 'baton' in body:
                code = 'protocol_baton'
            elif status == 400 and any(word in body for word in ('deserialize', 'json', 'missing field')):
                code = 'protocol_json'
    elif isinstance(error, requests.Timeout):
        code = 'timeout'
    elif isinstance(error, requests.ConnectionError):
        code = 'network'
    elif isinstance(error, ValueError):
        code = 'configuration'
    elif isinstance(error, RuntimeError):
        code = 'database_statement'
    return code


def _failure(stage, error):
    """Safe diagnostics: never log exception text, request bodies or credentials."""
    import streamlit as st
    code = failure_code(error)
    st.session_state['_analytics_status'] = stage + ':' + code
    logging.getLogger(__name__).warning('Analytics unavailable stage=%s code=%s', stage, code)


def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def valid_id(value):
    try:
        return str(uuid.UUID(str(value))) == value
    except (ValueError, TypeError, AttributeError):
        return False


def valid_consent(value):
    if not isinstance(value, dict) or value.get('choice') != 'accepted':
        return False
    if value.get('version') != CONSENT_VERSION:
        return False
    if not all(valid_id(value.get(k)) for k in ('visitor_id', 'session_id')):
        return False
    try:
        age = datetime.now(timezone.utc) - datetime.fromisoformat(value['consent_at'])
        return timedelta(0) <= age <= timedelta(days=90)
    except (ValueError, TypeError, KeyError):
        return False


def safe_vehicle(fields):
    """Allowlist only; never store seller text, URLs, VINs, contacts or arbitrary keys."""
    out = {}
    for key in VEHICLE_FIELDS & fields.keys():
        value = fields[key]
        if value is None:
            continue
        if key in {'year', 'odometer', 'auction_condition'}:
            try:
                number = float(value)
                if math.isfinite(number):
                    out[key] = number
            except (ValueError, TypeError):
                pass
        else:
            # These form fields are free text; reject obvious contact/address content.
            text = str(value).strip()[:80]
            if not re.search(r'@|https?://|www\.|\+?\d[\d ()-]{7,}', text, re.I):
                out[key] = text
    return out


def country_from_headers(headers, trusted_header=''):
    # Never guess country from UI state (the car's state is not visitor location).
    if not trusted_header:
        return None
    value = str(headers.get(trusted_header, '')).upper()
    return value if re.fullmatch('[A-Z]{2}', value) and value not in {'XX', 'T1'} else None


def _arg(value):
    if value is None:
        return {'type': 'null'}
    if isinstance(value, (int, float)):
        return {'type': 'float', 'value': float(value)}
    return {'type': 'text', 'value': str(value)}


class TursoStore:
    def __init__(self, url, token):
        parsed = urlsplit(url.replace('libsql://', 'https://', 1))
        if (parsed.scheme != 'https' or not parsed.hostname or
                not parsed.hostname.endswith('.turso.io') or parsed.username or
                parsed.password or parsed.query or parsed.fragment or
                parsed.path not in ('', '/')):
            raise ValueError('Expected a Turso HTTPS/libsql database origin')
        self.endpoint = 'https://' + parsed.netloc + '/v2/pipeline'
        self.token = str(token).strip()
        self.ready = False
        self.retry_after = 0
        self.last_purge = 0

    def execute(self, sql, args=()):
        response = requests.post(self.endpoint, headers={'Authorization': 'Bearer ' + self.token},
            json={'baton': None, 'requests': [{'type': 'execute', 'stmt': {'sql': sql,
                'args': [_arg(v) for v in args], 'named_args': [], 'want_rows': True}}, {'type': 'close'}]},
            timeout=(2, 3), allow_redirects=False)
        response.raise_for_status()
        result = response.json()['results'][0]
        if result.get('type') != 'ok':
            raise RuntimeError('Turso rejected analytics statement')
        return result['response']['result']

    def initialize(self):
        if not self.ready:
            for sql in SCHEMA:
                self.execute(sql)
            self.ready = True

    def purge(self):
        cutoff = (datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)).isoformat(timespec='seconds')
        self.execute('DELETE FROM pricing_queries WHERE timestamp < ?', [cutoff])
        self.execute('DELETE FROM visitors WHERE last_seen < ? AND NOT EXISTS '
                     '(SELECT 1 FROM pricing_queries q WHERE q.visitor_id=visitors.visitor_id)', [cutoff])

    def visit(self, consent, location):
        now = utcnow()
        browser = consent.get('browser') if consent.get('browser') in {'Chrome', 'Firefox', 'Safari', 'Edge'} else 'Other'
        device = consent.get('device') if consent.get('device') in {'mobile', 'desktop'} else 'unknown'
        self.execute('''INSERT INTO visitors VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(visitor_id) DO UPDATE SET last_seen=excluded.last_seen,
            country=excluded.country, city=excluded.city, region=excluded.region,
            geo_provider=excluded.geo_provider, browser=excluded.browser, device=excluded.device,
            consent_version=excluded.consent_version, consent_at=excluded.consent_at''',
            [consent['visitor_id'], now, now, location['country'], location['city'], location['region'],
             location['geo_provider'], browser, device, CONSENT_VERSION, consent['consent_at']])

    def query(self, consent, location, query_id, side, price, model, source, valuation_date, fields):
        self.execute('''INSERT OR IGNORE INTO pricing_queries VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            [query_id, consent['visitor_id'], consent['session_id'], utcnow(), location['country'],
             location['city'], location['region'], location['geo_provider'],
             side, price, str(model)[:160], str(source)[:80], str(valuation_date)[:10] if valuation_date else None,
             json.dumps(safe_vehicle(fields), ensure_ascii=False)])


def _setting(name, default=''):
    import streamlit as st
    # Streamlit mirrors root string secrets into os.environ. Reading that mirror
    # first can hide an updated secret. The current secrets file is authoritative.
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


def configured():
    return (str(_setting('ANALYTICS_ENABLED')).lower() == 'true' and
            all(_setting(k) for k in ('TURSO_DATABASE_URL', 'TURSO_AUTH_TOKEN',
                'ANALYTICS_CONTROLLER_NAME', 'ANALYTICS_CONTROLLER_CONTACT')))


def _store():
    import streamlit as st
    @st.cache_resource(show_spinner=False)
    def connect(url, token):
        return TursoStore(url, token)
    return connect(_setting('TURSO_DATABASE_URL'), _setting('TURSO_AUTH_TOKEN'))


def _location():
    import streamlit as st
    from pricing.geo import approximate_location
    if _setting('ANALYTICS_GEO_MODE', 'geojs') == 'geojs':
        consent = st.session_state.get('_analytics_consent') or {}
        value = consent.get('location') or {}
        if isinstance(value, dict) and value.get('geo_provider') == 'GeoJS / MaxMind GeoLite':
            return {k: (str(value[k])[:100] if value.get(k) else None)
                    for k in ('country', 'city', 'region', 'geo_provider')}
    location = approximate_location(st.context.headers, _setting('ANALYTICS_TRUSTED_IP_HEADER'),
        _setting('ANALYTICS_GEO_MMDB'), _setting('ANALYTICS_GEO_PROVIDER'))
    if not location['country']:
        location['country'] = country_from_headers(st.context.headers, _setting('ANALYTICS_COUNTRY_HEADER'))
    return location


def record_price(side, price, fields, model, source, valuation_date=None, event_key=None):
    """Best effort, opt-in only. A failed analytics request never breaks pricing."""
    import streamlit as st
    consent = st.session_state.get('_analytics_consent')
    if not configured() or not valid_consent(consent):
        return False
    if side not in {'ask', 'bid'}:
        return False
    try:
        price = float(price)
        if not math.isfinite(price) or price <= 0:
            return False
        store = _store()
        if time.monotonic() < store.retry_after:
            return False
        store.initialize()
        location = _location()
        # Event keys are per explicit user request, never a vehicle fingerprint.
        key = str(event_key or uuid.uuid4())
        query_id = str(uuid.uuid5(uuid.UUID(consent['session_id']), source + ':' + side + ':' + key))
        done = st.session_state.setdefault('_analytics_done', set())
        if query_id in done:
            return True
        store.visit(consent, location)
        store.query(consent, location, query_id, side, price, model, source, valuation_date, fields)
        st.session_state['_analytics_status'] = 'price_saved'
        done.add(query_id)
        if len(done) > 1000:
            done.clear()
        return True
    except Exception as error:
        # Do not put credentials, fields, URLs or server error payloads in logs/UI.
        _failure('price_write', error)
        if 'store' in locals():
            store.retry_after = time.monotonic() + 60
        return False


def render_privacy():
    """A missing/unsupported optional component cannot block the price calculator."""
    import streamlit as st
    try:
        _render_privacy_impl()
    except Exception as error:
        st.session_state['_analytics_consent'] = None
        _failure('consent_component', error)


def _persist_visit(consent, retry=False):
    """Check auth with SELECT 1, then acknowledge a real visitor upsert."""
    import streamlit as st
    if not configured() or not valid_consent(consent):
        return
    stage = 'connection_setup'
    try:
        store = _store()
        if retry:
            store.retry_after = 0
        if time.monotonic() < store.retry_after:
            return
        marker = consent['visitor_id'] + ':' + consent['session_id']
        if retry or st.session_state.get('_analytics_visit') != marker:
            stage = 'connection_check'
            store.execute('SELECT 1')
            stage = 'schema_setup'
            store.initialize()
            stage = 'visitor_write'
            store.visit(consent, _location())
            st.session_state['_analytics_visit'] = marker
            st.session_state['_analytics_status'] = 'visitor_saved'
        if time.monotonic() - store.last_purge > 3600:
            stage = 'retention'
            store.purge()
            store.last_purge = time.monotonic()
    except Exception as error:
        _failure(stage, error)
        if 'store' in locals():
            store.retry_after = time.monotonic() + 60


def _render_privacy_impl():
    import streamlit as st
    from pathlib import Path
    import streamlit.components.v2 as components
    folder = Path(__file__).parent / 'consent'
    component = components.component('pricing_consent', html=(folder/'notice.html').read_text(),
        js=(folder/'notice.js').read_text(), css=(folder/'notice.css').read_text())
    enabled = configured()
    previous = st.session_state.get('pricing_privacy') or {}
    consent = previous.get('consent')
    st.session_state['_analytics_consent'] = consent if enabled and valid_consent(consent) else None
    # Process the persisted component state before mounting, so the browser sees
    # the result of this check immediately instead of a one-rerun-old "pending".
    _persist_visit(consent, retry=bool(previous.get('check_connection')))
    result = component(key='pricing_privacy', data={'enabled': enabled, 'version': CONSENT_VERSION,
        'current': consent,
        'geo_mode': _setting('ANALYTICS_GEO_MODE', 'geojs'),
        'backend_status': st.session_state.get('_analytics_status', 'pending'),
        'controller': _setting('ANALYTICS_CONTROLLER_NAME', 'Car Price Studio operator'),
        'contact': _setting('ANALYTICS_CONTROLLER_CONTACT', 'Contact the studio operator')},
        on_consent_change=lambda: None, on_check_connection_change=lambda: None)
    consent = result.consent
    st.session_state['_analytics_consent'] = consent if valid_consent(consent) and enabled else None
    if consent and not valid_consent(consent) and consent.get('choice') == 'accepted':
        st.session_state['_analytics_status'] = 'consent:invalid'
    if not enabled or not valid_consent(consent):
        return
    _persist_visit(consent)


def render_connection_check():
    """Small troubleshooting view: no model loads, credentials or database records."""
    import streamlit as st
    st.title('Pricing · database connection check')
    st.write('This page checks whether the studio can save a consented visit. '
             'It does not load the car-pricing models or show database records.')
    st.info('First tap Privacy & cookies below and accept analytics. '
            'Then press Check database connection.')
    render_privacy()
    consent = st.session_state.get('_analytics_consent')
    allowed = bool(configured() and valid_consent(consent))
    if st.button('Check database connection', type='primary', disabled=not allowed):
        with st.spinner('Checking connection and visitor write…'):
            _persist_visit(consent, retry=True)
    if not configured():
        st.warning('Analytics is disabled or its private settings are incomplete.')
    elif not allowed:
        st.caption('The check becomes available after your consent reaches the server.')
    status = st.session_state.get('_analytics_status', 'pending')
    st.caption('Collection status')
    st.code(status, language=None)
    if status in {'visitor_saved', 'price_saved'}:
        st.success('Turso acknowledged a saved record. The connection is working.')
    elif status != 'pending':
        st.warning('The check did not confirm a saved record. Share only the status code above.')
    st.markdown('[Return to the car-price calculator](./)')
