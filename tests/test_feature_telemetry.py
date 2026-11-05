"""Feature diagnostics are consented, normalized and non-identifying by design."""
import sqlite3
import uuid
from datetime import datetime, timezone
from unittest.mock import patch

from pricing.analytics import SCHEMA, CONSENT_VERSION, TursoStore
from pricing.feature_telemetry import (
    safe_profile, safe_event, sync_session, record_browser_events,
    record_server_run, record_server_error, text_field,
)


class LocalStore(TursoStore):
    def __init__(self):
        super().__init__('libsql://test.turso.io', 'private-test-token')
        self.db = sqlite3.connect(':memory:')
        self.db.execute('PRAGMA foreign_keys=ON')

    def execute(self, sql, args=()):
        return self.db.execute(sql, args)

    def atomic(self, statements):
        self.db.commit()
        with self.db:
            for sql, args in statements:
                self.db.execute(sql, args)


def consent():
    return {
        'choice': 'accepted', 'version': CONSENT_VERSION,
        'visitor_id': str(uuid.uuid4()), 'session_id': str(uuid.uuid4()),
        'consent_at': datetime.now(timezone.utc).isoformat(),
        'browser': 'Firefox', 'device': 'desktop',
        'browser_language': 'de-DE',
        'browser_languages': ['de-DE', 'en-US'],
        'timezone_name': 'Europe/Berlin',
        'os_family': 'Linux', 'viewport_bucket': 'large',
    }


def location():
    return {'country':'DE','country_name':'Germany','city':'Hamburg',
            'region':'Hamburg','geo_provider':'GeoJS / MaxMind GeoLite'}


def test_complete_database_schema_has_foreign_keys_and_indexes():
    store = LocalStore()
    store.initialize()
    assert {'analytics_sessions', 'feature_events', 'feature_runs', 'feature_errors'} <= {
        row[0] for row in store.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    c = consent()
    store.visit(c, location())
    store.visit(c, location())
    assert store.db.execute('SELECT COUNT(*) FROM analytics_sessions').fetchone()[0] == 1
    assert store.db.execute('SELECT country_code,country_name,city,browser_language,timezone_name '
                            'FROM analytics_sessions').fetchone() == (
                            'DE','Germany','Hamburg','de-DE','Europe/Berlin')
    assert store.db.execute('SELECT geo_precision FROM analytics_sessions').fetchone() == ('ip_approximate',)


def test_profile_and_geography_do_not_capture_unbounded_values():
    c=consent()
    c.update(browser_language='bad<script>', browser_languages=['de-DE','en-US','bad<script>'],
             timezone_name='<script>', os_family='HackOS', viewport_bucket='3000')
    p=safe_profile(c)
    assert p['browser_language'] is None
    assert p['browser_languages_json'] == '["de-DE", "en-US"]'
    assert p['timezone_name'] is None
    assert p['os_family'] == 'Other'
    assert p['viewport_bucket'] is None
    assert text_field('user@example.com') is None
    assert text_field('Hamburg') == 'Hamburg'


def test_browser_event_allowlist_drops_user_values_and_invalid_ids():
    good={'event_id':str(uuid.uuid4()),'feature_name':'calculator',
          'subpanel_name':'regular_pricing','event_type':'estimate_requested',
          'action_name':'asking','duration_seconds':2}
    assert safe_event(good)[1:6] == ('calculator','regular_pricing','estimate_requested','asking',2)
    assert safe_event({**good,'feature_name':'my personal car details'}) is None
    assert safe_event({**good,'event_type':'seller_description'}) is None
    assert safe_event({**good,'event_id':'not-a-uuid'}) is None
    assert safe_event({**good,'action_name':'person@example.com'})[4] is None
    assert safe_event({**good,'duration_seconds':'inf'})[5] is None


def test_consent_gate_idempotency_and_server_outcomes():
    import streamlit as st
    store=LocalStore(); c=consent()
    event={'event_id':str(uuid.uuid4()),'feature_name':'taste_garage',
           'subpanel_name':'none','event_type':'panel_view'}
    with patch.dict(st.session_state, {}, clear=True), \\
         patch('pricing.analytics.configured', return_value=True), \\
         patch('pricing.analytics._store', return_value=store), \\
         patch('pricing.analytics._location', return_value=location()):
        assert not record_browser_events([event])
        st.session_state['_analytics_consent']=c
        assert record_browser_events([event,event])
        assert record_browser_events([event])
        assert store.db.execute('SELECT COUNT(*) FROM feature_events').fetchone()[0] == 1
        assert record_server_run('taste_garage','hdbscan','success',25,10,345,
                                 event_key='shortlist-one')
        assert record_server_run('taste_garage','hdbscan','success',25,10,345,
                                 event_key='shortlist-one')
        assert record_server_error('taste_garage','ranker_failed')
        assert store.db.execute('SELECT COUNT(*) FROM feature_runs').fetchone()[0] == 1
        assert store.db.execute('SELECT COUNT(*) FROM feature_errors').fetchone()[0] == 1
        assert store.db.execute('SELECT input_count,result_count,duration_ms '
                                'FROM feature_runs').fetchone() == (25,10,345)


def test_child_first_retention_does_not_leave_orphan_sessions():
    from datetime import timedelta
    store=LocalStore();store.initialize();c=consent()
    store.visit(c,location())
    old=(datetime.now(timezone.utc)-timedelta(days=91)).isoformat(timespec='seconds')
    store.execute('UPDATE visitors SET last_seen=?',(old,))
    store.execute('UPDATE analytics_sessions SET last_seen=?',(old,))
    store.purge()
    assert store.db.execute('SELECT COUNT(*) FROM visitors').fetchone()[0] == 0
    assert store.db.execute('SELECT COUNT(*) FROM analytics_sessions').fetchone()[0] == 0
