import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, Mock

import pytest
import requests

from pricing.analytics import (SCHEMA, CONSENT_VERSION, TursoStore, valid_consent,
    safe_vehicle, country_from_headers, record_price, _failure, failure_code,
    _setting, _persist_visit)
from pricing.geo import approximate_location


def consent():
    return dict(choice='accepted', version=CONSENT_VERSION,
        consent_at=datetime.now(timezone.utc).isoformat(),
        visitor_id=str(uuid.uuid4()), session_id=str(uuid.uuid4()))


class LocalStore(TursoStore):
    def __init__(self):
        super().__init__('libsql://test.turso.io', 'test-token')
        self.db = sqlite3.connect(':memory:')
        self.db.execute('PRAGMA foreign_keys=ON')

    def execute(self, sql, args=()):
        return self.db.execute(sql, args)


def test_consent_is_explicit_versioned_and_expiring():
    c = consent()
    assert valid_consent(c)
    for change in [dict(choice='rejected'), dict(version='old'), dict(visitor_id='bad'),
                   dict(consent_at='invalid'), dict(consent_at='2020-01-01T00:00:00+00:00')]:
        assert not valid_consent({**c, **change})
    assert not valid_consent(None)


def test_schema_links_location_and_deduplicates_queries():
    s = LocalStore(); s.initialize(); s.initialize()
    c = consent(); geo = dict(country='DE', city='Berlin', region='Berlin', geo_provider='DB-IP Lite')
    s.visit(c, geo)
    for _ in range(2):
        s.query(c, geo, 'event-1', 'bid', 7000, 'quantile', 'test', '2026-10-28', {'year': 2015})
    assert s.db.execute('SELECT country,city FROM visitors').fetchone() == ('DE', 'Berlin')
    assert s.db.execute('SELECT country,city,price_usd FROM pricing_queries').fetchall() == [('DE', 'Berlin', 7000)]
    with pytest.raises(sqlite3.IntegrityError):
        s.query({**c, 'visitor_id':'missing'}, geo, 'event-2', 'ask', 8000, 'aft', 'test', None, {})


def test_retention_and_child_first_deletion():
    s = LocalStore(); s.initialize(); c=consent()
    geo=dict(country=None, city=None, region=None, geo_provider=None)
    s.visit(c,geo);s.query(c,geo,'old','ask',50,'test','test',None,{})
    old=(datetime.now(timezone.utc)-timedelta(days=91)).isoformat(timespec='seconds')
    s.execute('UPDATE pricing_queries SET timestamp=?',[old]);s.execute('UPDATE visitors SET last_seen=?',[old])
    s.purge()
    assert s.db.execute('SELECT COUNT(*) FROM visitors').fetchone()[0] == 0
    assert s.db.execute('SELECT COUNT(*) FROM pricing_queries').fetchone()[0] == 0


def test_allowlist_excludes_text_contacts_and_nonfinite_values():
    result=safe_vehicle({'model':'Corolla','manufacturer':'person@example.com','text':'secret',
        'description':'secret','ip':'1.1.1.1','odometer':float('nan'),'year':2015})
    assert result == {'model':'Corolla','year':2015.0}


def test_bound_http_protocol_and_no_redirects():
    response=Mock(); response.json.return_value={'results':[{'type':'ok','response':{'result':{}}}]}
    with patch('pricing.analytics.requests.post',return_value=response) as request:
        TursoStore('libsql://test.turso.io','secret').execute('SELECT ?',["x');DROP TABLE visitors;--"])
    args=request.call_args.kwargs
    assert args['allow_redirects'] is False
    assert args['json']['baton'] is None
    assert args['json']['requests'][0]['stmt']['want_rows'] is True
    assert args['json']['requests'][0]['stmt']['sql'] == 'SELECT ?'
    assert args['json']['requests'][0]['stmt']['args'][0]['type'] == 'text'
    assert args['headers']['Authorization'] == 'Bearer secret'
    for url in ['http://test.turso.io','https://evil.example','https://x.turso.io/path',
                'https://user:pass@x.turso.io','https://x.turso.io?token=x']:
        with pytest.raises(ValueError):TursoStore(url,'x')


def test_geo_unknown_and_local_country_city_only():
    assert country_from_headers({'x-country':'DE'}) is None
    assert country_from_headers({'x-country':'DE'},'x-country') == 'DE'
    for ip in ['127.0.0.1','10.1.1.1','1.1.1.1, 2.2.2.2','bad']:
        assert approximate_location({'client':ip},'client','absent.mmdb','DB-IP Lite')['city'] is None
    reader=Mock();reader.get.return_value={'country':{'iso_code':'DE'},'city':{'names':{'en':'Berlin'}},
        'subdivisions':[{'names':{'en':'Berlin'}}],'location':{'latitude':52.5,'longitude':13.4}}
    with patch('pricing.geo.Path.stat',return_value=Mock(st_mtime_ns=1)),patch('pricing.geo._reader',return_value=reader):
        result=approximate_location({'client':'1.1.1.1'},'client','test.mmdb','DB-IP Lite')
    assert result == dict(country='DE',city='Berlin',region='Berlin',geo_provider='DB-IP Lite')
    assert not {'latitude','longitude','ip'} & result.keys()


def test_no_consent_no_database_or_geo_call():
    import streamlit as st
    with patch.dict(st.session_state,{'_analytics_consent':None}),patch('pricing.analytics.configured',return_value=True),\
         patch('pricing.analytics._store') as store,patch('pricing.analytics._location') as geo:
        assert not record_price('ask',100,{},'model','test')
        store.assert_not_called();geo.assert_not_called()


def test_database_failure_never_breaks_prediction():
    import streamlit as st
    c=consent(); broken=Mock(retry_after=0); broken.initialize.side_effect=RuntimeError('offline')
    with patch.dict(st.session_state,{'_analytics_consent':c}),patch('pricing.analytics.configured',return_value=True),\
         patch('pricing.analytics._store',return_value=broken):
        assert not record_price('bid',100,{},'model','test')
        assert broken.retry_after > 0


def test_authentication_diagnostic_never_exposes_secret_or_response(caplog):
    import streamlit as st
    response = requests.Response(); response.status_code = 401
    error = requests.HTTPError('SECRET_TOKEN_AND_SERVER_BODY', response=response)
    with patch.dict(st.session_state, {}, clear=True):
        _failure('visitor_write', error)
        assert st.session_state['_analytics_status'] == 'visitor_write:authentication_401'
    assert 'SECRET_TOKEN_AND_SERVER_BODY' not in caplog.text
    assert 'stage=visitor_write code=authentication_401' in caplog.text


def test_price_event_saves_linked_rows_and_only_marks_success_after_write():
    import streamlit as st
    s = LocalStore(); c = consent()
    geo = dict(country='DE', city='Berlin', region='Berlin', geo_provider='GeoJS / MaxMind GeoLite')
    with patch.dict(st.session_state, {'_analytics_consent':c}, clear=True), \
         patch('pricing.analytics.configured', return_value=True), \
         patch('pricing.analytics._store', return_value=s), \
         patch('pricing.analytics._location', return_value=geo):
        for _ in range(2):
            assert record_price('ask', 12996, {'year':2015}, 'aft', 'calculator', event_key='same-request')
        assert st.session_state['_analytics_status'] == 'price_saved'
        assert s.db.execute('SELECT COUNT(*) FROM visitors').fetchone()[0] == 1
        assert s.db.execute('SELECT COUNT(*) FROM pricing_queries').fetchone()[0] == 1
        assert s.db.execute('SELECT country,city,price_usd FROM pricing_queries').fetchone() == ('DE','Berlin',12996)


@pytest.mark.parametrize('message,expected', [
    ('Invalid JWT: token has expired SECRET', 'token_expired'),
    ('Invalid auth token: JWT not yet valid SECRET', 'token_not_yet_valid'),
    ('JWT signature verification failed SECRET', 'token_signature_rejected'),
    ('Failed to parse auth token: invalid base64 SECRET', 'token_format_rejected'),
    ('JWT audience or namespace mismatch SECRET', 'token_scope_rejected'),
    ('Invalid auth token SECRET', 'authentication_400'),
    ('JSON missing field want_rows SECRET', 'protocol_want_rows'),
    ('Invalid baton SECRET', 'protocol_baton'),
    ('Failed to deserialize JSON SECRET', 'protocol_json'),
    ('SQL unexpected token near SECRET', 'http_400'),
])
def test_provider_errors_are_allowlisted_and_sql_token_is_not_auth(message, expected):
    response = requests.Response(); response.status_code = 400
    response._content = message.encode()
    assert failure_code(requests.HTTPError(response=response)) == expected


def test_current_streamlit_secret_wins_over_environment_mirror():
    import streamlit as st
    with patch.dict('os.environ', {'TURSO_AUTH_TOKEN':'old-mirror'}), \
         patch.object(st, 'secrets', {'TURSO_AUTH_TOKEN':'current-secret'}):
        assert _setting('TURSO_AUTH_TOKEN') == 'current-secret'
    with patch.dict('os.environ', {'TURSO_AUTH_TOKEN':'cli-secret'}), \
         patch.object(st, 'secrets', {}):
        assert _setting('TURSO_AUTH_TOKEN') == 'cli-secret'


def test_connection_check_reports_auth_stage_and_never_initializes_on_failure():
    import streamlit as st
    response = requests.Response(); response.status_code = 400
    response._content = b'Failed to parse auth token: invalid base64 SECRET'
    s = Mock(retry_after=0, last_purge=0)
    s.execute.side_effect = requests.HTTPError(response=response)
    with patch.dict(st.session_state, {}, clear=True), \
         patch('pricing.analytics.configured', return_value=True), \
         patch('pricing.analytics._store', return_value=s):
        _persist_visit(consent(), retry=True)
        s.execute.assert_called_once_with('SELECT 1')
        s.initialize.assert_not_called(); s.visit.assert_not_called()
        assert st.session_state['_analytics_status'] == 'connection_check:token_format_rejected'
        assert s.retry_after > 0


def test_visit_check_is_consent_only_and_idempotent():
    import streamlit as st
    s = LocalStore(); c = consent()
    geo = dict(country='DE', city='Berlin', region='Berlin', geo_provider='GeoJS / MaxMind GeoLite')
    with patch.dict(st.session_state, {}, clear=True), \
         patch('pricing.analytics.configured', return_value=True), \
         patch('pricing.analytics._store', return_value=s) as connect, \
         patch('pricing.analytics._location', return_value=geo):
        _persist_visit(None, retry=True)
        connect.assert_not_called()
        for retry in (False, False, True):
            _persist_visit(c, retry=retry)
        assert st.session_state['_analytics_status'] == 'visitor_saved'
        assert s.db.execute('SELECT COUNT(*) FROM visitors').fetchone()[0] == 1
