from datetime import date,datetime,timedelta,timezone
from unittest.mock import patch
import io,sqlite3
import pytest
from PIL import Image
from tests.test_analytics import LocalStore
from pricing.sale_feedback import initialize,submit,delete_own,purge,validate_outcome
from tests.test_deal_tools import CAR

def test_private_feedback_atomic_evidence_and_own_deletion():
    s=LocalStore();initialize(s)
    image=io.BytesIO();Image.new('RGB',(50,50),'white').save(image,format='PNG')
    with patch('pricing.analytics._store',return_value=s),patch('pricing.analytics.configured',return_value=True):
        receipt=submit(CAR,11000,date.today(),'private_sale',12000,'test',image.getvalue())
        assert s.db.execute('SELECT verification_status FROM sale_outcomes').fetchone()==('unverified',)
        assert s.db.execute('SELECT COUNT(*) FROM sale_evidence').fetchone()==(1,)
        with pytest.raises(sqlite3.IntegrityError):submit(CAR,11001,date.today(),'private_sale',12000,'test',image.getvalue())
        assert s.db.execute('SELECT COUNT(*) FROM sale_outcomes').fetchone()==(1,)
        delete_own(receipt['submission_id'],'wrong-but-valid-length-key')
        assert s.db.execute('SELECT COUNT(*) FROM sale_outcomes').fetchone()==(1,)
        delete_own(receipt['submission_id'],receipt['deletion_key'])
        assert s.db.execute('SELECT COUNT(*) FROM sale_evidence').fetchone()==(0,)
        assert s.db.execute('SELECT COUNT(*) FROM sale_outcomes').fetchone()==(0,)

def test_retention_and_validation():
    s=LocalStore()
    with patch('pricing.analytics._store',return_value=s),patch('pricing.analytics.configured',return_value=True):
        submit(CAR,11000,date.today(),'private_sale',12000,'test')
    old=(datetime.now(timezone.utc)-timedelta(days=91)).isoformat(timespec='seconds')
    s.execute('UPDATE sale_outcomes SET submitted_at=?',(old,));purge(s)
    assert s.db.execute('SELECT COUNT(*) FROM sale_outcomes').fetchone()==(0,)
    for price,when,market in [(float('nan'),date.today(),'private_sale'),(10,date.today()+timedelta(days=1),'private_sale'),(10,date.today(),'fake')]:
        with pytest.raises(ValueError):validate_outcome(CAR,price,when,market)

def test_http_batch_conditions_bind_blobs_and_reject_failed_commit():
    from pricing.analytics import TursoStore
    from unittest.mock import Mock
    s=TursoStore('libsql://test.turso.io','not-real')
    result={'step_results':[{'affected_row_count':0}]*3+[None],'step_errors':[None]*4}
    with patch.object(s,'_send',return_value=result) as send:
        s.atomic([('INSERT INTO t VALUES (?)',(b'private',))])
    steps=send.call_args.args[0]['batch']['steps']
    assert steps[1]['stmt']['args'][0]['type']=='blob'
    assert steps[2]['condition']=={'type':'ok','step':1}
    assert steps[3]['condition']=={'type':'not','cond':{'type':'ok','step':2}}
    with patch.object(s,'_send',return_value={'step_results':[{},None,None,{}],'step_errors':[None,{'message':'failed'},None,None]}):
        with pytest.raises(RuntimeError):s.atomic([('bad',())])
