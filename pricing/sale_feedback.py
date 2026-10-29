"""Explicit, private sale-outcome submission. No automatic training or public records."""
import hashlib, io, json, math, secrets, uuid
from datetime import date, datetime, timezone, timedelta
from pricing.deal_tools import safe_fields

SCHEMA=[
'''CREATE TABLE IF NOT EXISTS sale_outcomes (
 submission_id TEXT PRIMARY KEY, submitted_at TEXT NOT NULL, sale_date TEXT NOT NULL,
 price_usd REAL NOT NULL, market TEXT NOT NULL, vehicle_json TEXT NOT NULL,
 reference_estimate_usd REAL, reference_date TEXT NOT NULL, model TEXT NOT NULL,
 verification_status TEXT NOT NULL DEFAULT 'unverified', review_reason TEXT,
 evidence_sha256 TEXT, deletion_key_hash TEXT NOT NULL)''',
'''CREATE TABLE IF NOT EXISTS sale_evidence (
 submission_id TEXT PRIMARY KEY REFERENCES sale_outcomes(submission_id),
 content BLOB NOT NULL, content_type TEXT NOT NULL)''',
'CREATE UNIQUE INDEX IF NOT EXISTS feedback_evidence_unique ON sale_outcomes(evidence_sha256) WHERE evidence_sha256 IS NOT NULL',
]


def initialize(store):
    if not getattr(store,'feedback_ready',False):
        store.atomic([(sql,()) for sql in SCHEMA])
        store.feedback_ready=True


def evidence_image(content):
    from PIL import Image, ImageOps
    if len(content)>4*1024*1024:raise ValueError('Evidence must be under 4 MB.')
    with Image.open(io.BytesIO(content)) as image:
        if image.width*image.height>16000000:raise ValueError('Evidence image is too large.')
        image=ImageOps.exif_transpose(image).convert('RGB');image.thumbnail((1400,1400))
        out=io.BytesIO();image.save(out,format='JPEG',quality=75)
    result=out.getvalue()
    if len(result)>1024*1024:raise ValueError('Use a smaller, cropped receipt image.')
    return result


def validate_outcome(fields, price, sold_on, market):
    f=safe_fields(fields);price=float(price)
    if not all(f.get(k) for k in ('year','manufacturer','model')):raise ValueError('Year, make and model are required.')
    if not math.isfinite(price) or not 1<=price<=1000000:raise ValueError('Enter a sale price between $1 and $1 million.')
    if isinstance(sold_on,str):sold_on=date.fromisoformat(sold_on)
    if not date(max(1900,int(f['year'])-1),1,1)<=sold_on<=date.today():raise ValueError('Sale date must be plausible and cannot be in the future.')
    if market not in {'private_sale','dealer_retail','wholesale_auction'}:raise ValueError('Choose a supported sale market.')
    return f,price,sold_on.isoformat()


def submit(fields,price,sold_on,market,estimate,model,evidence=None,submission_id=None):
    from pricing.analytics import _store, configured
    if not configured():raise ValueError('Private storage is not configured. Download your submission instead.')
    f,price,sold_on=validate_outcome(fields,price,sold_on,market)
    if estimate is not None and (not math.isfinite(float(estimate)) or estimate<=0):raise ValueError('Invalid reference estimate.')
    store=_store()
    initialize(store)
    sid=submission_id or str(uuid.uuid4());key=secrets.token_urlsafe(24)
    evidence=evidence_image(evidence) if evidence else None
    digest=hashlib.sha256(evidence).hexdigest() if evidence else None
    now=datetime.now(timezone.utc).isoformat(timespec='seconds')
    # One conditional batch transaction keeps evidence and its parent atomic.
    statements=[
      ('''INSERT INTO sale_outcomes (submission_id,submitted_at,sale_date,price_usd,market,vehicle_json,reference_estimate_usd,reference_date,model,verification_status,evidence_sha256,deletion_key_hash) VALUES (?,?,?,?,?,?,?,?,?,'unverified',?,?)''',
       (sid,now,sold_on,price,market,json.dumps(f),estimate,'2021-05-01',str(model)[:160],digest,hashlib.sha256(key.encode()).hexdigest()))]
    if evidence:statements.append(('INSERT INTO sale_evidence VALUES (?,?,?)',(sid,evidence,'image/jpeg')))
    # Store's atomic batch checks each step before committing.
    store.atomic(statements)
    return {'submission_id':sid,'deletion_key':key,'verification_status':'unverified','submitted_at':now}


def delete_own(sid,key):
    from pricing.analytics import _store
    if not isinstance(sid,str) or len(sid)>40 or not isinstance(key,str) or not 16<=len(key)<=100:raise ValueError('Invalid deletion receipt.')
    store=_store()
    initialize(store)
    digest=hashlib.sha256(key.encode()).hexdigest()
    store.atomic([
      ('DELETE FROM sale_evidence WHERE submission_id IN (SELECT submission_id FROM sale_outcomes WHERE submission_id=? AND deletion_key_hash=?)',(sid,digest)),
      ('DELETE FROM sale_outcomes WHERE submission_id=? AND deletion_key_hash=?',(sid,digest))])


def purge(store):
    initialize(store)
    cutoff=(datetime.now(timezone.utc)-timedelta(days=90)).isoformat(timespec='seconds')
    store.atomic([
      ('DELETE FROM sale_evidence WHERE submission_id IN (SELECT submission_id FROM sale_outcomes WHERE submitted_at<?)',(cutoff,)),
      ('DELETE FROM sale_outcomes WHERE submitted_at<?',(cutoff,))])
