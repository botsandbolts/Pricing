"""Private operator review. Credentials come from environment, never command arguments."""
import argparse,base64,json,os
from pathlib import Path
from pricing.analytics import TursoStore
from pricing.sale_feedback import initialize


def values(result):
    def decode(v):
        if v['type']=='null':return None
        if v['type']=='blob':return base64.b64decode(v['base64'])
        return v.get('value')
    return [dict(zip([c['name'] for c in result['cols']],map(decode,row))) for row in result['rows']]


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('pending')
    review=sub.add_parser('review');review.add_argument('id');review.add_argument('status',choices=['confirmed','rejected']);review.add_argument('--reason',required=True)
    evidence=sub.add_parser('evidence');evidence.add_argument('id');evidence.add_argument('destination',type=Path)
    a=p.parse_args();s=TursoStore(os.environ['TURSO_DATABASE_URL'],os.environ['TURSO_AUTH_TOKEN']);initialize(s)
    if a.command=='pending':
        print(json.dumps(values(s.execute("SELECT submission_id,submitted_at,sale_date,price_usd,market,vehicle_json,verification_status,evidence_sha256 FROM sale_outcomes WHERE verification_status='unverified' ORDER BY submitted_at LIMIT 100")),indent=2))
    elif a.command=='review':
        reason=a.reason.strip()
        if not 10<=len(reason)<=500:p.error('Give a 10–500 character review rationale, without personal data.')
        s.execute('UPDATE sale_outcomes SET verification_status=?,review_reason=? WHERE submission_id=? AND verification_status=?',(a.status,reason,a.id,'unverified'))
        print('Review update processed. Confirmed means operator-reviewed, not guaranteed independent proof. No training was triggered.')
    else:
        rows=values(s.execute('SELECT content FROM sale_evidence WHERE submission_id=?',(a.id,)))
        if not rows:p.error('No evidence found.')
        fd=os.open(a.destination,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'wb') as f:f.write(rows[0]['content'])
        print('Private receipt exported. Delete the local copy after review; local exports are outside automated retention.')

if __name__=='__main__':main()
