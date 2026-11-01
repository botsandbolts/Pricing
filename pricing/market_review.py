"""Bounded appraisal and listing review helpers. No network or ML imports."""
import csv
import hashlib
import io
import math
import re
from collections import defaultdict

REQUIRED=('manufacturer','model','year','odometer')
TEXT_LIMIT=8000
MAX_ROWS=100
MAX_BYTES=2_000_000

def norm(value):
    return re.sub(r'\s+',' ',str(value or '').strip().lower())

def number(value):
    try:
        result=float(str(value).replace(',','').replace('$',''))
        return result if math.isfinite(result) else None
    except (ValueError,TypeError):return None

def parse_inventory(payload):
    if len(payload)>MAX_BYTES:raise ValueError('Use a CSV smaller than 2 MB.')
    try:reader=csv.DictReader(io.StringIO(payload.decode('utf-8-sig')))
    except UnicodeError as exc:raise ValueError('Use a UTF-8 CSV.') from exc
    if not reader.fieldnames or not set(REQUIRED)<=set(reader.fieldnames):raise ValueError('Required columns: '+', '.join(REQUIRED))
    rows=[]
    try:
        for i,row in enumerate(reader):
            if i>=MAX_ROWS:raise ValueError('Use at most 100 vehicles per file.')
            fields={k:str(row.get(k) or '').strip()[:160] for k in ('manufacturer','model','state','type','condition','fuel','transmission','drive','title_status','cylinders','paint_color')}
            fields['year']=number(row.get('year'));fields['odometer']=number(row.get('odometer'))
            errors=[]
            if fields['year'] is None or not 1900<=fields['year']<=2021 or fields['year']%1:errors.append('Model year must be an integer from 1900 to 2021')
            if fields['odometer'] is None or not 0<=fields['odometer']<=2000000:errors.append('Mileage must be between 0 and 2,000,000')
            if not fields['manufacturer'] or not fields['model']:errors.append('Make and model are required')
            if not errors:fields['year']=int(fields['year'])
            amount=number(row.get('price'))
            if str(row.get('price') or '').strip() and (amount is None or amount<=0):errors.append('Price must be positive USD')
            vin=re.sub(r'\s','',str(row.get('vin') or '').upper())
            vin=vin if re.fullmatch('[A-HJ-NPR-Z0-9]{17}',vin) else ''
            rows.append({'id':str(row.get('id') or f'vehicle-{i+1}')[:80], 'fields':fields,'description':str(row.get('description') or '')[:TEXT_LIMIT], 'price':amount, 'vin_hash':hashlib.sha256(vin.encode()).hexdigest() if vin else '', 'errors':errors})
    except UnicodeError as exc:raise ValueError('Use a UTF-8 CSV.') from exc
    return rows

def export_csv(rows):
    if not rows:return b''
    columns=list(dict.fromkeys(k for row in rows for k in row))
    out=io.StringIO();writer=csv.DictWriter(out,fieldnames=columns);writer.writeheader()
    for row in rows:
        safe={}
        for key,value in row.items():
            # Spreadsheet exports must not execute user-supplied formulas.
            if isinstance(value,str) and value.lstrip().startswith(('=','+','-','@')):value="'"+value
            safe[key]=value
        writer.writerow(safe)
    return out.getvalue().encode('utf-8')

def tokens(text):return set(re.findall(r'[a-z0-9]+',norm(text)))

def duplicate_pairs(rows):
    """VIN evidence first; text templates alone never establish identity."""
    if len(rows)>MAX_ROWS:raise ValueError('Duplicate review is limited to 100 vehicles.')
    prepared=[tokens(r['description']) for r in rows]
    result=[]
    for i,a in enumerate(rows):
        for j in range(i+1,len(rows)):
            b=rows[j];same_vin=bool(a['vin_hash'] and a['vin_hash']==b['vin_hash'])
            fa,fb=a['fields'],b['fields']
            same_car=all(norm(fa.get(k))==norm(fb.get(k)) and norm(fa.get(k)) for k in ('manufacturer','model','year'))
            ma,mb=number(fa.get('odometer')),number(fb.get('odometer'))
            close_miles=ma is not None and mb is not None and abs(ma-mb)<=10000
            union=prepared[i]|prepared[j]
            similarity=len(prepared[i]&prepared[j])/len(union) if union else 0
            near=bool(same_car and close_miles and min(len(a['description']),len(b['description']))>=80 and similarity>=.85)
            if not (same_vin or near):continue
            conflicts=[]
            for key in ('manufacturer','model','year','odometer'):
                if norm(fa.get(key))!=norm(fb.get(key)):conflicts.append(key)
            if a['price'] is not None and b['price'] is not None and a['price']!=b['price']:conflicts.append('price')
            result.append({'first':a['id'],'second':b['id'],'evidence':'Matching supplied VIN' if same_vin else 'Similar text + matching make/model/year + close mileage','text_overlap_pct':round(100*similarity,1),'different_fields':', '.join(conflicts) or 'None found','action':'Confirm listing dates and vehicle identity; differences may be legitimate.'})
    return result

def review_estimate(receipt,listed_price):
    """Decision labels, never a calibrated probability of a bargain."""
    if 'error' in receipt:return {'priority':'Unavailable','reason':receipt['error']}
    estimate=number(receipt.get('price'));lo=number(receipt.get('interval_low'));hi=number(receipt.get('interval_high'))
    if estimate is None or estimate<=0:return {'priority':'Unavailable','reason':'Invalid valuation'}
    listed=number(listed_price)
    width=(hi-lo)/estimate if lo is not None and hi is not None else None
    unknown=list(receipt.get('unknown_categories',[]))
    rare=int(receipt.get('model_support',0))<12
    reasons=[]
    if rare:reasons.append('Rare or unseen model in calibration')
    if unknown:reasons.append('Unseen inputs: '+', '.join(unknown))
    if width is not None and width>1:reasons.append('Wide historical uncertainty range')
    if receipt.get('analogue'):reasons.append('Auction generation extrapolation')
    below=listed is not None and lo is not None and listed<lo
    if below:reasons.append('Listed below the date-adjusted historical range; verify condition and scope')
    return {'priority':'Check evidence first' if below and reasons[:-1] else 'Price worth investigating' if below else 'Specialist / information review' if rare or unknown or width is not None and width>1 else 'Routine review',
            'reason':'; '.join(reasons) or 'No rule-based escalation; vehicle remains unverified',
            'asking_estimate_usd':round(estimate,2),'listed_price_usd':listed,
            'difference_from_model_usd':round(estimate-listed,2) if listed is not None else None,
            'historical_range_low_usd':lo,'historical_range_high_usd':hi,
            'range_width_relative':round(width,3) if width is not None else None,
            'model_support':int(receipt.get('model_support',0))}

def batch_review(rows,evaluate,date,allow):
    """Sequential, explicit-action evaluation with row-level failure isolation."""
    if len(rows)>25:raise ValueError('Price at most 25 vehicles per batch; duplicate checks support 100.')
    output=[]
    for row in rows:
        base={'id':row['id'],'valuation_date':str(date)}
        if row['errors']:
            output.append({**base,'priority':'Invalid input','reason':'; '.join(row['errors'])});continue
        try:
            results=evaluate({**row['fields'],'text':row['description']},date,allow)
            reviewed=review_estimate(results.get('ask',{'error':'Asking estimate unavailable'}),row['price'])
            bid=results.get('bid',{})
            output.append({**base,**reviewed,'auction_estimate_usd':bid.get('price'),'auction_status':bid.get('error','Experimental date-adjusted estimate')})
        except (ValueError,OSError,KeyError,MemoryError,RuntimeError) as exc:
            output.append({**base,'priority':'Unavailable','reason':type(exc).__name__+'; other vehicles continue'})
    order={'Check evidence first':0,'Price worth investigating':1,'Specialist / information review':2,'Routine review':3,'Invalid input':4,'Unavailable':5}
    return sorted(output,key=lambda r:(order.get(r['priority'],6),r['id']))
