"""Explicit sale evidence, USD targets, pre-sale text, vehicle-disjoint cohorts."""
import argparse,hashlib,json,re,zipfile,csv,io
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import requests
from carbids.common import clean
from carbids.tabular import CATEGORICAL

SOURCE_URL='https://www.kaggle.com/datasets/tunguz/used-car-auction-prices'
DOWNLOAD_URL='https://www.kaggle.com/api/v1/datasets/download/tunguz/used-car-auction-prices'
TARGET='platform_reported_sale_price'
SOLD={'sold','sold_after_auction','completed_sale'}
HOSTS={'bringatrailer':{'bringatrailer.com','www.bringatrailer.com'},'carsandbids':{'carsandbids.com','www.carsandbids.com'},'wholesale_2015':{'www.kaggle.com','kaggle.com'}}

def download(destination):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists() and zipfile.is_zipfile(destination):return destination
    temporary=destination.with_suffix('.part')
    with requests.get(DOWNLOAD_URL,stream=True,timeout=(30,180)) as response:
        response.raise_for_status()
        with temporary.open('wb') as f:
            for chunk in response.iter_content(1024*1024):f.write(chunk)
    if not zipfile.is_zipfile(temporary):
        temporary.unlink(missing_ok=True)
        raise ValueError('Kaggle returned a non-ZIP response; attach the downloaded dataset instead')
    temporary.replace(destination);return destination

def valid_vin(value):
    text=str(value or '').strip().upper()
    return text if re.fullmatch('[A-HJ-NPR-Z0-9]{17}',text) else None

def assign_phases(d):
    d=d.sort_values(['date','row_id']).reset_index(drop=True)
    n=len(d)
    if n<30:raise ValueError('At least 30 eligible distinct vehicles required')
    # Keep entire date groups together. No same-time lots straddle partitions.
    dates=pd.to_datetime(d.date,utc=True)
    first=dates.iloc[int(n*.7)];second=dates.iloc[int(n*.85)]
    d['phase']=np.where(dates<first,'train',np.where(dates<second,'validation','test'))
    if set(d.phase)!={'train','validation','test'}:raise ValueError('Need sufficient distinct sale dates for chronological evaluation')
    return d

def normalize(frame,source_name,wholesale=False):
    kept=[];reasons={}
    def reject(reason):reasons[reason]=reasons.get(reason,0)+1
    for i,row in enumerate(frame.to_dict('records')):
        source=source_name if wholesale else str(row.get('source','')).lower().strip()
        status='sold' if wholesale else str(row.get('status','')).lower().strip().replace('-','_')
        kind=TARGET if wholesale else row.get('target_kind')
        currency='USD' if wholesale else str(row.get('currency','')).upper().strip()
        url=SOURCE_URL if wholesale else str(row.get('source_url',''))
        evidence='Provider dataset column sellingprice' if wholesale else str(row.get('price_evidence','')).strip()
        if source not in HOSTS or urlparse(url).hostname not in HOSTS.get(source,set()):reject('unrecognized_source_or_url');continue
        if status not in SOLD:reject('not_explicitly_sold');continue
        if kind!=TARGET or not evidence or evidence.lower() in {'nan','none'}:reject('missing_sale_price_evidence');continue
        if currency!='USD':reject('non_usd');continue
        price=pd.to_numeric(row.get('sellingprice') if wholesale else row.get('sale_price'),errors='coerce')
        if not np.isfinite(price) or not 1<=price<=10000000:reject('invalid_sale_price');continue
        if not wholesale:
            match=re.search(r'(?i)\bsold(?:\s+after\s+auction)?\s+for\s+(?:USD\s*)?\$?\s*([\d,]+(?:\.\d{1,2})?)',evidence)
            if not match or abs(float(match.group(1).replace(',',''))-price)>.01:reject('sale_evidence_amount_mismatch');continue
        date=pd.to_datetime(row.get('saledate') if wholesale else row.get('sale_date'),utc=True,errors='coerce')
        if pd.isna(date):reject('invalid_sale_date');continue
        year=pd.to_numeric(row.get('year'),errors='coerce')
        mileage=pd.to_numeric(row.get('odometer') if wholesale else row.get('mileage_miles'),errors='coerce')
        if not np.isfinite(year) or not 1886<=year<=date.year+1:reject('invalid_model_year');continue
        if np.isfinite(mileage) and not 0<=mileage<=2000000:reject('invalid_mileage');continue
        text=''
        if not wholesale:
            # Never consume result lines, auction comments, bid history, or generic HTML descriptions.
            supplied=str(row.get('description_pre_sale','')).strip()
            if supplied.lower() in {'nan','none'}:supplied=''
            if supplied:
                if row.get('text_scope')!='pre_sale':reject('unattested_text_scope');continue
                asof=pd.to_datetime(row.get('text_as_of'),utc=True,errors='coerce')
                if pd.isna(asof) or asof>date:reject('post_sale_or_undated_text');continue
                text=clean(supplied)[:8000]
        raw_id=str(row.get('auction_id',f'row-{i}'))
        rid=hashlib.sha256(f'{source}:{raw_id}'.encode()).hexdigest()[:24]
        vin=valid_vin(row.get('vin'))
        vehicle='vin:'+vin if vin else 'auction:'+rid
        if wholesale and not vin:reject('missing_valid_vin');continue
        # Require VIN for rich auction imports too: no silent relisting leakage.
        if not vin:reject('missing_valid_vin');continue
        item={c:None for c in CATEGORICAL}
        item.update(year=int(year),odometer=float(mileage) if np.isfinite(mileage) else None,
            manufacturer=row.get('make'),model=row.get('model'),condition=row.get('condition'),
            transmission=row.get('transmission'),type=row.get('body') if wholesale else row.get('body_type'),
            paint_color=row.get('color') if wholesale else row.get('paint_color'),state=row.get('state'),source=source)
        for c in ['fuel','cylinders','title_status','drive']:
            if not wholesale:item[c]=row.get(c)
        item.update(row_id=rid,vehicle_group=vehicle,price=float(price),date=date.isoformat(),text=text,
            source_url=url,auction_id=raw_id,target_kind=TARGET,currency='USD',status=status,
            price_evidence=evidence,price_basis='reported_sale_excluding_unspecified_fees',
            text_scope='pre_sale' if text else 'none',text_as_of=asof.isoformat() if text else None)
        kept.append(item)
    return pd.DataFrame(kept),reasons

def prepare(path,output,source='wholesale_2015'):
    path=Path(path);output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    malformed=[]
    read_csv=lambda f:pd.read_csv(f,engine='python',on_bad_lines=lambda row:malformed.append(row))
    if source=='wholesale_2015':
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as z:
                member=next(n for n in z.namelist() if n.endswith('.csv'))
                with z.open(member) as f:raw=read_csv(f)
        else:raw=read_csv(path)
        raw['auction_id']=np.arange(len(raw)).astype(str)
        raw['saledate']=pd.to_datetime(raw.saledate.astype(str).str.replace(r' \([^)]*\)$','',regex=True),format='%a %b %d %Y %H:%M:%S GMT%z',utc=True,errors='coerce')
        grade=pd.to_numeric(raw.condition,errors='coerce')
        raw['condition']=grade.where(grade<=5,grade/10).where(grade.between(1,50))
    else:raw=pd.read_json(path,lines=True) if path.suffix=='.jsonl' else pd.read_csv(path,low_memory=False)
    logical_rows=len(raw)+len(malformed)
    if source=='wholesale_2015':
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as z,z.open(next(n for n in z.namelist() if n.endswith('.csv'))) as f:
                reader=csv.reader(io.TextIOWrapper(f));next(reader);logical_rows=sum(1 for _ in reader)
        else:
            with path.open() as f:
                reader=csv.reader(f);next(reader);logical_rows=sum(1 for _ in reader)
    canonical,rejected=normalize(raw,source,wholesale=source=='wholesale_2015')
    if canonical.empty:raise ValueError('No rows satisfy explicit reported-sale rules: '+str(rejected))
    before=len(canonical)
    canonical=canonical.sort_values(['date','row_id']).drop_duplicates('row_id',keep='last').drop_duplicates('vehicle_group',keep='last')
    # Deduplicate repeated rich descriptions before splitting too.
    text_hash=canonical.text.map(lambda t:hashlib.sha256(t.encode()).hexdigest() if t else '')
    duplicate=text_hash.ne('')&text_hash.duplicated(keep='last');canonical=canonical.loc[~duplicate]
    canonical=assign_phases(canonical)
    canonical.to_json(output,orient='records',lines=True)
    report={'target_kind':TARGET,'settlement_verified':False,'currency':'USD','raw_rows':logical_rows,'malformed_csv_rows':len(malformed),'additional_unparsed_csv_rows':logical_rows-len(raw)-len(malformed),
        'eligible_rows':len(canonical),'duplicates_removed':before-len(canonical),'rejections':rejected,
        'sources':sorted(canonical.source.unique().tolist()),'counts':canonical.phase.value_counts().to_dict(),
        'rich_text_rows':int(canonical.text.str.len().ge(80).sum()),'fraction':1.0,
        'split':'Chronological dates, one canonical sale per VIN and duplicate description removed before splitting',
        'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    output.with_suffix('.meta.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    return report

def load_cohort(path,require_text=False):
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(f'Missing {path}; run python -m carbids.data first')
    d=pd.read_json(path,lines=True)
    required={'row_id','vehicle_group','price','date','phase','target_kind','currency','status','price_evidence','source_url','text','text_scope'}|set(CATEGORICAL)
    if not required.issubset(d):raise ValueError('Missing audited cohort fields: '+str(required-set(d)))
    if not d.target_kind.eq(TARGET).all() or not d.currency.eq('USD').all() or not d.status.isin(SOLD).all():raise ValueError('Non-sale or unsupported currency targets detected')
    if d.vehicle_group.duplicated().any() or d.row_id.duplicated().any():raise ValueError('Vehicle/auction duplication could leak across folds')
    if not d.price.between(1,10000000).all():raise ValueError('Invalid sale price')
    if 'auction_id' in d and d.auction_id.astype(str).str.startswith('EXAMPLE-').any():raise ValueError('Schema fixtures are not real sales and cannot train a production model')
    if d.price_evidence.fillna('').astype(str).str.strip().eq('').any():raise ValueError('Missing price provenance')
    if 'acs_available_at' in d:
        released=pd.to_datetime(d.acs_available_at,utc=True,errors='coerce')
        sales=pd.to_datetime(d.date,utc=True,errors='coerce')
        if ((released>=sales)&released.notna()).any():raise ValueError('Census context was not available before sale date')
    if require_text and (d.text.fillna('').str.len().lt(80).any() or not d.text_scope.eq('pre_sale').all()):raise ValueError('Enhanced mode needs at least 80 cleaned pre-sale characters for every row. The wholesale dataset has no descriptions; use tabular mode.')
    return d

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',default='data/auction_sales.zip');p.add_argument('--output',default='data/cohort.jsonl');p.add_argument('--source',choices=['wholesale_2015','auction_export'],default='wholesale_2015');p.add_argument('--download',action='store_true');a=p.parse_args()
    if a.download:download(a.input)
    prepare(a.input,a.output,a.source)
