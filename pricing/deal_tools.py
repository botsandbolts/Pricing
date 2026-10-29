"""Bounded, privacy-conscious public sharing and historical decision tools."""
import base64
import hashlib
import gzip
from functools import lru_cache
import io
import json
import math
import re
import textwrap
from datetime import date
from pathlib import Path
from urllib.parse import urlencode

APP_URL = 'https://pricing-5a4k2zenhw2nte9xpshxs3.streamlit.app/'
REFERENCE_DATE = '2021-05-01'
SHARE_FIELDS = ('year','manufacturer','model','odometer','state','type','fuel','transmission','condition','drive','title_status','cylinders','paint_color')
MAKES = ('alfa-romeo','aston-martin','audi','bmw','buick','cadillac','chevrolet','chrysler','dodge','fiat','ford','gmc','honda','hyundai','infiniti','jaguar','jeep','kia','land rover','lexus','lincoln','mazda','mercedes-benz','mercury','mini','mitsubishi','nissan','pontiac','porsche','ram','rover','saab','saturn','scion','subaru','tesla','toyota','volkswagen','volvo')


def safe_fields(fields):
    out = {}
    for key in SHARE_FIELDS:
        v = fields.get(key)
        if v is None: continue
        if key in ('year','odometer'):
            try: v = float(v)
            except (ValueError,TypeError): continue
            if not math.isfinite(v): continue
            if key == 'year' and (v != int(v) or not 1900 <= v <= 2021): continue
            if key == 'odometer' and not 0 <= v <= 2000000: continue
            out[key] = int(v)
        else:
            v = str(v).strip()[:80]
            if re.search(r'@|https?://|www\.|\b[A-HJ-NPR-Z0-9]{17}\b|\+?\d[\d ()-]{7,}',v,re.I): continue
            if v: out[key] = v
    return out


def car_title(fields):
    f = safe_fields(fields)
    return ' '.join(str(f.get(k,'')) for k in ('year','manufacturer','model')).strip() or 'Your mystery machine'


def share_link(cars):
    if not 1 <= len(cars) <= 3: raise ValueError('Share one to three cars.')
    payload = {'v':1,'cars':[safe_fields(c.get('fields',c)) for c in cars]}
    raw = json.dumps(payload,separators=(',',':'),ensure_ascii=True).encode()
    token = base64.urlsafe_b64encode(raw).decode().rstrip('=')
    if len(token)>6000: raise ValueError('Vehicle details are too long for a share link.')
    return APP_URL+'?'+urlencode({'garage':token})


def read_share(token):
    if not isinstance(token,str) or not 1 <= len(token) <= 6000: raise ValueError('Invalid share link size.')
    try:
        raw = base64.b64decode(token+'='*(-len(token)%4),altchars=b'-_',validate=True)
        payload = json.loads(raw)
        if not isinstance(payload,dict) or payload.get('v')!=1: raise ValueError()
        cars=payload['cars']
        if not isinstance(cars,list) or not 1<=len(cars)<=3 or not all(isinstance(c,dict) for c in cars): raise ValueError()
        cleaned=[safe_fields(c) for c in cars]
        if not all('year' in c and 'manufacturer' in c and 'model' in c for c in cleaned): raise ValueError()
        return cleaned
    except (KeyError,ValueError,TypeError,UnicodeError):
        raise ValueError('This share link is invalid or unsupported.') from None


def caption(fields,price,tone='Playful'):
    price=float(price)
    if not math.isfinite(price) or price<=0: raise ValueError('Positive finite estimate required.')
    hook = {'Playful':'My car has met the maths. The maths has questions.',
            'Professional':'Transparent car pricing, with the assumptions included.',
            'Challenge':'What would YOU pay? Guess before opening the garage.'}[tone]
    return f'{hook}\n{car_title(fields)} · ${price:,.0f} historical asking estimate.\nUS market reference: May 2021. Not a current sale guarantee.\n#CarPriceStudio #UsedCars #MachineLearning'


def social_links(url,text):
    if not url.startswith(APP_URL): raise ValueError('Only studio links can be shared.')
    return {'WhatsApp':'https://wa.me/?'+urlencode({'text':text+'\n'+url}),
            'LinkedIn':'https://www.linkedin.com/sharing/share-offsite/?'+urlencode({'url':url}),
            'Facebook':'https://www.facebook.com/sharer/sharer.php?'+urlencode({'u':url}),
            'Reddit':'https://www.reddit.com/submit?'+urlencode({'url':url,'title':text.splitlines()[0]}),
            'Email':'mailto:?'+urlencode({'subject':'Car Price Studio · what would you pay?','body':text+'\n'+url})}


def redact_description(text):
    text=re.sub(r'\b[A-HJ-NPR-Z0-9]{17}\b','[VIN removed]',str(text),flags=re.I)
    text=re.sub(r'https?://\S+|www\.\S+|\b\S+@\S+\b','[contact removed]',text,flags=re.I)
    text=re.sub(r'(?<!\d)(?:\+?1[ .-]*)?(?:\(\d{3}\)|\d{3})[ .-]*\d{3}[ .-]*\d{4}','[phone removed]',text)
    return text[:8000]


def parse_advert(text):
    text=str(text)[:16000]; fields={}; price=None
    match=re.search(r'\b(19\d{2}|20[01]\d|202[01])\b',text)
    if match: fields['year']=int(match[0])
    match=re.search(r'(?i)\b([\d,]+(?:\.\d+)?)\s*(k)?\s*(miles?|mi\b|km\b|kilomet(?:er|re)s?)',text)
    if match:
        n=float(match[1].replace(',',''))*(1000 if match[2] else 1)
        fields['odometer']=round(n/1.609344 if match[3].lower().startswith(('km','kilo')) else n)
    make_match=None
    for make in sorted(MAKES,key=len,reverse=True):
        m=re.search(r'\b'+re.escape(make).replace(r'\-',r'[- ]')+r'\b',text,re.I)
        if m: make_match=(make,m);break
    if make_match:
        fields['manufacturer']=make_match[0]
        tail=text[make_match[1].end():].splitlines()[0] if text[make_match[1].end():].splitlines() else ''
        tail=re.split(r'\$|\b\d[\d,]*\s*(?:miles|mi|km)\b|\s[|·]',tail,flags=re.I)[0]
        fields['model']=' '.join(tail.strip(' :-').split()[:5])[:80]
    m=re.search(r'(?i)(?:asking\s*(?:price)?|price)\s*[:=]?\s*\$?([\d,]+(?:\.\d{1,2})?)|\$\s*([\d,]+(?:\.\d{1,2})?)',text)
    if m:
        price=float((m[1] or m[2]).replace(',',''))
        if not 1<=price<=10000000:price=None
    for key,choices in {'transmission':('automatic','manual'),'fuel':('diesel','hybrid','electric','gas'),'drive':('4wd','fwd','rwd'),'type':('suv','sedan','pickup','truck','hatchback','coupe','wagon','van','convertible'),'title_status':('salvage','rebuilt','clean')}.items():
        for v in choices:
            if re.search(r'\b'+v+r'\b',text,re.I):fields[key]='SUV' if v=='suv' else v;break
    return {'fields':safe_fields(fields),'seller_price':price,'text':redact_description(text)}


def ocr_image(content):
    from PIL import Image,ImageOps,UnidentifiedImageError
    import pytesseract
    if len(content)>8*1024*1024:raise ValueError('Use an image smaller than 8 MB.')
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.width*image.height>16000000:raise ValueError('Use an image below 16 megapixels.')
            image=ImageOps.exif_transpose(image).convert('RGB');image.thumbnail((2200,3000))
            return pytesseract.image_to_string(image,config='--psm 6',timeout=15)[:16000]
    except (UnidentifiedImageError,OSError):raise ValueError('Upload a readable PNG or JPEG screenshot.') from None


@lru_cache(maxsize=1)
def reference_cases():
    path=Path(__file__).resolve().parents[1]/'data/deal-reference.json.gz'
    return json.loads(gzip.decompress(path.read_bytes()))


def comparables(fields,cases,count=6):
    f=safe_fields(fields);make=str(f.get('manufacturer','')).casefold();model=str(f.get('model','')).casefold()
    if not make or 'year' not in f:return []
    out=[]
    for c in cases:
        if str(c['manufacturer']).casefold()!=make:continue
        delta=abs(c['year']-f['year'])
        if delta>3:continue
        same=str(c['model']).casefold()==model
        if model and not same and model not in str(c['model']).casefold() and str(c['model']).casefold() not in model:continue
        miles=abs(c['odometer']-f.get('odometer',c['odometer']))
        score=delta+miles/25000+(0 if same else 2)
        out.append({**c,'year_difference':delta,'mileage_difference':miles,'match_distance':round(score,3)})
    return sorted(out,key=lambda c:(c['match_distance'],c['id']))[:count]


def daily_case(cases,day=None):
    if not cases:raise ValueError('No historical cases available.')
    day=day or date.today()
    return cases[int(hashlib.sha256(('deal-daily-v1:'+day.isoformat()).encode()).hexdigest(),16)%len(cases)]


def font(size):
    from PIL import ImageFont
    for path in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf'):
        if Path(path).exists():return ImageFont.truetype(path,size)
    return ImageFont.load_default(size=size)


def card_png(fields,price,url,portrait=False):
    from PIL import Image,ImageDraw
    import qrcode
    price=float(price)
    if not math.isfinite(price) or not 0<price<=10000000:raise ValueError('Invalid card estimate.')
    w,h=(1080,1350) if portrait else (1200,630)
    image=Image.new('RGB',(w,h),'#241d35');draw=ImageDraw.Draw(image)
    draw.ellipse((w-440,-200,w+190,400),fill='#463580')
    draw.rounded_rectangle((45,40,360,86),radius=22,fill='#d9fb77')
    draw.text((63,48),'CAR PRICE STUDIO',font=font(24),fill='#241d35')
    y=130
    for line in textwrap.wrap(car_title(fields),35)[:3]:
        draw.text((55,y),line,font=font(42),fill='white');y+=54
    draw.text((55,y+12),'HISTORICAL ASKING ESTIMATE',font=font(22),fill='#cbbef1')
    draw.text((50,y+58),f'${price:,.0f}',font=font(100),fill='#d9fb77')
    if portrait:
        y+=230
        for line in ['Serious maths.','Zero tyre-kicking.']:
            draw.text((55,y),line,font=font(54),fill='#ff997d');y+=68
    bottom=h-145
    draw.text((55,bottom),'US market reference: May 2021',font=font(24),fill='white')
    draw.text((55,bottom+40),'Estimate, not a current sale guarantee.',font=font(20),fill='#cbbef1')
    draw.text((55,bottom+75),('Scan to open vehicle details.' if '?garage=' in url else 'Scan to try Car Price Studio.'),font=font(20),fill='#cbbef1')
    qr=qrcode.QRCode(border=4,error_correction=qrcode.constants.ERROR_CORRECT_M);qr.add_data(url);qr.make(fit=True)
    qrimage=qr.make_image(fill_color='#241d35',back_color='white').convert('RGB')
    # Preserve integral QR modules for reliable scanning.
    side=qrimage.width;target=230 if not portrait else 330
    factor=max(1,target//(len(qr.get_matrix())))
    matrix=qr.get_matrix();q=Image.new('RGB',(len(matrix)*factor,len(matrix)*factor),'white');qd=ImageDraw.Draw(q)
    for yy,row in enumerate(matrix):
        for xx,value in enumerate(row):
            if value:qd.rectangle((xx*factor,yy*factor,(xx+1)*factor-1,(yy+1)*factor-1),fill='#241d35')
    image.paste(q,(w-q.width-48,h-q.height-210 if portrait else 120))
    out=io.BytesIO();image.save(out,format='PNG');return out.getvalue()
