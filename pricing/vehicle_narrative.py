"""Structured vehicle narrative shared with the Falconsai summarizer."""
import html
import math
import re

DETAILS = {'condition':'listed condition','fuel':'fuel','transmission':'transmission',
           'drive':'drivetrain','type':'body type','title_status':'listed title status',
           'cylinders':'cylinders','paint_color':'color','state':'US state'}


def clean(value):
    if value is None:return ''
    if isinstance(value,float) and not math.isfinite(value):return ''
    value=re.sub(r'\s+',' ',str(value)).strip()
    return '' if value.lower() in {'unknown','none','nan','null',''} else value[:160]


def narrative(fields, description):
    """All supplied specs form a story; seller wording stays verbatim after HTML cleanup."""
    identity=' '.join(clean(fields.get(k)) for k in ['year','manufacturer','model']).strip()
    miles=fields.get('odometer')
    try:mileage=f' with {float(miles):,.0f} miles' if miles is not None and math.isfinite(float(miles)) else ''
    except (ValueError,TypeError):mileage=''
    first=f'The listed vehicle is {identity or "unspecified"}{mileage}.'
    details=[f'{label}: {clean(fields.get(key))}' for key,label in DETAILS.items() if clean(fields.get(key))]
    facts=first+(' Supplied details — '+ '; '.join(details)+'.' if details else '')
    text=html.unescape(re.sub(r'<[^>]*>',' ',str(description or '')[:8000]))
    text=re.sub(r'https?://\S+|\b[^\s@]+@[^\s@]+\.[^\s@]+','[contact removed]',text)
    text=re.sub(r'(?<!\w)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\w)','[phone removed]',text)
    text=re.sub(r'\b[A-HJ-NPR-Z0-9]{17}\b','[VIN removed]',text)
    return facts, text.strip()

