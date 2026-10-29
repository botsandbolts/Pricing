"""Deterministic vehicle narrative + continuous LexRank. No neural model downloads."""
import html
import math
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

DETAILS = {'condition':'listed condition','fuel':'fuel','transmission':'transmission',
           'drive':'drivetrain','type':'body type','title_status':'listed title status',
           'cylinders':'cylinders','paint_color':'color','state':'US state'}
CRITICAL = re.compile(r'\b(check.engine|salvage|rebuilt|accident|damage|rust|leak|broken|not working|does not work|doesn.t work|needs? repair|overheat|warning light|flood|non.running)\b',re.I)
SERVICE = re.compile(r'\b(service|maintenance|replaced|new tires|new tyres|brakes|timing belt|oil change)\b',re.I)
AD = re.compile(r'\b(financing|apply now|call now|monthly payment|credit approval|visit our website)\b',re.I)


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


def lexrank(sentences):
    """Cosine TF-IDF sentence graph + damped centrality; dangling rows uniform."""
    n=len(sentences)
    if not n:return np.array([])
    try:x=TfidfVectorizer(stop_words='english',sublinear_tf=True).fit_transform(sentences)
    except ValueError:return np.ones(n)/n
    graph=(x@x.T).toarray();np.fill_diagonal(graph,0)
    degree=graph.sum(axis=1,keepdims=True)
    matrix=np.divide(graph,degree,out=np.full_like(graph,1/n),where=degree>0)
    score=np.ones(n)/n
    for _ in range(100):
        new=.15/n+.85*matrix.T@score
        if np.linalg.norm(new-score,ord=1)<1e-8:break
        score=new
    return new


def summarize(fields,description,max_sentences=3):
    facts,text=narrative(fields,description)
    # Preserve punctuation and complete negations. Newline-separated fragments are also candidates.
    sentences=[s.strip() for s in re.split(r'(?<=[.!?])\s+|[\r\n]+',text) if s.strip()]
    unique=[];seen=set()
    for s in sentences:
        key=re.sub(r'\W+',' ',s.lower()).strip()
        if key and key not in seen and (not AD.search(s) or CRITICAL.search(s)):
            unique.append(s);seen.add(key)
    # LexRank sees the complete structured story plus all eligible seller sentences.
    scores=lexrank([facts]+unique)[1:]
    priority=sorted(range(len(unique)),key=lambda i:(bool(CRITICAL.search(unique[i])),bool(SERVICE.search(unique[i])),scores[i]),reverse=True)
    chosen=[]
    for i in priority:
        if any(len(set(re.findall(r'\w+',unique[i].lower())) & set(re.findall(r'\w+',unique[j].lower()))) / max(1,len(set(re.findall(r'\w+',unique[i].lower())) | set(re.findall(r'\w+',unique[j].lower()))))>.85 for j in chosen):continue
        chosen.append(i)
        if len(chosen)>=max_sentences:break
    selected=[unique[i] for i in sorted(chosen)]
    return {'story':facts+ ('\nSeller description (unverified):\n'+text if text else ''),
            'facts':facts,'sentences':selected,'candidates':len(unique)}


def render_story(fields,description):
    import streamlit as st
    result=summarize(fields,description)
    st.markdown('#### 📖 The car’s story · quick read')
    st.write(result['facts'])
    if result['sentences']:
        st.caption('From the seller’s description · unverified claims')
        for sentence in result['sentences']:st.write(sentence)
    else:st.caption('No usable seller-description sentences were supplied.')
    st.caption('LexRank selects original sentences; defect and maintenance rules adjust priorities. No generative model. Important details may still be omitted; this does not change the price estimate.')
    with st.expander('Full supplied story · how this summary works'):
        st.text(result['story'])
        st.caption('Vehicle facts are rendered from your fields. TF-IDF cosine similarity builds a sentence graph; damped centrality ranks sentences. Disclosed defects and service history receive priority; common sales pitches and near-duplicates are suppressed. Selection retains original sentence order and wording. Seller claims are not verified inspections.')
