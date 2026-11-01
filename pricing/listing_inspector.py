"""Small, deterministic listing checks; no model loading or network requests."""
import math
import re
from difflib import SequenceMatcher

def normalized(value):
    return str(value or '').strip().lower()

def inspect(fields, description):
    findings=[]
    sentences=re.split(r'(?<=[.!?])\s+|\n+',description.strip())
    patterns={
        'transmission':{'automatic':r'\bautomatic\b','manual':r'\bmanual transmission\b|\bstick shift\b'},
        'title_status':{'salvage':r'\bsalvage title\b','rebuilt':r'\brebuilt title\b'},
    }
    for field,values in patterns.items():
        selected=normalized(fields.get(field))
        for value,pattern in values.items():
            evidence=next((s for s in sentences if re.search(pattern,s,re.I) and not re.search(r'\b(no|not|never|without)\b',s,re.I)),None)
            if evidence and selected not in ('','unknown',value):
                findings.append({'check':'Check '+field.replace('_',' '),'evidence':evidence,'action':f'The form says {selected}; this sentence mentions {value}. Confirm which is correct.'})
    for label,pattern,question in [
        ('Repair details',r'\b(repair|replaced|rebuild|rebuilt|transmission work)\b','What was repaired, when, and is an invoice available?'),
        ('Mechanical condition',r'\b(rattle|leak|overheat|misfire|check engine)\w*\b','Has the cause been diagnosed? Ask for the inspection report.'),
        ('Maintenance evidence',r'\b(service history|maintained|maintenance)\b','Which maintenance records are available, and when was the last service?'),
        ('Title documentation',r'\b(salvage|rebuilt|lien)\b','Can the seller provide the title and supporting documentation?')]:
        evidence=next((s for s in sentences if re.search(pattern,s,re.I)),None)
        if evidence:findings.append({'check':label,'evidence':evidence,'action':question})
    for field in ('odometer','title_status','transmission'):
        if normalized(fields.get(field)) in ('','unknown','none'):
            findings.append({'check':'Missing '+field.replace('_',' '),'evidence':'Not supplied in the form.','action':'Confirm this before comparing prices.'})
    return findings

def comparable_score(query, candidate):
    """Explicit matching heuristic, not trained relevance or a probability."""
    make=normalized(query.get('manufacturer'))
    if not make or make!=normalized(candidate.get('manufacturer')):return None
    try:
        years=abs(float(query['year'])-float(candidate['year']))
        miles=abs(float(query['odometer'])-float(candidate['odometer']))
        if not all(map(math.isfinite,(years,miles))) or years>5:return None
    except (ValueError,TypeError,KeyError):return None
    model=SequenceMatcher(None,normalized(query.get('model')),normalized(candidate.get('model'))).ratio()
    return 4*model-years/5-miles/100000

def acquisition_scenario(resale, purchase, repairs, transport, fees, holding):
    values=[resale,purchase,repairs,transport,fees,holding]
    if not all(math.isfinite(float(x)) and float(x)>=0 for x in values):raise ValueError('Amounts must be finite and nonnegative.')
    return float(resale)-sum(map(float,values[1:]))

def render(root):
    import gzip,json
    import pandas as pd
    import streamlit as st
    st.title('🔎 Listing Inspector')
    st.write('Read the evidence. Compare the right cars. Know what to ask.')
    st.caption('Historical research tools. Checks identify questions, not verified defects or fraud. Existing pricing models are unchanged.')
    with st.form('inspector_form'):
        a,b,c=st.columns(3)
        make=a.text_input('Make','Toyota');model=b.text_input('Model / trim','Corolla');year=c.number_input('Model year',1900,2030,2015)
        a,b,c=st.columns(3)
        miles=a.number_input('Mileage',0,2000000,95000)
        transmission=b.selectbox('Transmission',['unknown','automatic','manual'])
        title=c.selectbox('Title status',['unknown','clean','salvage','rebuilt','lien'])
        description=st.text_area('Paste the seller description',max_chars=8000)
        submitted=st.form_submit_button('Inspect this advert',type='primary')
    if submitted:st.session_state['inspector_result']=({'manufacturer':make,'model':model,'year':year,'odometer':miles,'transmission':transmission,'title_status':title},description)
    saved=st.session_state.get('inspector_result')
    tabs=st.tabs(['Advert checks & questions','Comparable cars','Purchase sandbox','Methods & limits'])
    with tabs[0]:
        if not saved:st.info('Paste an advert above to get started.')
        else:
            findings=inspect(*saved)
            if not findings:st.info('No rule-based flags found. This does not verify the advert or vehicle.')
            for f in findings:
                st.subheader(f['check']);st.text(f['evidence']);st.write(f['action'])
    with tabs[1]:
        st.subheader('Find the family resemblance')
        st.caption('SVD vehicle matching with separate year and mileage penalties. Similarity is not a valuation or a probability.')
        market=st.radio('Compare against', ['Historical asking adverts','Historical auction sales'],horizontal=True,key='inspector_market')
        if saved:
            from pricing.comparable_retrieval import ComparableIndex,reference_rows
            from pricing.comparable_semantics import attach,encode_query
            side='ask' if market.startswith('Historical asking') else 'bid'
            @st.cache_resource(max_entries=2,show_spinner=False)
            def get_index(folder,side,version):
                index=ComparableIndex(reference_rows(folder,side),32)
                semantic=attach(index,folder) if side=='ask' else False
                return index,semantic
            manifest=root/'data/comparable-minilm/manifest.json'
            version=manifest.stat().st_mtime_ns if manifest.exists() else 0
            try:
                with st.spinner('Finding comparable vehicles…'):
                    index,semantic=get_index(str(root),side,version)
                    vector=None
                    if semantic and saved[1].strip():
                        @st.cache_data(max_entries=16,show_spinner=False)
                        def query_vector(folder,text,version):return encode_query(folder,text)
                        try:
                            vector=query_vector(str(root),saved[1],version)
                        except Exception:
                            st.caption('Description matching is temporarily unavailable; vehicle matching continues.')
                    matches=index.search(saved[0],description_vector=vector)
                st.caption('MiniLM description similarity included.' if vector is not None else 'Vehicle attributes only. '+('Auction records contain no descriptions.' if side=='bid' else 'Description similarity is used when its offline index is installed and you supply text.'))
                if not matches:st.info('No sufficiently close comparisons in this limited historical sample. Try checking the make/model spelling or switch reference sets.')
                for rank,match in enumerate(matches,1):
                    car=match['vehicle']
                    with st.container(border=True):
                        st.markdown(f"### {rank}. {car['year']} {car['manufacturer'].title()} {car['model']}")
                        st.caption(f"{int(car['odometer']):,} miles · {car.get('sale_date',car.get('reference_period','Historical'))}")
                        if side=='bid':st.metric('Historical auction price · USD',f"{car['price']:,.0f}")
                        else:st.caption('Advertised price omitted from this reference sample.')
                        for reason in match['evidence']:st.text('• '+reason)
                        if car.get('description'):
                            with st.expander('Read historical seller description'):st.text(car['description'])
                st.caption('References: 400 manufacturer-balanced 2021 adverts or 4,000 sampled 2015 auction records. Samples may overlap model training and are not an independent accuracy benchmark. No current availability is implied.')
            except (ValueError,OSError,KeyError) as exc:
                st.warning('Comparisons are temporarily unavailable. Your advert checks and purchase sandbox remain available.')
        else:st.info('Inspect an advert above to find comparisons.')
    with tabs[2]:
        st.write('Enter your own assumptions. This is a contribution scenario before overhead and tax, not a predicted profit.')
        a,b=st.columns(2)
        resale=a.number_input('Assumed realized resale proceeds · USD',0.0,1000000.0,15000.0,100.0)
        purchase=b.number_input('Purchase price · USD',0.0,1000000.0,10000.0,100.0)
        repairs=a.number_input('Repair allowance · USD',0.0,1000000.0,1000.0,100.0)
        transport=b.number_input('Transport · USD',0.0,1000000.0,300.0,50.0)
        fees=a.number_input('Fees · USD',0.0,1000000.0,300.0,50.0)
        holding=b.number_input('Holding costs · USD',0.0,1000000.0,200.0,50.0)
        st.metric('Scenario contribution · USD',f'{acquisition_scenario(resale,purchase,repairs,transport,fees,holding):,.0f}')
        st.caption('Advertised asking prices and auction estimates describe different markets. Their difference is not dealer profit.')
    with tabs[3]:
        st.write('Checks are conservative phrase rules with quoted evidence. They may miss negation and complex language. Confirm all findings with the seller.')
        st.write('Comparable ranking uses 32-dimensional truncated SVD of sparse, weighted categorical attributes; model-name agreement, year and mileage remain explicit. Same-make, model-name and year-distance guardrails apply. Prices never enter retrieval. MiniLM description cosine similarity is an additional signal when available.')
        st.write('Future supervised reranking: LambdaRank, if reviewed relevance labels become available. Training requires judged comparable pairs or logged relevance feedback, query-group splits and a held-out NDCG assessment. SVD is unsupervised; no human relevance improvement is claimed. The engineering audit compares 32/64/128 dimensions and tests guardrails, latency and memory.')
        st.write('Regional effects, calibrated bargain probabilities, duplicate detection, information-premium experiments and batch review require separate data preparation and validation. They are not activated in this release.')
