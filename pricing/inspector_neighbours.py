"""Three separate SVDs; detailed neighbours in one similarity-sorted list."""
from datetime import date
import json,html
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from pricing.comparable_retrieval import ComparableIndex,reference_rows,clean
from pricing.reference_prices import feature_prices,adjust_reference
from pricing.pricing_comparison import selected_receipt

@st.cache_resource(max_entries=1,show_spinner=False)
def joint_imputer(root,version):
    from pricing.description_price_imputation import load
    return load(root,reference_rows(root,'ask'),reference_rows(root,'bid'))

@st.cache_data(max_entries=16,show_spinner=False)
def query_vector(root,text,version):
    from pricing.comparable_semantics import encode_query
    return encode_query(root,text)

def price_on(row,side,price,source,target,transport):
    if price is None:return np.nan
    return feature_prices([{**row,'price':price,'observed_date':str(source)}],side,target,transport)[0]

def description_to_date(rows,values,target,transport,mileage):
    synthetic=[dict(r,price=float(v),observed_date='2021-05-01') for r,v in zip(rows,values)]
    return feature_prices(synthetic,'ask',target,transport,mileage)

@st.cache_resource(max_entries=3,show_spinner=False)
def build_index(root,kind,version,target,mileage,query_price_key=None):
    from pricing.comparable_semantics import attach
    from pricing.retrieval_imputation import load as load_imputer
    root=Path(root);rows=reference_rows(root,kind);imputer=load_imputer(root,kind,rows)
    auxiliary=None;description_prices=None;basis=None
    if kind!='hybrid':
        from pricing.price_context_imputation import load as load_context
        contextual,basis=load_context(root,kind,rows)
        if contextual is not None:imputer=contextual
    if kind=='bid':
        from pricing.current_value import current_transport
        try:
            transport,_=current_transport(root);auxiliary=joint_imputer(str(root),version)
            if auxiliary is not None:
                observed=feature_prices(rows,'bid',date(2021,5,1),transport)
                signal,_=auxiliary.transform(rows,observed)
                description_prices=description_to_date(rows,signal,pd.Timestamp(target).date(),transport,mileage)
        except (ValueError,OSError,KeyError,TypeError):pass
    # Observed advert/auction prices are NEVER passed into the SVD representation.
    index=ComparableIndex(rows,32,imputer=imputer,description_prices=description_prices)
    semantic=attach(index,root) if kind=='ask' else False
    return index,semantic,auxiliary,basis

def asset_version(root):
    paths=['models/xgboost-hierarchy/ask/minilm-heads.json','data/comparable-minilm/manifest.json','data/retrieval-imputation/ask.json','data/retrieval-imputation/bid.json','data/retrieval-imputation/hybrid.json','data/retrieval-imputation/ask-price-context.json','data/retrieval-imputation/bid-price-context.json','data/retrieval-imputation/description-price.json','data/comparable-catalog/ask.json.gz','data/deal-reference.json.gz']
    return tuple((p,(root/p).stat().st_mtime_ns,(root/p).stat().st_size) for p in paths if (root/p).exists())

def ordered_matches(groups):
    """Hybrid shares one geometry, so its two market lists can be sorted together."""
    return sorted([(side,m) for side,items in groups.items() for m in items],key=lambda item:(-item[1]['faiss_cosine_similarity'],item[0],str(item[1]['vehicle'].get('id',''))))

def differences(query,car):
    out=[]
    for key,label in [('manufacturer','Make'),('model','Model / trim'),('type','Body'),('fuel','Fuel'),('transmission','Transmission'),('drive','Drive'),('title_status','Title'),('condition','Condition'),('cylinders','Cylinders'),('paint_color','Colour'),('state','State')]:
        a,b=clean(query.get(key)),clean(car.get(key))
        if a=='__missing__' and b=='__missing__':continue
        left='not supplied' if a=='__missing__' else str(query[key]);right='not supplied' if b=='__missing__' else str(car[key])
        out.append(dict(Attribute=label,**{'Your car':left,'Neighbour':right,'Difference':'Same' if a==b else 'Different' if '__missing__' not in (a,b) else 'Not comparable'}))
    return out

def render(root,saved,own,valuation_date,pricing_view):
    root=Path(root)
    st.subheader('Your car’s neighbours · the maths brought name tags')
    mode=st.radio('Matching approach',['Ask · asking adverts','Bid · auction records','Hybrid · vehicle attributes only'],index=2,horizontal=True,key='inspector_svd_mode',persist_state='session')
    kind='ask' if mode.startswith('Ask') else 'bid' if mode.startswith('Bid') else 'hybrid'
    if kind=='hybrid':st.caption('Combined vehicle-only SVD. No prices, descriptions, text embeddings or description-price signals, including in imputation. Five asking and five auction neighbours, mixed by similarity.')
    elif kind=='ask':st.caption('Asking-only SVD: completed vehicle attributes plus optional MiniLM text similarity. Adjusted advert prices assist imputation, then are dropped. No scalar price enters SVD.')
    else:st.caption('Auction-only SVD: completed vehicle attributes and the imputed MiniLM description-price signal. Adjusted auction prices assist imputation, then are dropped. No auction description is invented.')
    if not saved:st.info('Choose an example or enter your car, then press Price my car & find neighbours.');return
    fields,text=saved;normalize=st.checkbox('Compare prices at my mileage',value=True,key='inspector_normalize_mileage') and fields.get('odometer') is not None
    mileage=fields.get('odometer') if normalize else None;version=asset_version(root);target=str(valuation_date)
    receipts=own.get('results',{}) if own and own.get('date')==target else {}
    selected={s:selected_receipt(receipts.get(s,{'error':'Unavailable'}),pricing_view.startswith('Bayesian')) for s in ['ask','bid']}
    try:
        from pricing.current_value import current_transport
        try:transport,_=current_transport(root)
        except (ValueError,OSError,KeyError):transport=None
        with st.spinner('Finding nearest neighbours with FAISS cosine search…'):
            index,semantic,auxiliary,basis=build_index(str(root),kind,version,target,mileage)
            vector=query_vector(str(root),text,version) if kind=='ask' and semantic and text.strip() else None
            imputation_price=np.nan;description_feature=np.nan
            if basis and transport is not None:
                imputation_price=price_on(fields,kind,selected[kind].get('price'),valuation_date,pd.Timestamp(basis).date(),transport)
            if kind=='bid' and auxiliary is not None and transport is not None:
                observed=np.nan
                if text.strip():
                    from pricing.minilm_signal import mean_price
                    vector=query_vector(str(root),text,version)
                    heads=json.loads((root/'models/xgboost-hierarchy/ask/minilm-heads.json').read_text())['heads']
                    observed=float(mean_price(heads,np.asarray(vector).reshape(1,-1))[0])
                context=price_on(fields,'bid',selected['bid'].get('price'),valuation_date,date(2021,5,1),transport)
                signals,_=auxiliary.transform([fields],np.array([context]),np.array([observed]))
                description_feature=description_to_date([fields],signals,valuation_date,transport,mileage)[0]
            query={**fields,'_imputation_price':imputation_price,'_retrieval_description_price':description_feature}
            sides=['ask','bid'] if kind=='hybrid' else [kind]
            groups={s:index.search(query,k=5,description_vector=vector if kind=='ask' else None,exclude_id=st.session_state.get('inspector_source_id'),market=s if kind=='hybrid' else None) for s in sides}
        assert not index.use_price
        if kind=='hybrid':assert not index.use_description_price and index.description_vectors is None
        if kind!='hybrid':
            st.caption('Price-assisted imputation fitted on '+basis+'. Its price column is dropped before SVD.' if basis else 'Price-assisted imputation is updating; the installed vehicle-only imputer is temporarily used.')
            if kind=='bid':st.caption('MiniLM auction text-value feature: imputed statistical proxy, not an observed description.' if auxiliary is not None else 'The auction text-value imputer is updating; no substitute text is invented.')
        st.markdown(f'#### Neighbourhood · prices for {valuation_date:%d %b %Y}')
        st.write(f"Your query: {fields['year']} {fields['manufacturer']} {fields['model']} · "+(f"{int(fields['odometer']):,} miles" if fields.get('odometer') is not None else 'mileage not supplied'))
        st.caption('🟣 Ask = historical seller advert. 🔵 Auction = historical wholesale sale. Similarity is a cosine score, not a confidence probability. Different badges, same neighbourhood.')
        matches=ordered_matches(groups)
        if not matches:st.info('No usable reference records are installed for this search.');return
        st.caption(f'{len(matches)} neighbours · exact FAISS cosine search · no make/model/year compatibility exclusions.')
        for rank,(side,match) in enumerate(matches,1):
            car=match['vehicle'];label='Ask' if side=='ask' else 'Auction';colour='#eee7ff' if side=='ask' else '#e3f2ff'
            name=f"{car['year']} {str(car['manufacturer']).title()} {car['model']}";adjusted=None;reason=''
            try:
                if transport is None:raise ValueError('Market-date adjustment unavailable')
                adjusted=adjust_reference(car,side,valuation_date,transport,mileage)
            except (ValueError,OSError,KeyError,TypeError) as exc:reason=str(exc)
            with st.container(border=True):
                st.markdown(f'<div style="background:{colour};color:#251b39;padding:12px 15px;border-radius:12px;font-size:1.15rem;font-weight:700">{rank}. {html.escape(name)} · {label}</div>',unsafe_allow_html=True)
                score=match['faiss_cosine_similarity'];st.write(f'Cosine similarity {score:.4f} · rank {rank} across the displayed neighbours')
                year_delta=int(car['year'])-int(fields['year']);deltas=[f"Model year: {year_delta:+d} vs your car" if year_delta else 'Same model year']
                if car.get('odometer') is not None:
                    deltas.append(f"Recorded mileage: {int(car['odometer']):,} miles")
                    if fields.get('odometer') is not None:deltas.append(f"{int(float(car['odometer'])-float(fields['odometer'])):+,} miles vs your car")
                else:deltas.append('Source mileage not supplied')
                st.write(' · '.join(deltas))
                a,b,c=st.columns(3)
                value=adjusted['price_usd'] if adjusted else None
                a.metric('Adjusted asking scenario' if side=='ask' else 'Adjusted auction scenario',f'${value:,.0f}' if value is not None else 'Unavailable')
                original=car.get('price');b.metric('Original advertised price' if side=='ask' else 'Original auction sale',f'${float(original):,.0f}' if original is not None else 'Unavailable')
                own_price=selected[side].get('price');gap=value-own_price if value is not None and own_price is not None else None
                c.metric('Vs your '+('asking' if side=='ask' else 'auction')+' estimate',f'${gap:+,.0f}' if gap is not None else 'Re-estimate your car')
                source_date=str(car.get('observed_date') or car.get('sale_date') or 'Source date unavailable')[:10]
                st.caption(f'Original date: {source_date} · adjusted to {valuation_date:%d %b %Y}'+(f' at your {int(mileage):,} miles.' if mileage is not None else ' at its recorded mileage.'))
                if reason:st.caption(reason)
                changed=[x for x in differences(fields,car) if x['Difference']!='Same']
                if changed:st.write('Different from your car: '+'; '.join(f"{x['Attribute']}: {x['Your car']} → {x['Neighbour']}" for x in changed))
                else:st.write('The supplied categorical specifications match your car; missing facts remain unverified.')
                with st.expander(f'All specifications & retrieval evidence · {rank}'):
                    st.dataframe(differences(fields,car),hide_index=True,width='stretch')
                    for item in match['evidence']:st.write(item)
                    if kind=='bid':st.caption('Description-price is an imputed asking-style feature. No auction text exists in this source.')
                if side=='ask' and kind!='hybrid':
                    with st.expander(f'Read historical seller description · {rank}'):st.text(car.get('description') or 'Description not supplied.')
        st.info('Adjusted historical scenarios, not live offers. Asking and auction values describe different markets; their gap is not dealer profit. Current-date individual accuracy is unvalidated.')
        with st.expander('How prices are brought to the selected date'):
            st.write('The existing age/mileage depreciation curves and retail used-car CPI market index transport each original price. CPI is also used as a wholesale proxy. The existing damped Holt forecast fills the lag after the latest published observation. No physical car’s subsequent mileage or condition is known.')
    except (ValueError,OSError,KeyError,TypeError,RuntimeError,MemoryError):
        st.warning('Neighbour comparisons are temporarily unavailable. Your car estimate and the other tools remain available.')
