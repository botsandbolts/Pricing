"""Three independent SVD geometries and compact, colour-labelled comparisons."""
from datetime import date
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from pricing.comparable_retrieval import ComparableIndex,reference_rows
from pricing.reference_prices import feature_prices,adjust_reference
from pricing.pricing_comparison import selected_receipt
from marketbridge.time_engine import today

@st.cache_resource(max_entries=1,show_spinner=False)
def joint_imputer(root,version):
    from pricing.description_price_imputation import load
    return load(root,reference_rows(root,'ask'),reference_rows(root,'bid'))

@st.cache_data(max_entries=16,show_spinner=False)
def query_vector(root,text,version):
    from pricing.comparable_semantics import encode_query
    return encode_query(root,text)

def price21(row,side,current_price,target,transport):
    if current_price is None:return np.nan
    copy={**row,'price':current_price,'observed_date':str(target)}
    return feature_prices([copy],side,date(2021,5,1),transport)[0]

def description_to_date(rows,values,target,transport,mileage):
    synthetic=[dict(r,price=float(v),observed_date='2021-05-01') for r,v in zip(rows,values)]
    return feature_prices(synthetic,'ask',target,transport,mileage)

@st.cache_resource(max_entries=3,show_spinner=False)
def build_index(root,kind,version,target,mileage,query_price_key):
    from pricing.comparable_semantics import attach
    from pricing.retrieval_imputation import load as load_imputer
    root=Path(root);rows=reference_rows(root,kind)
    imputer=load_imputer(root,kind,rows)
    prices=description_prices=None;description_missing=None
    auxiliary=None
    if kind!='hybrid':
        from pricing.current_value import current_transport
        try:
            transport,_=current_transport(root)
            if query_price_key is not None:prices=feature_prices(rows,kind,pd.Timestamp(target).date(),transport,mileage)
            auxiliary=joint_imputer(str(root),version)
            if auxiliary is not None:
                imputer=auxiliary.base
                observed=feature_prices(rows,kind,date(2021,5,1),transport)
                signal,description_missing=auxiliary.transform(rows,observed)
                description_prices=description_to_date(rows,signal,pd.Timestamp(target).date(),transport,mileage)
        except (ValueError,OSError,KeyError,TypeError):pass
    index=ComparableIndex(rows,32,imputer=imputer,reference_prices=prices,description_prices=description_prices)
    semantic=attach(index,root) if kind=='ask' else False
    return index,semantic,auxiliary,description_missing

def asset_version(root):
    paths=['models/xgboost-hierarchy/ask/minilm-heads.json','data/comparable-minilm/manifest.json','data/retrieval-imputation/ask.json','data/retrieval-imputation/bid.json','data/retrieval-imputation/hybrid.json','data/retrieval-imputation/description-price.json','data/comparable-catalog/ask.json.gz','data/deal-reference.json.gz']
    return tuple((p,(root/p).stat().st_mtime_ns,(root/p).stat().st_size) for p in paths if (root/p).exists())

def render(root,saved,own,valuation_date,pricing_view):
    st.subheader('Your car’s neighbours · the maths brought name tags')
    mode=st.radio('Matching approach',['Ask · asking adverts','Bid · auction records','Hybrid · vehicle attributes only'],index=2,horizontal=True,key='inspector_svd_mode',persist_state='session')
    kind='ask' if mode.startswith('Ask') else 'bid' if mode.startswith('Bid') else 'hybrid'
    if kind=='hybrid':st.caption('One separate SVD fitted on both datasets. Vehicle attributes only: no price, description, text embedding or transformer-price feature. Five asking and five auction neighbours are shown.')
    elif kind=='ask':st.caption('Asking-only SVD: vehicle attributes, same-date price and description-derived signals. Five asking adverts.')
    else:st.caption('Auction-only SVD: vehicle attributes, same-date auction price and an imputed text-value feature. Auction records contain no seller description. Five auction records.')
    if not saved:st.info('Choose an example or enter your car, then press Price my car & find neighbours.');return
    fields,text=saved
    normalize=st.checkbox('Compare prices at my mileage',value=True,key='inspector_normalize_mileage') and fields.get('odometer') is not None
    mileage=fields.get('odometer') if normalize else None
    version=asset_version(root);target=str(valuation_date)
    receipts=own.get('results',{}) if own and own.get('date')==target else {}
    sides=['ask','bid'] if kind=='hybrid' else [kind]
    selected={s:selected_receipt(receipts.get(s,{'error':'Unavailable'}),pricing_view.startswith('Bayesian')) for s in sides}
    query_price=selected.get(kind,{}).get('price') if kind!='hybrid' else None
    try:
        with st.spinner('Finding nearest neighbours with FAISS cosine search…'):
            index,semantic,auxiliary,missing=build_index(str(root),kind,version,target,mileage,query_price)
            vector=query_vector(str(root),text,version) if kind=='ask' and semantic and text.strip() else None
            description_feature=np.nan;description_inferred=False
            if kind!='hybrid' and auxiliary is not None:
                from pricing.current_value import current_transport
                transport,_=current_transport(root)
                observed=np.nan
                if text.strip():
                    if vector is None:vector=query_vector(str(root),text,version)
                    from pricing.minilm_signal import mean_price
                    heads=json.loads((root/'models/xgboost-hierarchy/ask/minilm-heads.json').read_text())['heads']
                    observed=float(mean_price(heads,np.asarray(vector).reshape(1,-1))[0])
                context=price21(fields,kind,query_price,valuation_date,transport)
                signals,imputed=auxiliary.transform([fields],np.array([context]),np.array([observed]))
                description_feature=description_to_date([fields],signals,valuation_date,transport,mileage)[0]
                description_inferred=bool(imputed[0])
            query={**fields,'_retrieval_price':query_price,'_retrieval_description_price':description_feature}
            groups={side:index.search(query,k=5,description_vector=vector if kind=='ask' else None,exclude_id=st.session_state.get('inspector_source_id'),market=side if kind=='hybrid' else None) for side in sides}
        if kind=='hybrid':
            assert not index.use_price and not index.use_description_price and index.description_vectors is None
        elif not index.use_price:st.caption('The same-date price feature is unavailable for this query; matching uses the remaining supported features.')
        if kind!='hybrid' and index.use_price:st.caption(f"SVD includes log price on the {valuation_date:%d %b %Y} basis. Price-conditioned neighbours are not independent validation of your estimate.")
        if kind=='bid':st.caption('Description-value signal: imputed by the shared iterative model.' if auxiliary is not None else 'Description-value imputation is preparing; no auction text is invented.')
        elif kind=='ask' and auxiliary is not None:st.caption('Description-value signal: '+('imputed because your description is empty.' if description_inferred else 'computed from your description.'))
        st.markdown(f'#### Price comparisons · {valuation_date:%d %b %Y}')
        if kind=='hybrid':st.markdown('🟣 **Ask = seller’s advertised price** &nbsp; 🔵 **Auction = wholesale sale record**')
        elif kind=='ask':st.markdown('🟣 **Seller asking adverts**')
        else:st.markdown('🔵 **Wholesale auction records**')
        st.caption('All values below are date-adjusted historical scenarios. Asking prices precede negotiation; auction prices describe another market. The gap is not dealer profit.')
        from pricing.current_value import current_transport
        try:transport,_=current_transport(root)
        except (ValueError,OSError,KeyError):transport=None
        display=[];details=[]
        for side,matches in groups.items():
            for rank,match in enumerate(matches,1):
                car=match['vehicle'];name=f"{car['year']} {str(car['manufacturer']).title()} {car['model']}"
                adjusted=None;reason=''
                try:
                    if transport is None:raise ValueError('Market-date adjustment unavailable')
                    adjusted=adjust_reference(car,side,valuation_date,transport,mileage)
                except (ValueError,OSError,KeyError,TypeError) as exc:reason=str(exc)
                label='Ask' if side=='ask' else 'Auction'
                display.append({'Market':label,'Car':name,'Value (USD)':adjusted['price_usd'] if adjusted else np.nan})
                details.append({'Market':label,'Rank':rank,'Car':name,'Original USD':car.get('price'),'Original date':str(car.get('observed_date') or car.get('sale_date') or 'Pending source date')[:10],
                                'Adjusted USD':adjusted['price_usd'] if adjusted else np.nan,'Comparison mileage':mileage if normalize else car.get('odometer'),
                                'Cosine similarity':match['faiss_cosine_similarity'],'Text-value signal':'Excluded in hybrid' if kind=='hybrid' else 'Imputed · no auction description' if side=='bid' and auxiliary is not None else 'Not available yet' if side=='bid' else 'Transformer feature',
                                'Adjustment status':reason or 'Model scenario'})
        if display:
            frame=pd.DataFrame(display)
            def colour(row):return ['background-color: '+('#eee7ff' if row['Market']=='Ask' else '#e3f2ff')+'; color: #251b39']*len(row)
            styled=frame.style.apply(colour,axis=1).format({'Value (USD)':'${:,.0f}'},na_rep='—')
            st.dataframe(styled,hide_index=True,width='stretch',height=min(430,38+35*len(frame)))
            st.caption(f"{sum(len(v) for v in groups.values())} neighbours · exact FAISS cosine search · no make/model/year compatibility exclusions.")
            with st.expander('Original prices, dates, similarity and feature provenance'):
                st.dataframe(pd.DataFrame(details),hide_index=True,width='stretch')
                st.caption('Depreciation uses the existing pooled age/mileage curves. Market movement uses retail used-car CPI, also as a wholesale proxy, with the existing damped forecast when observations lag. Individual current-car accuracy and interval coverage remain unvalidated.')
                for side,matches in groups.items():
                    for rank,match in enumerate(matches,1):
                        st.write(f"{'Ask' if side=='ask' else 'Auction'} {rank}: "+'; '.join(match['evidence']))
            if kind=='ask':
                with st.expander('Read the five historical advert descriptions'):
                    for n,match in enumerate(groups['ask'],1):st.write(f"Advert {n}");st.text(match['vehicle'].get('description',''))
        else:st.info('No usable reference records are installed for this search.')
    except (ValueError,OSError,KeyError,TypeError,RuntimeError,MemoryError) as exc:
        st.warning('Neighbour comparisons are temporarily unavailable. Your car estimate and the other tools remain available.')
