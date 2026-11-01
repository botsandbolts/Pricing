"""Independent workspace for comparable vehicles and statistical appraisal tools."""
import math

def acquisition_scenario(resale, purchase, repairs, transport, fees, holding):
    values=[resale,purchase,repairs,transport,fees,holding]
    if not all(math.isfinite(float(x)) and float(x)>=0 for x in values):raise ValueError('Amounts must be finite and nonnegative.')
    return float(resale)-sum(map(float,values[1:]))

def render(root):
    import gzip,json
    import pandas as pd
    import streamlit as st
    st.title('🔎 Listing Inspector')
    st.write('Compare similar cars. Explore prices, uncertainty and market evidence.')
    st.caption('Historical comparisons and statistical appraisal tools. Existing pricing models are unchanged.')
    from pricing.vehicle_presets import render as render_examples
    render_examples(root)
    from marketbridge.time_engine import today
    date_mode=st.radio('Valuation date',['Today','Choose a date'],horizontal=True,key='inspector_date_mode',persist_state='session')
    price_date=today() if date_mode=='Today' else st.date_input('Value this car on',value=today(),key='inspector_price_date',persist_state='session')
    pricing_view=st.radio('Own-car pricing',['Regular XGBoost','Bayesian + historical range'],horizontal=True,key='inspector_own_view',persist_state='session')
    with st.form('inspector_form'):
        a,b,c=st.columns(3)
        make=a.text_input('Make','Toyota',key='inspector_make',persist_state='session');model=b.text_input('Model / trim','Corolla',key='inspector_model',persist_state='session');year=c.number_input('Model year',1900,2030,2015,key='inspector_year',persist_state='session')
        a,b,c=st.columns(3)
        miles=a.number_input('Mileage',0,2000000,95000,key='inspector_mileage',persist_state='session')
        transmission=b.selectbox('Transmission',['unknown','automatic','manual'],key='inspector_transmission',persist_state='session')
        title=c.selectbox('Title status',['unknown','clean','salvage','rebuilt','lien'],key='inspector_title',persist_state='session')
        with st.expander('More vehicle details · improves the pricing review'):
            a,b,c=st.columns(3)
            state=a.text_input('US state abbreviation','ca',max_chars=2,key='inspector_state',persist_state='session')
            body=b.selectbox('Body type',['unknown','sedan','SUV','truck','pickup','hatchback','coupe','wagon','van','convertible','other'],key='inspector_body',persist_state='session')
            condition=c.selectbox('Condition',['unknown','excellent','good','fair','like new','new','salvage'],key='inspector_condition',persist_state='session')
            fuel=a.selectbox('Fuel',['unknown','gas','diesel','hybrid','electric','other'],key='inspector_fuel',persist_state='session')
            drive=b.selectbox('Drive',['unknown','fwd','rwd','4wd'],key='inspector_drive',persist_state='session')
            cylinders=c.selectbox('Cylinders',['unknown','3 cylinders','4 cylinders','5 cylinders','6 cylinders','8 cylinders','10 cylinders','12 cylinders','other'],key='inspector_cylinders',persist_state='session')
            color=a.selectbox('Paint color',['unknown','white','black','silver','grey','blue','red','green','brown','yellow','orange','purple','custom'],key='inspector_color',persist_state='session')
            grade=b.number_input('Auction condition grade · optional (1–5)',min_value=1.,max_value=5.,value=None,step=.1,key='inspector_grade',persist_state='session')
        description=st.text_area('Paste the seller description',max_chars=8000,key='inspector_description',persist_state='session')
        allow=st.checkbox('Allow an auction-generation analogue for 2016–2021 cars',value=True,key='inspector_own_analogue',persist_state='session')
        submitted=st.form_submit_button('Price my car & find 5 similar cars',type='primary')
    if submitted:
        st.session_state['inspector_result']=({'manufacturer':make,'model':model,'year':year,'odometer':miles,'transmission':transmission,'title_status':title,'state':state,'type':body,'condition':condition,'fuel':fuel,'drive':drive,'cylinders':cylinders,'paint_color':color,'auction_grade':grade},description)
        st.session_state.pop('inspector_own_price',None)
        try:
            from pricing.inspector_estimate import quote
            fields,notes=st.session_state['inspector_result']
            with st.spinner('Pricing your car with the installed models, then finding five neighbours…'):
                st.session_state['inspector_own_price']=quote(root,fields,notes,price_date,allow)
        except (ValueError,OSError,KeyError,MemoryError,RuntimeError,TypeError):
            st.info('Own-car pricing is unavailable for these inputs. The neighbour search can still run; no substitute estimate is invented.')
    from pricing.inspector_estimate import render as render_own_price
    own=st.session_state.get('inspector_own_price')
    if own and own.get('date')==str(price_date):render_own_price(root,own,pricing_view.startswith('Bayesian'))
    elif own:st.info('Valuation date changed. Press Price my car & find 5 similar cars to refresh your car’s estimates.')
    saved=st.session_state.get('inspector_result')
    tabs=st.tabs(['Comparable cars','Purchase sandbox','Deal & confidence','Duplicate check','Regional prices','Batch review','Information premium','Methods & limits'])
    with tabs[0]:
        st.subheader('Find the family resemblance')
        st.caption('FAISS retrieves five nearest cars by cosine similarity of normalized SVD vectors. Scaled numerical features enter SVD; descriptions add normalized MiniLM similarity when available. Similarity is not a valuation or a probability.')
        market=st.radio('Compare against', ['Historical asking adverts','Historical auction sales'],horizontal=True,key='inspector_market')
        if saved:
            from pricing.comparable_retrieval import ComparableIndex,reference_rows
            from pricing.comparable_semantics import attach,encode_query
            side='ask' if market.startswith('Historical asking') else 'bid'
            @st.cache_resource(max_entries=2,show_spinner=False)
            def get_index(folder,side,version):
                rows=reference_rows(folder,side)
                from pricing.retrieval_imputation import load as load_imputer
                imputer=load_imputer(folder,side,rows)
                index=ComparableIndex(rows,32,imputer=imputer)
                semantic=attach(index,folder) if side=='ask' else False
                return index,semantic
            manifest=root/'data/comparable-minilm/manifest.json'
            imputation_manifest=root/'data/retrieval-imputation'/(side+'.json')
            reference_file=root/('data/comparable-catalog/ask.json.gz' if side=='ask' else 'data/deal-reference.json.gz')
            version=(manifest.stat().st_mtime_ns if manifest.exists() else 0,imputation_manifest.stat().st_mtime_ns if imputation_manifest.exists() else 0,reference_file.stat().st_mtime_ns if reference_file.exists() else 0)
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
                if not matches:st.info('No usable neighbours in this reference set. Try the other historical market.')
                from marketbridge.time_engine import today
                from pricing.reference_prices import adjust_reference
                from pricing.current_value import current_transport
                normalize_mileage=st.checkbox('Compare prices at my mileage',value=True,key='inspector_normalize_mileage') and saved[0].get('odometer') is not None
                st.caption(f"Prices adjusted to {price_date:%d %b %Y}. "+(f"Comparison mileage: {int(saved[0]['odometer']):,} miles." if normalize_mileage else 'Each reference keeps its original stated mileage.'))
                transport=None
                if any(m['vehicle'].get('price') is not None for m in matches):
                    try:transport,refresh_error=current_transport(root)
                    except (ValueError,OSError,KeyError):pass
                price_rows=[]

                for rank,match in enumerate(matches,1):
                    car=match['vehicle']
                    with st.container(border=True):
                        st.markdown(f"### {rank}. {car['year']} {car['manufacturer'].title()} {car['model']}")
                        st.caption(f"{format(int(car['odometer']),',')+' miles' if car.get('odometer') is not None else 'Mileage not supplied'} · {car.get('sale_date',car.get('reference_period','Historical'))}")
                        if car.get('price') is not None:
                            source_date=car.get('observed_date') or car.get('sale_date') or 'Historical'
                            st.caption(f"Original {'asking' if side=='ask' else 'auction'} price: USD {car['price']:,.0f} · {str(source_date)[:10]}")
                            try:
                                if transport is None:raise ValueError('The market-date adjustment is temporarily unavailable.')
                                adjusted=adjust_reference(car,side,price_date,transport,saved[0]['odometer'] if normalize_mileage else None)
                                st.metric(f"Adjusted {'asking' if side=='ask' else 'auction'} comparison · {price_date:%d %b %Y} · USD",f"{adjusted['price_usd']:,.0f}")
                                st.caption(f"Age/mileage factor ×{adjusted['age_mileage_factor']:.3f} · market factor ×{adjusted['market_factor']:.3f}"+(' · pooled make/body depreciation' if adjusted['pooled_model_curve'] else ''))
                                market_status=adjusted['market'];st.caption(f"Market index observed through {market_status['last_known_month']}"+(f" · {market_status['horizon']}-month {market_status.get('method','')} forecast" if market_status['horizon'] else ' · observed month'))
                                price_rows.append({'Car':f"{car['year']} {car['manufacturer']} {car['model']}",'Original date':adjusted['original_date'],'Original price USD':round(adjusted['original_price_usd']),f"Adjusted to {price_date} USD":round(adjusted['price_usd'])})
                            except (ValueError,OSError,KeyError,TypeError) as exc:st.caption('Date-adjusted comparison unavailable: '+str(exc))
                        else:st.caption('Restoring this record’s original advertised price from the source data; no substitute price is invented.')
                        st.caption(f"FAISS cosine similarity: {match['faiss_cosine_similarity']:.3f} · relative similarity, not a confidence probability")
                        for reason in match['evidence']:st.text('• '+reason)
                        if car.get('description'):
                            with st.expander('Read historical seller description'):
                                st.text(car['description'])
                                if car.get('description_is_excerpt'):st.caption('Historical description excerpt · first 1,200 characters.')
                if price_rows:
                    st.markdown('#### Compare the prices · same valuation date')
                    st.dataframe(price_rows,hide_index=True,width='stretch')
                st.caption('Date-adjusted historical price scenarios, not current listings or offers. Age/mileage uses the existing pooled depreciation curves; market movement uses retail used-car CPI, including a damped forecast when observations lag. It is also a proxy for wholesale movement. No individual current-price accuracy or interval coverage is established.')
                st.caption(f'Reference set: {len(index.rows):,} historical vehicles. Asking adverts are from 2021; auction records from 2015. The reference sample may overlap pricing-model training and is not an independent accuracy benchmark. No current availability is implied.')
            except (ValueError,OSError,KeyError) as exc:
                st.warning('Comparisons are temporarily unavailable. Your other appraisal tools remain available.')
        else:st.info('Choose your car above, then press Price my car & find 5 similar cars.')
    with tabs[1]:
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
    from pricing.listing_insights_ui import render_deal,render_duplicates,render_regions,render_batch,render_information
    with tabs[2]:render_deal(root,saved)
    with tabs[3]:render_duplicates()
    with tabs[4]:render_regions(root)
    with tabs[5]:render_batch(root)
    with tabs[6]:render_information(root)
    with tabs[7]:
        st.write('Retrieval uses 32-dimensional truncated SVD of weighted one-hot categories and scaled numerical features, followed by L2 normalization and exact cosine search with FAISS IndexFlatIP. No make, model or year compatibility filter is applied. Prices never enter retrieval. MiniLM description cosine similarity is an additional signal when available.')
        st.write('SVD is unsupervised. Five comparisons are retrieved directly by cosine nearest-neighbour search. When text is available, separately normalized SVD/MiniLM blocks have 80%/20% weights before final normalization. There is no intermediate candidate pool or second-stage ranker. No LambdaRank or FM model is used. The engineering audit checks 32/64/128 dimensions, guardrails, latency and memory; no human-judged relevance gain or NDCG improvement is claimed.')
        st.write('Regional associations and information-premium experiments are separate offline research reports. Duplicate checks and batch queues process your uploaded CSV only on request; no pricing model is retrained. Price comparisons are model outputs, not calibrated bargain probabilities. Nothing here establishes demand, time to sell, realized dealer margin or a causal effect of wording.')

    with tabs[7]:
        for label,name in [('Business guide · how to use every tool','LISTING_DECISION_TOOLS.md'),('Technical guide · methods, evaluation and limits','LISTING_DECISION_TECHNICAL.md')]:
            document=root/name
            if document.exists():
                with st.expander(label):st.markdown(document.read_text())
