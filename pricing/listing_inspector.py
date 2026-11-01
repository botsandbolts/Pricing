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
        submitted=st.form_submit_button('Find comparable cars',type='primary')
    if submitted:st.session_state['inspector_result']=({'manufacturer':make,'model':model,'year':year,'odometer':miles,'transmission':transmission,'title_status':title,'state':state,'type':body,'condition':condition,'fuel':fuel,'drive':drive,'cylinders':cylinders,'paint_color':color,'auction_grade':grade},description)
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
            version=(manifest.stat().st_mtime_ns if manifest.exists() else 0,imputation_manifest.stat().st_mtime_ns if imputation_manifest.exists() else 0)
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
                for rank,match in enumerate(matches,1):
                    car=match['vehicle']
                    with st.container(border=True):
                        st.markdown(f"### {rank}. {car['year']} {car['manufacturer'].title()} {car['model']}")
                        st.caption(f"{format(int(car['odometer']),',')+' miles' if car.get('odometer') is not None else 'Mileage not supplied'} · {car.get('sale_date',car.get('reference_period','Historical'))}")
                        if side=='bid':st.metric('Historical auction price · USD',f"{car['price']:,.0f}")
                        else:st.caption('Advertised price omitted from this reference sample.')
                        st.caption(f"FAISS cosine similarity: {match['faiss_cosine_similarity']:.3f} · relative similarity, not a confidence probability")
                        for reason in match['evidence']:st.text('• '+reason)
                        if car.get('description'):
                            with st.expander('Read historical seller description'):
                                st.text(car['description'])
                                if car.get('description_is_excerpt'):st.caption('Historical description excerpt · first 1,200 characters.')
                st.caption(f'Reference set: {len(index.rows):,} historical vehicles. Asking adverts are from 2021; auction records from 2015. The reference sample may overlap pricing-model training and is not an independent accuracy benchmark. No current availability is implied.')
            except (ValueError,OSError,KeyError) as exc:
                st.warning('Comparisons are temporarily unavailable. Your other appraisal tools remain available.')
        else:st.info('Choose your car above, then select Find comparable cars.')
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
