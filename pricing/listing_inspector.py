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
    from pricing.studio_design import page_intro,journey_steps
    page_intro('Look a little closer','Listing Inspector','Compare similar cars. Explore prices, uncertainty and market evidence.','🔎')
    journey_steps([('Bring a listing','Start with the car and its description.'),('Meet the neighbours','Compare similar historical vehicles.'),('Explore the evidence','Price scenarios, checks and market context.')])
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
        submitted=st.form_submit_button('Price my car & find neighbours',type='primary')
    if submitted:
        import time as _telemetry_time
        from pricing.feature_telemetry import record_server_run, record_server_error
        started=_telemetry_time.monotonic()
        st.session_state['inspector_result']=({'manufacturer':make,'model':model,'year':year,'odometer':miles,'transmission':transmission,'title_status':title,'state':state,'type':body,'condition':condition,'fuel':fuel,'drive':drive,'cylinders':cylinders,'paint_color':color,'auction_grade':grade},description)
        st.session_state.pop('inspector_own_price',None)
        try:
            from pricing.inspector_estimate import quote
            fields,notes=st.session_state['inspector_result']
            with st.spinner('Pricing your car with the installed models, then finding nearest neighbours…'):
                st.session_state['inspector_own_price']=quote(root,fields,notes,price_date,allow)
            record_server_run('listing_inspector','bayesian' if pricing_view.startswith('Bayesian') else 'xgboost',
                              'success',1,1,round(1000*(_telemetry_time.monotonic()-started)))
        except (ValueError,OSError,KeyError,MemoryError,RuntimeError,TypeError):
            record_server_run('listing_inspector','bayesian' if pricing_view.startswith('Bayesian') else 'xgboost',
                              'unavailable',1,0,round(1000*(_telemetry_time.monotonic()-started)))
            record_server_error('listing_inspector','pricing_unavailable')
            st.info('Own-car pricing is unavailable for these inputs. The neighbour search can still run; no substitute estimate is invented.')
    from pricing.inspector_estimate import render as render_own_price
    own=st.session_state.get('inspector_own_price')
    if own and own.get('date')==str(price_date):render_own_price(root,own,pricing_view.startswith('Bayesian'))
    elif own:st.info('Valuation date changed. Press Price my car & find neighbours to refresh your car’s estimates.')
    saved=st.session_state.get('inspector_result')
    tabs=st.tabs(['Comparable cars','Purchase sandbox','Deal & confidence','Duplicate check','Regional prices','Batch review','Information premium','Methods & limits'])
    with tabs[0]:
        from pricing.inspector_neighbours import render as render_neighbours
        render_neighbours(root,saved,own,price_date,pricing_view)
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
        st.write('Retrieval uses 32-dimensional truncated SVD of weighted one-hot categories and scaled numerical features, followed by L2 normalization and exact cosine search with FAISS IndexFlatIP. No make, model or year compatibility filter is applied. Three independent SVDs are fitted: Ask uses vehicle attributes and optional MiniLM text similarity; Bid uses vehicle attributes and an imputed MiniLM description-price signal; Hybrid uses vehicle attributes from both markets. Advert/auction prices can assist Ask/Bid imputation but are then dropped before SVD. Hybrid excludes price and text even during imputation.')
        st.write('SVD is unsupervised. Ask and Bid return five neighbours from their respective markets. Hybrid returns five asking and five auction neighbours using the same combined vehicle-only projection. In Ask mode, when text is available, separately normalized SVD/MiniLM blocks have 80%/20% weights before final normalization. There is no intermediate candidate pool or second-stage ranker. No LambdaRank or FM model is used. The engineering audit checks 32/64/128 dimensions, cross-brand behaviour, latency and memory; no human-judged relevance gain or NDCG improvement is claimed.')
        st.write('Regional associations and information-premium experiments are separate offline research reports. Duplicate checks and batch queues process your uploaded CSV only on request; no pricing model is retrained. Price comparisons are model outputs, not calibrated bargain probabilities. Nothing here establishes demand, time to sell, realized dealer margin or a causal effect of wording.')

    with tabs[7]:
        for label,name in [('Business guide · how to use every tool','LISTING_DECISION_TOOLS.md'),('Technical guide · methods, evaluation and limits','LISTING_DECISION_TECHNICAL.md')]:
            document=root/name
            if document.exists():
                with st.expander(label):st.markdown(document.read_text())

