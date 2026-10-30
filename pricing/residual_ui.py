"""Route unchanged asking AFT and independently selected auction defaults."""
import gzip,json
from pathlib import Path
import pandas as pd
import streamlit as st
from pricing.model_suite import encode,predict,predictors,IndexPanel,apply_category_pooling
from pricing.suite_storage import read_model_file


def model_root(root):
    from pricing.iterative_release import completed
    if completed(root):return Path(root)/'models/iterative-catboost'
    for name in ['minilm-catboost-aft','catboost-aft','residual-trimmed-xgb']:
        folder=Path(root)/'models'/name
        if all((folder/side/'manifest.json').exists() and json.loads((folder/side/'manifest.json').read_text()).get('status')=='ready' for side in ['ask','bid']):return folder
    return Path(root)/'models/catboost-aft'

def bid_folder(root):
    from pricing.iterative_release import completed
    if completed(root):return Path(root)/'models/iterative-catboost/bid'
    candidate=Path(root)/'models/bid-quantile-cleaned';mf=candidate/'manifest.json'
    if mf.exists() and json.loads(mf.read_text()).get('status')=='ready':return candidate
    return model_root(root)/'bid'


def ready(root):
    folder=model_root(root)
    return all((folder/side/'manifest.json').exists() for side in ['ask','bid'])

def estimate(model,schema,record,raw,side):
    if record['pipeline'] in ['catboost-aft','catboost-log-quantile']:
        text_price=None
        if record.get('text_feature'):
            from pricing.text import predict_text
            descriptions=raw['text'].fillna('').astype(str).tolist() if 'text' in raw else ['']*len(raw)
            text_price=predict_text(model.text_model,model.tokenizer,descriptions,model.device)
        return model.predict_frame(raw,text_price)
    x,valid,_=predictors(raw,side)
    if not valid.all():raise ValueError('Unsupported vehicle')
    return predict(model,'xgb_aft_normal',encode(apply_category_pooling(x,record.get('category_pooling',{})),schema))


@st.cache_resource
def load(path,version):
    folder=Path(path);record=json.loads((folder/'manifest.json').read_text())
    if record['pipeline'] in ['catboost-aft','catboost-log-quantile']:
        from pricing.catboost_aft_pipeline import CatBoostAFTPipeline
        model=CatBoostAFTPipeline(folder);return model,model.schema,record
    import xgboost as xgb
    model=xgb.Booster();model.load_model(bytearray(gzip.decompress(read_model_file(folder/'xgb_aft_normal.ubj.gz'))))
    return model,json.loads((folder/'schema.json').read_text()),json.loads((folder/'manifest.json').read_text())


def render_bid(root,fields):
    default=bid_folder(root);previous=model_root(root)/('bid-aft' if model_root(root).name=='iterative-catboost' else 'bid')
    options={'CatBoost log-median · quantile 0.5 (default)':default,('CatBoost normal AFT · same five-stage cleaning' if model_root(root).name=='iterative-catboost' else 'CatBoost normal AFT · previous'):previous} if default!=previous else {'CatBoost normal AFT':previous}
    price_slot=st.container()
    details=st.expander('Auction model · controls & technical receipt')
    with details:
        selected=st.selectbox('Auction pricing model',list(options),index=0,key='companion_bid_model')
    folder=options[selected];mf=folder/'manifest.json';model,schema,record=load(str(folder),mf.stat().st_mtime_ns)
    raw=pd.DataFrame([{**fields,'as_of':'2021-05-01','condition':None}]);x,valid,dates=predictors(raw,'bid');ratios,_=IndexPanel(Path(root)/'models/structural-transfer/used-car-index.csv').ratios(dates)
    if not valid.all() or ratios.isna().any():raise ValueError('Unsupported vehicle/date')
    value=float(estimate(model,schema,record,raw,'bid')[0]*ratios.iloc[0])
    from pricing.analytics import record_price
    request=st.session_state.get('_price_request_id')
    if request:
        record_price('bid',value,fields,selected,'calculator-companion','2021-05-01',request+':'+selected)
    with price_slot:
        from pricing.price_meaning import render_market_price
        render_market_price('bid',value,'2021-05-01',adjusted=True)
    with details:
        st.write('CatBoost log-median · quantile 0.5' if record['pipeline']=='catboost-log-quantile' else 'CatBoost normal AFT')
        if record.get('training_regime')=='five-stage-catboost-residual':
            st.caption('Five fresh CatBoost residual rounds, each removing ceil(5% of survivors) independently per phase. Both quantile and AFT share the resulting cohort. Filtered scores are optimistic.')
        else:st.caption(f"{record['selected_boosting_rounds']:,} trees; {record['trained_rows']:,} retained cars. {('Exactly ceil(25%) per phase removed by a separate log-quantile CatBoost residual cleaner; no extra OutlierTree cuts.' if record['pipeline']=='catboost-log-quantile' else 'Highest-residual 25% per phase plus OutlierTree flags excluded.')} These are historical wholesale equivalents, not guaranteed bids.")
        if record['pipeline']=='catboost-log-quantile':
            selection=record.get('selection_imr',{})
            st.caption('Cleaner-only IMR: '+selection.get('status','unavailable')+'. '+selection.get('reason','Lambda is excluded from the final price model.'))


def render_assessment(root):
    folder=model_root(root)
    if folder.name=='iterative-catboost':
        from pricing.iterative_release import render_release
        render_release(root)
    st.subheader('Default models · asking AFT and auction log-median')
    st.warning('The requested residual filtering uses validation/test prices to select easier cars. Filtered scores are optimistic selected-cohort results. Original test scores are also shown, but this test has already informed filtering-rule development; a new external test is needed for an unbiased comparison.')
    rows=[]
    for side in ['ask','bid']:
        active=folder/'ask' if side=='ask' else bid_folder(root)
        r=json.loads((active/'evaluation.json').read_text())
        for cohort,m in r['assessment'].items():rows.append({'Market':side,'Estimator':r.get('parameters',{}).get('loss_function','historical'),'Dataset':'Craigslist Used Cars' if side=='ask' else 'Used Car Auction Prices (Kaggle / tunguz)','Cohort':cohort,**m})
    st.dataframe(pd.DataFrame(rows),hide_index=True)
    if folder.name=='iterative-catboost':
        st.markdown('#### Five pit stops · survivor counts')
        counts=[]
        for side in ['ask','bid']:
            r=json.loads((folder/(side+'-cleaning')/'filter-report.json').read_text())
            for stage in r['stages']:
                counts.extend({'Market':side,'Stage':stage['stage'],**row} for row in stage['counts'])
        st.dataframe(pd.DataFrame(counts),hide_index=True)
    with st.expander('Filter counts, training recipe and limitations'):
        for side in ['ask','bid']:
            active=folder/'ask' if side=='ask' else bid_folder(root)
            st.write(side);st.json(json.loads((active/'filter-report.json').read_text()));st.json(json.loads((active/'manifest.json').read_text()))
    st.caption(('MiniLM description-price stacking is active for Craigslist; the auction dataset has no descriptions. ' if json.loads((folder/'ask/manifest.json').read_text()).get('text_feature') else 'Tabular-only comparison. ')+'Native categorical splits. Strong regularization, row/column subsampling, validation stopping and a separate all-retained-data deployment refit. Selecting easy exam questions does not make the mechanic clairvoyant.')


def render_calculator(root,key_prefix='robust_pair'):
    from pricing.price_meaning import render_market_guide,render_price_pair
    st.subheader('Advertised asking price vs wholesale auction sale')
    render_market_guide()
    st.caption('Asking model: MiniLM + CatBoost AFT. Auction model: CatBoost log-median 0.5.' if json.loads((model_root(root)/'ask/manifest.json').read_text()).get('text_feature') else 'Tabular CatBoost comparison models.')
    from pricing.current_value import clock_controls
    date,allow=clock_controls(key_prefix)
    with st.form(key_prefix+'_form'):
        a,b,c=st.columns(3)
        year=a.number_input('Year',1900,2021,2015,key=key_prefix+'_year')
        make=b.text_input('Make','Toyota',key=key_prefix+'_make');name=c.text_input('Model / trim','Corolla',key=key_prefix+'_model')
        miles=a.number_input('Miles',0,2000000,95000,key=key_prefix+'_miles');body=b.selectbox('Body',['sedan','suv','truck','coupe','wagon','van','hatchback','convertible'],key=key_prefix+'_body');state=c.text_input('State','ca',key=key_prefix+'_state')
        from marketbridge.time_engine import today
        transmission=a.selectbox('Transmission',['automatic','manual','unknown'],key=key_prefix+'_transmission');fuel=b.selectbox('Fuel',['gas','diesel','hybrid','electric','unknown'],key=key_prefix+'_fuel')
        grade=a.selectbox('Actual auction grade',[None,1.,2.,3.,4.,5.],format_func=lambda g:'Unknown' if g is None else str(g),key=key_prefix+'_grade')
        description=st.text_area('Seller description · MiniLM asking-price signal',max_chars=8000,key=key_prefix+'_description')
        submit=st.form_submit_button('Price it in both markets')
    if submit:
        raw=pd.DataFrame([{'year':year,'manufacturer':make,'model':name,'odometer':miles,'type':body,'state':state,'transmission':transmission,'fuel':fuel,'as_of':str(date),'auction_condition':grade,'text':description}]);rows=[]
        from pricing.current_value import current_transport,render_market_clock,render_adjustment
        try:transport,refresh_error=current_transport(root)
        except (ValueError,OSError,KeyError) as exc:st.warning('Date adjustment unavailable: '+str(exc));return
        for side in ['ask','bid']:
            folder=model_root(root)/'ask' if side=='ask' else bid_folder(root);mf=folder/'manifest.json';model,schema,record=load(str(folder),mf.stat().st_mtime_ns)
            try:
                receipt=transport.estimate(model,description,raw.iloc[0].to_dict(),date,side,allow,as_of=today())
                value=receipt['price']
                if receipt['generation_analogue']:render_adjustment(receipt)
            except ValueError as exc:
                st.warning(side.upper()+' estimate unavailable: '+str(exc));continue
            rows.append({'Market':side,'Price USD':value,'Model':'MiniLM + CatBoost AFT' if record.get('text_feature') else 'CatBoost log-median 0.5' if record['pipeline']=='catboost-log-quantile' else 'Tabular CatBoost AFT'})
            from pricing.analytics import record_price
            record_price(side,value,raw.iloc[0].to_dict(),str(folder),key_prefix,str(date))
        prices={r['Market']:r['Price USD'] for r in rows}
        render_price_pair(prices.get('ask'),prices.get('bid'),str(date),asking_date=str(date),scenario=True)
        try:render_market_clock(transport,date,refresh_error)
        except ValueError as exc:st.warning(str(exc))
        with st.expander('Model names and exact outputs'):
            st.dataframe(pd.DataFrame(rows),hide_index=True)
        st.caption('Both estimates use the selected date, with separate vehicle-age and market adjustments. Asking is advertised price; auction is wholesale sale price. Neither is a verified current retail transaction. MiniLM reads asking descriptions; auction training has no descriptions.')
