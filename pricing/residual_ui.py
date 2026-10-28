"""Serve the requested residual-trimmed XGBoost defaults with honest cohort labels."""
import gzip,json
from pathlib import Path
import pandas as pd
import streamlit as st
from pricing.model_suite import encode,predict,predictors,IndexPanel,apply_category_pooling
from pricing.suite_storage import read_model_file


def model_root(root):
    for name in ['catboost-aft','residual-trimmed-xgb']:
        folder=Path(root)/'models'/name
        if all((folder/side/'manifest.json').exists() and json.loads((folder/side/'manifest.json').read_text()).get('status')=='ready' for side in ['ask','bid']):return folder
    return Path(root)/'models/catboost-aft'

def ready(root):
    folder=model_root(root)
    return all((folder/side/'manifest.json').exists() for side in ['ask','bid'])

def estimate(model,schema,record,raw,side):
    if record['pipeline']=='catboost-aft':return model.predict_frame(raw)
    x,valid,_=predictors(raw,side)
    if not valid.all():raise ValueError('Unsupported vehicle')
    return predict(model,'xgb_aft_normal',encode(apply_category_pooling(x,record.get('category_pooling',{})),schema))


@st.cache_resource
def load(path,version):
    folder=Path(path);record=json.loads((folder/'manifest.json').read_text())
    if record['pipeline']=='catboost-aft':
        from pricing.catboost_aft_pipeline import CatBoostAFTPipeline
        model=CatBoostAFTPipeline(folder);return model,model.schema,record
    import xgboost as xgb
    model=xgb.Booster();model.load_model(bytearray(gzip.decompress(read_model_file(folder/'xgb_aft_normal.ubj.gz'))))
    return model,json.loads((folder/'schema.json').read_text()),json.loads((folder/'manifest.json').read_text())


def render_bid(root,fields):
    folder=model_root(root)/'bid';mf=folder/'manifest.json';model,schema,record=load(str(folder),mf.stat().st_mtime_ns)
    raw=pd.DataFrame([{**fields,'as_of':'2021-05-01','condition':None}]);x,valid,dates=predictors(raw,'bid');ratios,_=IndexPanel(Path(root)/'models/structural-transfer/used-car-index.csv').ratios(dates)
    if not valid.all() or ratios.isna().any():raise ValueError('Unsupported vehicle/date')
    value=float(estimate(model,schema,record,raw,'bid')[0]*ratios.iloc[0])
    st.subheader('Default auction model · CatBoost AFT')
    st.metric('Transported wholesale auction estimate',f'${value:,.0f}')
    st.caption(f"{record['selected_boosting_rounds']:,} trees; {record['trained_rows']:,} retained cars. Highest-residual 25% per phase plus OutlierTree flags excluded. These are historical wholesale equivalents, not guaranteed bids.")


def render_assessment(root):
    folder=model_root(root)
    st.subheader('New default · regularized AFT models')
    st.warning('The requested residual filtering uses validation/test prices to select easier cars. Filtered scores are optimistic selected-cohort results. Original test scores are also shown, but this test has already informed filtering-rule development; a new external test is needed for an unbiased comparison.')
    rows=[]
    for side in ['ask','bid']:
        r=json.loads((folder/side/'evaluation.json').read_text())
        for cohort,m in r['assessment'].items():rows.append({'Market':side,'Dataset':'Craigslist Used Cars' if side=='ask' else 'Used Car Auction Prices (Kaggle / tunguz)','Cohort':cohort,**m})
    st.dataframe(pd.DataFrame(rows),hide_index=True)
    with st.expander('Filter counts, training recipe and limitations'):
        for side in ['ask','bid']:
            st.write(side);st.json(json.loads((folder/side/'filter-report.json').read_text()));st.json(json.loads((folder/side/'manifest.json').read_text()))
    st.caption('No transformer text feature in this refit. Native categorical splits. Strong regularization, row/column subsampling, validation stopping and a separate all-retained-data deployment refit. Selecting easy exam questions does not make the mechanic clairvoyant.')


def render_calculator(root,key_prefix='robust_pair'):
    st.subheader('Default ask + auction pricing · CatBoost AFT')
    st.caption('Two market estimates. One car. The dealer and the auctioneer are still allowed to disagree.')
    with st.form(key_prefix+'_form'):
        a,b,c=st.columns(3)
        year=a.number_input('Year',1900,2027,2015,key=key_prefix+'_year')
        make=b.text_input('Make','Toyota',key=key_prefix+'_make');name=c.text_input('Model / trim','Corolla',key=key_prefix+'_model')
        miles=a.number_input('Miles',0,2000000,95000,key=key_prefix+'_miles');body=b.selectbox('Body',['sedan','suv','truck','coupe','wagon','van','hatchback','convertible'],key=key_prefix+'_body');state=c.text_input('State','ca',key=key_prefix+'_state')
        transmission=a.selectbox('Transmission',['automatic','manual','unknown'],key=key_prefix+'_transmission');fuel=b.selectbox('Fuel',['gas','diesel','hybrid','electric','unknown'],key=key_prefix+'_fuel');date=c.date_input('Valuation date',pd.Timestamp('2021-05-01').date(),key=key_prefix+'_date')
        grade=a.selectbox('Actual auction grade',[None,1.,2.,3.,4.,5.],format_func=lambda g:'Unknown' if g is None else str(g),key=key_prefix+'_grade')
        submit=st.form_submit_button('Price it in both markets')
    if submit:
        raw=pd.DataFrame([{'year':year,'manufacturer':make,'model':name,'odometer':miles,'type':body,'state':state,'transmission':transmission,'fuel':fuel,'as_of':str(date),'auction_condition':grade}]);rows=[]
        for side in ['ask','bid']:
            folder=model_root(root)/side;mf=folder/'manifest.json';model,schema,record=load(str(folder),mf.stat().st_mtime_ns)
            x,valid,dates=predictors(raw,side);ratios,_=IndexPanel(Path(root)/'models/structural-transfer/used-car-index.csv').ratios(dates)
            if not valid.all() or (side=='bid' and ratios.isna().any()):st.error('Unsupported vehicle/date');return
            value=float(estimate(model,schema,record,raw,side)[0])
            if side=='bid':value*=float(ratios.iloc[0])
            rows.append({'Market':side,'Price USD':value,'Model':'Residual-trimmed CatBoost AFT'})
        st.dataframe(pd.DataFrame(rows),hide_index=True)
        st.caption('Asking prices retain their historical listing-market target; auctions are CPI transported wholesale estimates. Neither is a verified current retail transaction. No description text feature in this refit.')
