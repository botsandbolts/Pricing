"""Selectable modest ask/bid models, rank agreement, and Ridge calibration."""
import json
import gzip
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from pricing.suite_storage import read_model_file
from pricing.model_suite import NAMES,encode,predict,predictors,IndexPanel

LABELS={'lgb_huber':'LightGBM Huber · defaults','lgb_quantile_50':'LightGBM median · quantile 0.50',
        'xgb_aft_normal':'Original XGBoost AFT · historical benchmark','ridge_baseline':'Ridge · comparison baseline'}


@st.cache_resource
def load_suite(path,version):
    import joblib
    import lightgbm as lgb
    import xgboost as xgb
    path=Path(path);models={}
    for name in NAMES:
        if name.startswith('lgb_'):
            with gzip.open(path/f'{name}.txt.gz','rt') as f:models[name]=lgb.Booster(model_str=f.read())
        elif name.startswith('xgb_'):
            m=xgb.Booster()
            m.load_model(bytearray(gzip.decompress(read_model_file(path/f'{name}.ubj.gz'))))
            models[name]=m
        else:models[name]=joblib.load(path/f'{name}.joblib')
    return models,json.loads((path/'schema.json').read_text()),json.loads((path/'manifest.json').read_text())


def render(root,key_prefix="suite"):
    root=Path(root);suite=root/'models/price-model-suite'
    if not (suite/'comparison.json').exists():suite=root/'models/price-model-suite-tabular'
    from pricing.residual_ui import ready as residual_ready,render_assessment,render_calculator
    if residual_ready(root):
        render_calculator(root,key_prefix=key_prefix+'_robust');render_assessment(root)
    st.subheader('Historical comparison: three models walk into a car dealership')
    st.write('Huber takes extreme prices with a pinch of salt. Median regression aims at the middle car. AFT models a bell-shaped uncertainty in log price. Ridge is the straight-line accountant checking their arithmetic.')
    if not (suite/'comparison.json').exists():
        st.info('New model training has not completed. Scores and prices appear here only after the fitted bundles are ready.');return
    st.info('These are the original unfiltered benchmark models. The new residual-trimmed defaults and their separate filtered/original-test assessments appear above when ready.')
    with st.expander('Why AFT? Tobit and interval censoring · plain language and maths'):
        doc=root/'TOBIT_AND_CENSORING.md'
        if doc.exists():st.markdown(doc.read_text())
    manifest=json.loads((suite/'ask/manifest.json').read_text())
    st.caption(f"Cleanlab deployment audit: {manifest['outlier_filter']['removed_rows']:,} rows flagged for exclusion. Held-out validation and test cars remain unfiltered. "+('MiniLM descriptions enabled.' if manifest['text_feature'] else 'Structured-only comparison. MiniLM text refit is separate; descriptions do not affect these prices.'))
    rerun=root/'reports/cleanlab-rerun-20261027/summary.json'
    if rerun.exists():
        with st.expander('Fresh Cleanlab rerun · asking and auction data'):
            audit=json.loads(rerun.read_text());audit_rows=[]
            for side,record in audit['markets'].items():
                for stage,r in record['stages'].items():
                    audit_rows.append({'Market':side,'Stage':stage,'Candidates':r['input_rows'],'Retained':r['retained_rows'],'Flagged':r['removed_rows']})
            st.dataframe(pd.DataFrame(audit_rows),hide_index=True)
            st.caption(audit['weight_compatibility'])
    with st.form(f'{key_prefix}_modest-model-lab'):
        a,b,c=st.columns(3)
        year=a.number_input('Model year',1900,2027,2015,key=f'{key_prefix}_year')
        make=b.text_input('Manufacturer','Toyota',key=f'{key_prefix}_make');model=c.text_input('Model / trim','Corolla',key=f'{key_prefix}_model')
        miles=a.number_input('Miles',0,2000000,95000,key=f'{key_prefix}_miles')
        body=b.selectbox('Body style',['sedan','suv','truck','coupe','wagon','van','hatchback','convertible'],key=f'{key_prefix}_body')
        state=c.text_input('State abbreviation','ca',key=f'{key_prefix}_state')
        transmission=a.selectbox('Transmission',['automatic','manual','unknown'],key=f'{key_prefix}_transmission')
        fuel=b.selectbox('Fuel',['gas','diesel','hybrid','electric','unknown'],key=f'{key_prefix}_fuel')
        date=c.date_input('Valuation date',pd.Timestamp('2021-05-01').date(),key=f'{key_prefix}_date')
        grade=a.selectbox('Auction condition grade',[None,1.,2.,3.,4.,5.],format_func=lambda v:'Unknown' if v is None else str(v),key=f'{key_prefix}_grade')
        notes=st.text_area('Description',disabled=not manifest['text_feature'],key=f'{key_prefix}_description')
        ask_choice=b.selectbox('Asking model',NAMES,index=NAMES.index('xgb_aft_normal'),format_func=LABELS.get,key=f'{key_prefix}_ask')
        bid_choice=c.selectbox('Bid / auction model',NAMES,index=NAMES.index('xgb_aft_normal'),format_func=LABELS.get,key=f'{key_prefix}_bid')
        submit=st.form_submit_button('Let the pricing committee argue')
    if submit:
        fields={'year':year,'manufacturer':make,'model':model,'odometer':miles,'type':body,'state':state,
            'transmission':transmission,'fuel':fuel,'as_of':str(date),'auction_condition':grade}
        raw=pd.DataFrame([fields]);results=[];text_price=None
        if manifest['text_feature']:
            from pricing.text import load_text,predict_text
            neural,tok=load_text(root/'models/minilm-cleanlab/text-model','cpu')
            text_price=float(predict_text(neural,tok,[notes],'cpu')[0])
        panel=IndexPanel(root/'models/structural-transfer/used-car-index.csv')
        for side,choice in [('ask',ask_choice),('bid',bid_choice)]:
            folder=suite/side;mf=folder/'manifest.json';models,schema,record=load_suite(str(folder),mf.stat().st_mtime_ns)
            x,valid,dates=predictors(raw,side);ratios,_=panel.ratios(dates)
            if not valid.all() or (side=='bid' and ratios.isna().any()):
                st.error('Choose a supported valuation date and vehicle year.');continue
            if side=='ask' and record['text_feature']:x['predicted_text_price']=text_price
            p=float(predict(models[choice],choice,encode(x,schema))[0])
            if side=='bid':p*=float(ratios.iloc[0])
            results.append({'Side':'Historical asking' if side=='ask' else 'Transported wholesale auction','Model':LABELS[choice],'Price USD':p})
        st.dataframe(pd.DataFrame(results),hide_index=True)
        st.caption('Asking prices describe the 2021 listing market. Auction prices are CPI- and age-transported historical wholesale estimates, not a promised bid or verified retail sale.')
    rows=[]
    for side in ['ask','bid']:
        report=json.loads((suite/side/'evaluation.json').read_text())
        for name,r in report['models'].items():rows.append({'Side':side,'Model':LABELS[name],**r['test']})
    st.markdown('**Untouched test-set accuracy**');st.dataframe(pd.DataFrame(rows),hide_index=True)
    comparison=json.loads((suite/'comparison.json').read_text());rho=pd.DataFrame(comparison['spearman'])
    st.markdown('**Spearman rank agreement · same asking test cars**')
    st.write('A value near +1 means two models rank expensive and cheap cars similarly. It does not mean their dollar estimates match, or that either model is accurate. Two wrong speedometers can agree beautifully.')
    import plotly.express as px
    st.plotly_chart(px.imshow(rho.astype(float),zmin=-1,zmax=1,color_continuous_scale='RdBu_r',text_auto='.3f'),use_container_width=True,key=f'{key_prefix}_spearman')
    st.markdown('**Ridge: slopes, intercepts and bias**')
    st.write('Each row fits response = intercept + beta × predictor on validation cars, then checks it on test cars. Beta 1 and intercept $0 mean matching scales. Bias is mean(prediction − reference): positive means overpricing. An intercept is not the same thing as bias.')
    st.dataframe(pd.DataFrame(comparison['ridge_pairs']),hide_index=True)
    with st.expander('Calibrating each model to observed prices · technical'):
        for side in ['ask','bid']:
            report=json.loads((suite/side/'evaluation.json').read_text())
            st.write(side);st.dataframe(pd.DataFrame(report['comparisons']['ridge_observed_price_calibration']),hide_index=True)
        st.write('Ridge alpha=1; predictor standardized using validation statistics; beta and intercept converted back to original dollar units. Ridge feature baseline uses one-hot categories and standardized numeric fields, log1p target, alpha=1. Cross-market tables measure prediction agreement only: these data have no paired observed retail transactions.')
        st.write('AFT normal scale=1, 100 boosting rounds, native categorical splits. Bounds equal observed prices because no measured censoring bounds exist. This is an uncensored log-normal price model; Tobit censoring requires actual censoring information. LightGBM uses default statistical parameters with Huber or quantile alpha=.5 and log1p target. Auction targets are CPI-adjusted to January 2015 before fitting and transported to the requested valuation month.')
    st.download_button('Download correlations and Ridge coefficients',json.dumps(comparison,indent=2),'model-comparison.json','application/json',key=f'{key_prefix}_download_comparison')


def render_bid_companion(root,fields):
    """New Cleanlab bid models on the main calculator's submitted car."""
    from pricing.residual_ui import ready as residual_ready,render_bid
    if residual_ready(root):
        render_bid(root,fields);return
    root=Path(root);folder=root/'models/price-model-suite/bid'
    if not (folder/'manifest.json').exists():folder=root/'models/price-model-suite-tabular/bid'
    record=json.loads((folder/'manifest.json').read_text())
    if record.get('status')!='ready':raise ValueError('Auction suite is not ready')
    models,schema,_=load_suite(str(folder),(folder/'manifest.json').stat().st_mtime_ns)
    raw=pd.DataFrame([{**fields,'as_of':'2021-05-01','condition':None}])
    x,valid,dates=predictors(raw,'bid');ratios,_=IndexPanel(root/'models/structural-transfer/used-car-index.csv').ratios(dates)
    if not valid.all() or ratios.isna().any():raise ValueError('Vehicle or date outside supported index')
    st.subheader('Cleanlab auction companion · May 2021')
    choice=st.selectbox('Auction pricing option',NAMES[:3],index=NAMES[:3].index('xgb_aft_normal'),format_func=LABELS.get,key='cleanlab_auction_companion')
    value=float(predict(models[choice],choice,encode(x,schema))[0]*ratios.iloc[0])
    st.metric('Transported wholesale auction estimate',f'${value:,.0f}')
    st.caption('Newly fitted after Cleanlab auditing. Age and lagged CPI transport a historical wholesale model; this is not a guaranteed bid or calibrated retail sale price. Auction descriptions are unavailable, so no text feature is invented.')
