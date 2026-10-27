"""Serve the requested residual-trimmed XGBoost defaults with honest cohort labels."""
import gzip,json
from pathlib import Path
import pandas as pd
import streamlit as st
from pricing.model_suite import encode,predict,predictors,IndexPanel
from pricing.suite_storage import read_model_file


def ready(root):
    root=Path(root)/'models/residual-trimmed-xgb'
    return all((root/side/'manifest.json').exists() and json.loads((root/side/'manifest.json').read_text()).get('status')=='ready' for side in ['ask','bid'])


@st.cache_resource
def load(path,version):
    import xgboost as xgb
    folder=Path(path);model=xgb.Booster();model.load_model(bytearray(gzip.decompress(read_model_file(folder/'xgb_aft_normal.ubj.gz'))))
    return model,json.loads((folder/'schema.json').read_text()),json.loads((folder/'manifest.json').read_text())


def render_bid(root,fields):
    folder=Path(root)/'models/residual-trimmed-xgb/bid';mf=folder/'manifest.json';model,schema,record=load(str(folder),mf.stat().st_mtime_ns)
    raw=pd.DataFrame([{**fields,'as_of':'2021-05-01','condition':None}]);x,valid,dates=predictors(raw,'bid');ratios,_=IndexPanel(Path(root)/'models/structural-transfer/used-car-index.csv').ratios(dates)
    if not valid.all() or ratios.isna().any():raise ValueError('Unsupported vehicle/date')
    value=float(predict(model,'xgb_aft_normal',encode(x,schema))[0]*ratios.iloc[0])
    st.subheader('Default auction model · regularized XGBoost AFT')
    st.metric('Transported wholesale auction estimate',f'${value:,.0f}')
    st.caption(f"{record['selected_boosting_rounds']:,} trees; {record['trained_rows']:,} retained cars. Highest-residual 25% per phase plus OutlierTree flags excluded. These are historical wholesale equivalents, not guaranteed bids.")


def render_assessment(root):
    folder=Path(root)/'models/residual-trimmed-xgb'
    st.subheader('New default · larger regularized XGBoost AFT')
    st.warning('The requested residual filtering uses validation/test prices to select easier cars. Filtered scores are optimistic selected-cohort results. Original test scores are also shown, but this test has already informed filtering-rule development; a new external test is needed for an unbiased comparison.')
    rows=[]
    for side in ['ask','bid']:
        r=json.loads((folder/side/'evaluation.json').read_text())
        for cohort,m in r['assessment'].items():rows.append({'Market':side,'Cohort':cohort,**m})
    st.dataframe(pd.DataFrame(rows),hide_index=True)
    with st.expander('Filter counts, training recipe and limitations'):
        for side in ['ask','bid']:
            st.write(side);st.json(json.loads((folder/side/'filter-report.json').read_text()));st.json(json.loads((folder/side/'manifest.json').read_text()))
    st.caption('No transformer text feature in this refit. Native categorical splits. Strong regularization, row/column subsampling, validation stopping and a separate all-retained-data deployment refit. Selecting easy exam questions does not make the mechanic clairvoyant.')
