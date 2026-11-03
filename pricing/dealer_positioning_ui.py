"""Lightweight historical dealer association research panel."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st
def render(root):
    st.header('Merchant lens · cars, markets and historical evidence')
    from pricing.same_car_markets import render as render_same_car
    render_same_car(root)
    st.divider()
    st.subheader('Historical merchant-name associations · separate research')
    st.write('Different cars. Different services. A cautious look at the remaining associations.')
    folder=Path(root)/'reports/dealer-positioning'
    if not (folder/'method-and-metrics.json').exists():
        st.info('The research report is not installed.');return
    report=json.loads((folder/'method-and-metrics.json').read_text())
    st.warning('Exploratory historical research—not a dealer quality, fairness or overcharging ranking. Names are automatically inferred and unverified; extraction can mistake platforms such as Craigslist for merchants. These are not current dealer price policies or completed transaction prices.')
    st.info('Merchant services may vary considerably: warranties, vehicle preparation, inspection, delivery, financing and other services can explain differences. Unmeasured vehicle characteristics can also explain them. No inference about dealer profit, misconduct or value for money is supported.')
    a=report['base_oof_mae_usd'];b=report['dealer_features_oof_mae_usd']
    c1,c2,c3=st.columns(3)
    c1.metric('Vehicle-only research MAE',f'$'+format(a,',.0f'))
    c2.metric('With dealer signals · MAE',f'$'+format(b,',.0f'))
    c3.metric('MAE reduction',f'{100*(a-b)/a:.2f}%')
    st.caption('Same grouped five-fold historical comparison. A modest association, not proof of a reliable improvement. This research uses log squared-error XGBoost, not the installed AFT/Bayesian pricing pipeline. Live estimates are unchanged.')
    final_path=Path(root)/'reports/dealer-overlap/completed-analysis.json'
    if final_path.exists() and json.loads(final_path.read_text()).get('version',1)>=2:
        final=json.loads(final_path.read_text())
        st.subheader('Comparable inventory · completed model-level analysis')
        st.write(final['status'])
        st.warning('Model-level comparison only: trim differences remain uncontrolled. Canonical date provenance is unverified. The estimates are associations, not causal effects.')
        summary=[]
        for run in final['runs']:
            for key,label in [('primary','Single candidate · no score cutoff'),('alternative_top_candidate','Top candidate · no score cutoff')]:
                r=run[key]
                summary.append({'Imputation seed':run['seed'],'Merchant assignment':label,'Input adverts':r['input_adverts'],
                  'Shared-support adverts':r['shared_adverts'],'Retained (%)':round(100*r['retained_fraction'],2),
                  'Vehicle-only OOF MAE (USD)':r.get('oof_mae_usd'),'Status':r['status'],
                  'Time holdout':r.get('time_holdout',{}).get('status','Not estimable')})
        st.dataframe(pd.DataFrame(summary),hide_index=True,width='stretch')
        st.subheader('Merchant evidence · alphabetical, not a leaderboard')
        if final['table']:
            st.dataframe(pd.DataFrame(final['table']).round(2),hide_index=True,width='stretch')
            st.caption('Blank associations are deliberately suppressed: support or sensitivity checks did not qualify. Conditional ranges are not total uncertainty or validated coverage.')
        else:
            st.info('No merchant qualified across all five imputations. No merchant percentage comparison can be supported under this specification.')
        with st.expander('Methods, temporal checks and uncertainty limitations'):
            st.write(final['method'])
            st.write(final['support_definition'])
            st.warning(final['limitations'])
            for run in final['runs']:
                st.write('Imputation seed '+str(run['seed']))
                for key in ['primary','alternative_top_candidate']:
                    st.write(key)
                    st.json(run[key].get('time_holdout',{'status':'Insufficient support'}))
        st.download_button('Download completed research report',json.dumps(final,indent=2),'merchant-model-level-research.json','application/json')
    else:
        st.info('The full analysis is rerunning without downstream NER score cutoffs. Earlier thresholded results are withheld until the replacement report is ready.')
    with st.expander('Original exploratory experiment · provenance'):
        st.json(report)
    st.caption('Research provenance: GitHub Actions run 41680537032. No claim about a named business should be drawn without checking the source adverts, identity and services.')
