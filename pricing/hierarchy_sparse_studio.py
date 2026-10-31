"""Separate regular prices, Bayesian estimates and matched-cohort accuracy."""
import json
from pathlib import Path
import streamlit as st
from pricing.pricing_comparison import mae_comparison

def reports(root):
    return {side:json.loads((Path(root)/'models/xgboost-hierarchy'/side/'bayesian/bayesian-evaluation.json').read_text()) for side in ['ask','bid']}

def scorecard(root,*,details=False):
    data=reports(root)
    st.markdown('### Did the Bayesian layer help?')
    st.caption('Same XGBoost models, same eligible historical test cars. Lower MAE means a smaller average dollar error.')
    rows=[]
    for col,side,label in zip(st.columns(2),['ask','bid'],['Seller asking · Craigslist','Wholesale auction']):
        r=data[side];c=mae_comparison(r['base_test_mae_usd'],r['point_scores']['mae_usd'])
        with col:
            with st.container(border=True):
                st.markdown('#### '+label)
                a,b=st.columns(2)
                a.metric('Regular XGBoost MAE',f"${c['base']:,.0f}")
                b.metric('Bayesian MAE',f"${c['bayesian']:,.0f}")
                outcome=f"${abs(c['saved_usd']):,.2f} "+('less' if c['improved'] else 'more')+f" average error · {abs(c['improvement_pct']):.2f}% "+('better' if c['improved'] else 'worse')
                (st.success if c['improved'] else st.warning)(outcome)
                st.caption(f"{r['test_rows']:,} unchanged eligible test cars. Historical 90% range coverage: {r['test_coverage']:.2%}.")
        rows.append({'Market':label,'Regular MAE ($)':c['base'],'Bayesian MAE ($)':c['bayesian'],'MAE reduction ($)':c['saved_usd'],'MAE reduction (%)':c['improvement_pct'],'Test cars':r['test_rows']})
    if details:
        st.dataframe(rows,hide_index=True)
        st.caption('Positive reduction means improvement; negative means worsening. This comparison isolates the Bayesian layer after cleaning. It does not establish that the 25% cleaning itself improved accuracy.')
        for side,r in data.items():
            with st.expander(side.title()+' · complete historical receipt'):
                p=Path(root)/'models/xgboost-hierarchy'/side
                st.json({'cleaning':json.loads((p/'cleaning.json').read_text()),'xgboost':json.loads((p/'evaluation.json').read_text()),'bayesian':r})
        st.warning('These are historical test results. Date-adjusted scenarios and transported ranges have not been validated on current-market car outcomes.')

def render(root):
    from pricing.analytics import render_privacy
    render_privacy()
    st.markdown('<div id="studio-calculator"></div>',unsafe_allow_html=True)
    st.markdown('## 🍋 The Sparse Lemon Squeezer')
    st.caption('Regular prices first. Bayesian family advice when you want a second opinion.')
    hierarchy_page=st.session_state.get('studio_tabs')=='🌳 Hierarchical model'
    views=['🌳 Bayesian pricing','📊 Accuracy comparison','🧠 How it works'] if hierarchy_page else ['🚗 Regular pricing','🌳 Bayesian pricing','📊 Accuracy comparison','🧠 How it works']
    panel=st.radio('Pricing view',views,horizontal=True,key='hierarchical_pricing_view' if hierarchy_page else 'separate_pricing_panel')
    if panel=='📊 Accuracy comparison':
        scorecard(root,details=True)
        return
    if panel=='🧠 How it works':
        explain()
        return
    bayesian=panel=='🌳 Bayesian pricing' or st.session_state.get('studio_tabs')=='🌳 Hierarchical model'
    if bayesian:
        st.markdown('### 🌳 Bayesian pricing · the family council')
        st.write('Adds make, model and age corrections to the regular XGBoost estimate. Rare models borrow evidence from their make and the wider market.')
        scorecard(root)
        st.info('The correction helped asking-price MAE but worsened auction-price MAE. Regular pricing is available in its own panel.')
    else:
        st.markdown('### 🚗 Regular pricing · straight from the trees')
        st.write('Your cleaned XGBoost models: asking AFT with TF-IDF descriptions, and wholesale median quantile regression.')
        st.caption('Choose Bayesian pricing for the additional correction and historical ranges. Your car details carry across.')
    from pricing.hierarchy_sparse_live import render as live
    live(root,bayesian=bayesian)

def explain():
    st.markdown('### Business English')
    st.write('Each training car is priced by a fresh TF-IDF XGBoost model that did not train on that car. We rank the gap between its recorded and predicted log price, remove the largest 25% of training misses, and train fresh XGBoost models. This may remove errors, but can also remove valid unusual cars. The original validation and test cars stay in the exam.')
    st.write('This hierarchical asking model reads the advert through sparse TF-IDF. No MiniLM or other transformer is loaded. The auction model estimates a typical wholesale sale using median regression. Bayesian make and model effects borrow evidence for rare cars and provide historical outcome ranges. Today’s prices additionally use the existing ageing and market-index scenario.')
    st.markdown('### Technical English')
    st.write('Selection: absolute log(observed/predicted), from five vehicle-group-held-out XGBoost folds; ceil(25% of eligible calibration rows) removed once per market. Ask: normal survival:aft, exact [price,price] bounds. Bid: reg:quantileerror, alpha=0.5, log-dollar labels. Both use 300 boosting rounds, max depth 5, two CPU threads and native categorical splits. Asking descriptions use a fold-fitted 2,000-feature sparse TF-IDF vocabulary. No UMAP or dense TF-IDF matrix is loaded. This panel has no transformer price feature. It uses compact XGBoost models followed by a Bayesian correction fitted to their own OOF log residuals.')
    st.write('Bayesian layer: separate Gaussian make/model-within-make/age residual posteriors for ask and bid, sampled with two blocked Gibbs chains. Validation calibrates 90% historical individual prediction ranges; full test measures coverage. This is not a jointly pooled ask/bid market model, and Gaussian/t/mixture LOO comparison is not implemented.')
    st.warning('Cleaning is adaptive. Post-selection OOF scores are conditional diagnostics, not nested CV of the entire selection pipeline. A matched uncleaned benchmark on the same held-out cars is required to establish whether cleaning helped. Historical interval coverage does not validate today’s extrapolation.')
