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
                if 'previous_bayesian_scores' in r:
                    previous=r['previous_bayesian_scores']['mae_usd'];saving=r['improvement_over_previous']
                    st.markdown(f"Previous Bayesian MAE **${previous:,.0f}** → redesigned **${c['bayesian']:,.0f}**")
                    if details:
                        low,high=saving['group_bootstrap_95_usd']
                        st.caption(f"Paired improvement over previous Bayes: ${saving['mean_mae_saved_usd']:,.2f}/car; 95% vehicle-group bootstrap range ${low:,.2f}–${high:,.2f}. This concerns average improvement, not an individual car’s range.")
                    st.caption('Selected errors: '+r['selected_likelihood'].replace('_',' ')+' · nonlinear effects · segment-specific noise.')
        rows.append({'Market':label,'Regular MAE ($)':c['base'],'Bayesian MAE ($)':c['bayesian'],'MAE reduction ($)':c['saved_usd'],'MAE reduction (%)':c['improvement_pct'],'Test cars':r['test_rows']})
    if details:
        st.dataframe(rows,hide_index=True)
        st.caption('Positive reduction means improvement; negative means worsening. This comparison isolates the Bayesian layer after cleaning. It does not establish that the 25% cleaning itself improved accuracy.')
        for side,r in data.items():
            if r.get('candidates'):
                with st.expander(side.title()+' · model bake-off & difficult-car results'):
                    st.dataframe(r['candidates'],hide_index=True)
                    st.caption(f"{r['selection_rows']:,} validation cars choose the model; a separate {r['interval_calibration_rows']:,} calibrate ranges. ELPD is held-out log-price predictive density, not LOO. Lower MAE and higher ELPD are better; selection prioritizes dollar MAE after convergence checks.")
                    st.dataframe([{'Segment':name,**values} for name,values in r['test_segments'].items()],hide_index=True)
                    st.caption('Segments overlap. Overall improvement does not guarantee improvement or nominal coverage in every subgroup.')
            with st.expander(side.title()+' · complete historical receipt'):
                p=Path(root)/'models/xgboost-hierarchy'/side
                st.json({'cleaning':json.loads((p/'cleaning.json').read_text()),'xgboost':json.loads((p/'evaluation.json').read_text()),'bayesian':r})
        st.warning('These are historical test results. Date-adjusted scenarios and transported ranges have not been validated on current-market car outcomes.')

def render(root):
    from pricing.analytics import render_privacy
    render_privacy()
    st.markdown('<div id="studio-calculator"></div>',unsafe_allow_html=True)
    from pricing.description_features import label
    feature_label=label(root)
    st.markdown('## 🍋 The Mini Lemon Squeezer' if feature_label.startswith('MiniLM') else '## 🍋 The Sparse Lemon Squeezer')
    st.caption('Regular prices first. Bayesian family advice when you want a second opinion.')
    hierarchy_page=st.session_state.get('studio_tabs')=='🌳 Hierarchical model'
    views=['🌳 Bayesian pricing','📊 Accuracy comparison','🧠 How it works'] if hierarchy_page else ['🚗 Regular pricing','🌳 Bayesian pricing','📊 Accuracy comparison','🧠 How it works']
    panel=st.radio('Pricing view',views,horizontal=True,key='hierarchical_pricing_view' if hierarchy_page else 'separate_pricing_panel')
    if panel=='📊 Accuracy comparison':
        scorecard(root,details=True)
        return
    if panel=='🧠 How it works':
        explain(root)
        return
    bayesian=panel=='🌳 Bayesian pricing' or st.session_state.get('studio_tabs')=='🌳 Hierarchical model'
    if bayesian:
        st.markdown('### 🌳 Bayesian pricing · the family council')
        redesigned=any(r.get('method')=='robust_cross_classified_gibbs_v1' for r in reports(root).values())
        st.write('A second opinion that learns where the trees systematically miss. The redesigned hierarchy combines nonlinear age, mileage and price effects with pooled make, model and vehicle-detail corrections. Each market’s selected errors and measured value are shown below.' if redesigned else 'Adds make, model and age corrections to the regular XGBoost estimate. Rare models borrow evidence from their make and the wider market.')
        scorecard(root)
        st.info('The matched historical errors above show whether the correction helped each market. Regular pricing remains available separately.')
    else:
        st.markdown('### 🚗 Regular pricing · straight from the trees')
        st.write(f'Cleaned XGBoost uses AFT for asking prices, with {feature_label} as its description input. Auction prices use median quantile regression.')
        st.caption('Choose Bayesian pricing for the additional correction and historical ranges. Your car details carry across.')
    from pricing.hierarchy_sparse_live import render as live
    live(root,bayesian=bayesian)

def explain(root=Path('.')):
    from pricing.description_features import label
    feature_label=label(root)
    st.markdown('### Business English')
    st.write(f'Each training car is priced by a fresh {feature_label} XGBoost model that did not train on that car. We rank the gap between its recorded and predicted log price, remove the largest 25% of training misses, and train fresh XGBoost models. This may remove errors, but can also remove valid unusual cars. The original validation and test cars stay in the exam.')
    st.write(f'This hierarchical asking model reads the advert through {feature_label}. The auction model estimates a typical wholesale sale using median regression. Bayesian make and model effects borrow evidence for rare cars and provide historical outcome ranges. Today’s prices additionally use the existing ageing and market-index scenario.')
    st.markdown('### Technical English')
    st.write(f'Selection: absolute log(observed/predicted), from five vehicle-group-held-out XGBoost folds; ceil(25% of eligible calibration rows) removed once per market. Ask: normal survival:aft, exact [price,price] bounds. Bid: reg:quantileerror, alpha=0.5, log-dollar labels. Both use 300 boosting rounds, max depth 5, two CPU threads and native categorical splits. Asking descriptions use {feature_label}. For the MiniLM price release, one scalar averages two dollar-price predictions. Training rows receive two genuinely group-held-out predictions from repeated two-fold text-head fitting nested inside each outer XGBoost fold. The compact frozen encoder is internal; trees do not receive its raw dimensions. For the earlier embedding experiment, the frozen encoder supplies 384 label-free dimensions; for historical TF-IDF releases, a 2,000-term vocabulary is fitted within each fold. No UMAP is loaded. It uses compact XGBoost models followed by a Bayesian correction fitted to their own OOF log residuals.')
    redesigned=[side for side,r in reports(root).items() if r.get('method')=='robust_cross_classified_gibbs_v1']
    if redesigned:
        st.markdown('#### The second opinion · a reality check')
        st.write('An inexpensive high-mileage car and a premium low-mileage car need different error bars. The redesign learns curved age, mileage and baseline-price effects, make-specific age slopes, and pooled corrections for state, body, transmission, fuel, drive, title, condition, cylinders and paint where available. Model names with fewer than 12 calibration cars share a rare-model pool within their make. No additional adverts are removed.')
        st.write('Gaussian, heavy-tailed Student-t and narrow/wide Gaussian contamination candidates face the same validation exam. Dollar MAE selects among candidates passing monitored convergence checks. Different validation cars calibrate ranges; the unchanged historical test cohort assesses the locked selection. Complete comparisons and difficult-car results are in Accuracy comparison.')
        st.write('Technical: log(observed/base) has a fixed 16-column piecewise-linear basis, cross-classified Gaussian random intercepts, make age slopes and 12 price/age-specific observation variances. Two conditional Gibbs chains sample coefficients, variance hyperparameters and Student-t latent precisions or mixture allocations. Proper Normal/inverse-gamma priors regularize effects. Student-t df=4; contamination scale=5 with Beta(1,9) weight. The posterior location median is inverse-transformed to dollars; unseen groups add variance.')
        st.write('Vehicle-group hashing separates validation into two-thirds selection and one-third Mondrian split-conformal calibration. Risk groups distinguish rare models, baseline prices below $5,000, and age above 12 years or mileage above 150,000. Groups need 100 calibration cars, otherwise they fall back to rarity and then global pools; actual support is shown. Candidate ELPD uses held-out log-price predictive density, not PSIS-LOO. Promotion requires validation improvement, positive lower 95% paired vehicle-group bootstrap savings versus both previous Bayes and regular models, monitored split R-hat <1.1, and test coverage 88–95%. These checks do not prove every latent effect has converged or validate current-date transport.')
    else:
        st.write('Bayesian layer: separate Gaussian make/model/age posteriors sampled with two Gibbs chains. Validation calibrates historical outcome ranges and the original test cohort measures coverage.')
    st.caption('Asking and auction posteriors remain separate. Verified censoring intervals and matched retail/wholesale outcomes are unavailable, so no Tobit bounds or dealer-margin relationship are invented. The test cohort has been reported in earlier releases; this is not a new external benchmark.')
    st.warning('Cleaning is adaptive. Post-selection OOF scores are conditional diagnostics, not nested CV of the entire selection pipeline. A matched uncleaned benchmark on the same held-out cars is required to establish whether cleaning helped. Historical interval coverage does not validate today’s extrapolation.')
