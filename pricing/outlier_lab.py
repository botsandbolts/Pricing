"""Outlier lab: observed experiment results, never invented production status."""
import json
import math
from pathlib import Path

def survivor_rows(counts, stages=5, fraction=.05):
    values=dict(counts); rows=[{'Round':0,**values,'Total':sum(values.values())}]
    for i in range(1,stages+1):
        values={k:v-math.ceil(v*fraction) for k,v in values.items()}
        rows.append({'Round':i,**values,'Total':sum(values.values())})
    return rows

def render(root):
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    from pricing.iterative_release import completed
    report_path=Path(__file__).with_name('outlier_experiment.json')
    report=json.loads(report_path.read_text())
    active=completed(root)
    st.markdown('## 🧹 Outlier lab')
    st.caption('Five pit stops. One uncomfortable truth: a cleaner dataset can make a worse price model.')
    if active:
        st.info('Serving status: the five-stage model bundles are installed. Compare their historical evidence below.')
    else:
        st.info('Experiment finished on Kaggle. These five-stage weights are not installed as the calculator default. The calculator still uses its previous installed models.')
    intro,stats=st.columns([2,1])
    with intro:
        with st.container(border=True):
            st.markdown('### What is the cleaner looking for?')
            st.write('Cars whose recorded price strongly disagrees with a separate CatBoost price estimate. That disagreement is a residual. We rank its size on a log-price scale, so proportional surprises matter.')
            st.write('After each pass, the cleaner learns again from surviving training cars. It then removes the largest residuals in each original split. After five passes, a fresh final pricing model is trained.')
            st.caption('A typo can be an outlier. So can a beautifully restored classic. The cleaner cannot certify which is which. “Unusual” is not a synonym for “wrong.”')
    with stats:
        st.metric('Cleaning rounds','5')
        st.metric('Removed each round','5% of survivors')
        st.metric('Removed overall','≈22.6%')
        st.caption('Five rounds do not add up to 25%. The pool gets smaller each time; counts are rounded up.')
    st.divider()
    st.markdown('### 1 · Follow the cars through the five pit stops')
    side=st.radio('Choose a market',['ask','bid'],format_func=lambda x:'🚗 Asking · Craigslist' if x=='ask' else '🔨 Wholesale · auction sales',horizontal=True,key='outlier_market')
    cohort=report['cohorts'][side]
    stages=survivor_rows(cohort['counts'])
    first,last=stages[0],stages[-1]
    a,b,c=st.columns(3)
    a.metric('Eligible cars',f"{first['Total']:,}")
    b.metric('Retained for final refit',f"{last['Total']:,}")
    c.metric('Excluded by this rule',f"{first['Total']-last['Total']:,}")
    frame=pd.DataFrame(stages)
    melted=frame.melt(id_vars=['Round'],value_vars=['train','validation','test'],var_name='Original split',value_name='Cars remaining')
    fig=px.area(melted,x='Round',y='Cars remaining',color='Original split',color_discrete_sequence=['#7357D8','#A7C957','#F2B880'],markers=True)
    fig.update_layout(template='plotly_white',height=310,margin=dict(l=10,r=10,t=15,b=10),legend=dict(orientation='h',y=1.12))
    fig.update_xaxes(dtick=1,title='Cleaning round · 0 = original eligible cohort')
    st.plotly_chart(fig,width='stretch',key='outlier_survivor_chart')
    st.caption('Counts follow the recorded run: ceil(5% × remaining cars), separately within training, validation and test. Exclusions affect training cohorts, not the car you type into the calculator.')
    with st.expander('Exact survivor counts · no disappearing-car magic'):
        st.dataframe(frame.rename(columns={'train':'Training','validation':'Validation','test':'Test'}),hide_index=True,width='stretch')
    st.markdown('### 2 · Did it actually improve prices?')
    st.error('On the full original test population, all four cleaned models had higher MAE than their equivalent unfiltered-training baselines. Lower error on an easier selected test is not evidence of a better general-purpose model.')
    chosen=[r for r in report['models'] if r['side']==side]
    bars=[]
    for r in chosen:
        for label,key in [('Unfiltered fit · full test','baseline'),('Five-stage fit · full test','original'),('Five-stage fit · selected test','filtered')]:
            bars.append({'Final pricing loss':r['loss'],'Evaluation':label,'MAE (USD)':r[key]})
    fig=px.bar(pd.DataFrame(bars),x='Final pricing loss',y='MAE (USD)',color='Evaluation',barmode='group',
        color_discrete_sequence=['#7357D8','#DD6B52','#89A846'],text_auto='$,.0f')
    fig.update_layout(template='plotly_white',height=360,margin=dict(l=10,r=10,t=15,b=10),legend=dict(orientation='h',y=-.25))
    st.plotly_chart(fig,width='stretch',key='outlier_mae_chart')
    for r in chosen:
        change=r['original']-r['baseline']
        with st.container(border=True):
            st.markdown('**'+r['loss']+'**')
            x,y,z=st.columns(3)
            x.metric('Baseline · original test',f"$"+format(r['baseline'],',.0f'))
            y.metric('Cleaned · original test',f"$"+format(r['original'],',.0f'),f"+{100*change/r['baseline']:.1f}% error",delta_color='inverse')
            z.metric('Cleaned · selected test',f"$"+format(r['filtered'],',.0f'))
            st.caption(f"Excluded difficult test cars: MAE "+format(r['excluded'],',.0f')+f" USD. Final deployment refit: {r['trees']:,} trees.")
    st.write('**What happened?** The rule selected cars the cleaner already understood. Removing hard examples can reduce coverage of unusual cars and weaken predictions for the original market. This experiment demonstrates that outcome; it does not establish the exact cause for every vehicle.')
    st.caption('The equivalent unfiltered fits are fresh benchmarks from this experiment. They are not necessarily the previously deployed calculator weights. These are historical dollar errors, not validated accuracy for today’s adjusted prices.')
    st.markdown('### 3 · What this means for your car')
    a,b,c=st.columns(3)
    with a:
        with st.container(border=True):
            st.markdown('#### Everyday cars')
            st.write('The remaining cohort may be more consistent. That alone does not guarantee better predictions—even ordinary cars need independent testing.')
    with b:
        with st.container(border=True):
            st.markdown('#### Rare, damaged or modified cars')
            st.write('A large residual may reflect information the model missed. These cars can lose representation when we discard surprises.')
    with c:
        with st.container(border=True):
            st.markdown('#### Your estimate')
            st.write('Your submitted car is not removed or declared fraudulent. This is a historical training-data filter, not a vehicle inspection or live rejection rule.')
    with st.expander('Try the residual ruler · an illustration, not a trained prediction'):
        expected=st.number_input('Cleaner estimate · illustrative dollars',min_value=100,value=10000,step=500,key='outlier_demo_expected')
        observed=st.number_input('Recorded price · illustrative dollars',min_value=100,value=20000,step=500,key='outlier_demo_observed')
        st.metric('Absolute log residual',f"{abs(math.log(observed/expected)):.3f}")
        st.write('A twofold overprice and a twofold underprice have the same absolute log residual. A fixed dollar difference does not have the same weight for a cheap car and an expensive car.')
        st.caption('This demonstration does not reproduce a fitted cleaner. The actual removal cutoff is a relative ranking within each surviving split, not a universal dollar threshold.')
    with st.expander('How this differs from the earlier cleaning methods'):
        st.dataframe(pd.DataFrame([
            {'Method':'Isolation Forest','Question':'Are the input characteristics unusual?','Consequence':'Can flag legitimate rare combinations; a price-label error can still look ordinary.'},
            {'Method':'Cleanlab regression','Question':'Does a price label look unreliable under cross-validated predictions and uncertainty?','Consequence':'Suspected label issue, not verified error; depends on its predictor and uncertainty estimates.'},
            {'Method':'OutlierTree','Question':'Does this car violate a conditional subgroup pattern?','Consequence':'Provides interpretable subgroup reasons; unusual does not prove invalid.'},
            {'Method':'Single residual cut','Question':'Which cars disagree most with one cleaner?','Consequence':'One ranking and one removal pass.'},
            {'Method':'Five-stage residual cleaning','Question':'Which surviving cars disagree after each fresh refit?','Consequence':'Adaptive exclusions; about 22.6% removed overall; all four full-test MAEs worsened here.'}
        ]),hide_index=True,width='stretch')
        st.caption('Method descriptions are not a controlled ranking of these algorithms. Earlier experiments used different filters, cohorts and final estimators; their scores cannot be attributed to the cleaner alone.')
    with st.expander('Technical receipt · cross-fitting, losses and the test-set catch'):
        st.markdown('**Cleaner:** tabular CatBoost, Quantile:alpha=0.5, log reference-dollar price, 400 trees, depth 6. Vehicle facts use native categories; no make/model embeddings. MiniLM is excluded from the cleaner.')
        st.latex(r"r_i=|\log(P_i)-\widehat{\log(P_i)}|,\qquad n_{s,k+1}=n_{s,k}-\lceil0.05n_{s,k}\rceil")
        st.write('Each round uses three-fold cross-fitted training predictions. A separate training-only fit predicts surviving validation and test cars. Each split loses its own highest-residual ceil(5%) rows. Removed rows never return. Sequential selection is adaptive; this is not a fully nested cross-validation of the whole pipeline.')
        st.write('Both AFT and median quantile final estimators use the same retained cohort in each market. Asking retains the existing verified MiniLM description-price stacking feature. Assessment uses training-only text OOF predictions and a training-only held-out head; deployment uses full-cohort OOF exports paired with the matching full-data head. Auction data have no descriptions.')
        st.write('Normal SurvivalAft receives exact positive dollar bounds [P,P] and transforms internally. Quantile receives log prices. Exponentiating the raw score serves a median-type price, not the arithmetic mean. No observed censoring intervals or fabricated sold/unsold population are supplied.')
        st.write('Final models allow up to 1,500 trees, depth 8, learning rate .05, L2 10 and validation stopping after 100 rounds without improvement. Selected tree count is reused for a separate deployment refit on all retained phases. GPU runs cleaners and quantile; CatBoost SurvivalAft runs on CPU.')
        st.warning('Validation and test labels are used to choose retained cars, making selected-cohort scores optimistic. The full original test remains intact for reporting, but this historical test has informed development. A genuinely new external cohort is required for an unbiased improvement claim. The post-assessment all-retained-data refit has no new independent score.')
    st.markdown('### The sensible next experiment')
    st.write('Freeze a new external test before touching its outcomes. Compare training-only trimming, robust losses without row deletion, and down-weighting instead of exclusion. Audit expensive, damaged and rare-car groups separately. Let the evidence choose the cleaner; the broom does not get a vote.')
    st.caption(report['source']+' · Saved experiment evidence; no retraining occurs when you open this tab.')
    st.link_button('Open saved Kaggle experiment',report['run_url'])
    st.download_button('Download experiment evidence · JSON',json.dumps(report,indent=2),'five-stage-outlier-evidence.json','application/json',key='outlier_evidence_download')
