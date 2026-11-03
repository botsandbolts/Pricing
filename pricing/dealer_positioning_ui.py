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
    table=pd.read_csv(folder/'historical-associations.csv').sort_values('inferred_dealer',key=lambda s:s.str.casefold())
    st.subheader('Inferred name groups · alphabetical, not ranked')
    st.write('Percentages describe residual asking-price associations after the research model’s vehicle controls. Zero is the overall residual baseline, not a fair-price threshold. A positive value does not establish overpricing; a negative value does not establish a bargain.')
    st.error('Interval limitation: these conditional 95% Bayesian intervals omit dealer-identification errors, baseline-model uncertainty and prior uncertainty. They are not calibrated total-uncertainty intervals. Branches may be conflated, and one business may appear under several names.')
    table=table.rename(columns={'inferred_dealer':'Inferred name · unverified','distinct_vehicle_groups':'Distinct vehicle groups','adjusted_association_pct':'Residual association (%)','conditional_95_low_pct':'Conditional 95% lower (%)','conditional_95_high_pct':'Conditional 95% upper (%)'})
    st.warning('Named residual percentages are withheld pending shared-support checks, within-support refitting, and temporal sensitivity evaluation.')
    audit_path=Path(root)/'reports/dealer-overlap/audit.json'
    if audit_path.exists():
        audit=json.loads(audit_path.read_text())
        st.subheader('Shared inventory support')
        st.dataframe(pd.DataFrame(audit['primary']['summary']),hide_index=True,width='stretch')
        st.write(audit['primary']['time_holdout'])
        missing=audit['primary']['missing_counts']
        st.write('Missing comparison fields among confidence-selected adverts:')
        st.dataframe(pd.DataFrame([{'Field':k,'Missing adverts':v} for k,v in missing.items()]),hide_index=True,width='stretch')
        if not any(x['shared_support_adverts']>=50 for x in audit['primary']['summary']):
            st.warning('No merchant has sufficient joint support for this specification. Dealer percentage estimates and within-support refitting are blocked. More complete trim and listing-date evidence is required.')

        with st.expander('Inventory distributions and alternative merchant-match threshold'):
            st.json(audit)
    else:st.info('The overlap audit is running. Existing residual percentages are not evidence of comparable inventory.')
    with st.expander('Business explanation'):
        st.write('The model first estimates historical asking prices without dealer names. We examine the errors by inferred name group, borrowing evidence from the overall population so small samples are pulled toward zero. Absence of a dealer name means unknown seller type, not a private seller. Historical cohort selection and missing service information limit interpretation.')
    with st.expander('Technical method, assumptions and limitations'):
        st.json(report)
        st.write('Names require a single candidate, extraction score at least 0.90 and at least 50 distinct vehicle groups. Scores are not calibrated identity probabilities. Vehicle grouping is VIN-based where available, otherwise make/model/year/mileage. The fixed normal prior has SD 0.10 log units; a plug-in residual SD supplies the normal likelihood. This first version is not a robust Student-t hierarchy or a causal dealer effect estimate. State controls are included; listing-date controls and private-seller comparisons are not established.')
    st.caption('Research provenance: GitHub Actions run 41680537032. No claim about a named business should be drawn without checking the source adverts, identity and services.')
