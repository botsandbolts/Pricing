"""Separate recorded out-of-sample assessments from deployed-weight audit errors."""
import json
from pathlib import Path


def render_model_comparison():
    import streamlit as st
    rows=[]
    for folder,label in [('minilm-assessment','MiniLM + Isolation Forest + LightGBM'),('robust-asking','Unfiltered robust ensemble'),('robust-asking-isolation','Isolation Forest robust ensemble'),('studio-baseline','Tabular baseline')]:
        file=Path('models')/folder/'evaluation.json'
        if not file.exists():continue
        e=json.loads(file.read_text());score=e.get('stacked',e.get('test'))
        if not score:continue
        rows.append({'Model':label,'Original test MAE (USD)':score['mae_usd'],'Original test RMSE (USD)':score['rmse_usd']})
        if 'description_only' in e:
            rows.append({'Model':'MiniLM description only','Original test MAE (USD)':e['description_only']['mae_usd'],'Original test RMSE (USD)':e['description_only']['rmse_usd']})
    st.subheader('All models · the same dollar ruler')
    st.dataframe(rows,hide_index=True)
    st.caption('Recorded chronological assessments before full-data deployment refits. Lower MAE is better: an MAE of $3,500 means $3,500 average absolute error, not a $3,500 guarantee for your car. The transformer does not automatically win just because it has read more words than your mechanic.')
    file=Path('reports/studio-audit/studio-audit.json')
    if file.exists():
        audit=json.loads(file.read_text())
        st.subheader('Recomputed deployed-model audit')
        st.info(audit['split_note'])
        st.write('New partition sizes:',audit['counts'])
        st.caption(f"Deterministic sample from {audit.get('original_test_pool_rows',31777):,} original test listings. Confidence intervals use row bootstrap resampling and can understate uncertainty when listings are correlated.")
        comparison=[]
        for name,result in audit['models'].items():
            for phase in ['validation','test']:
                comparison.append({'Model':name,'Partition':phase,'MAE (USD)':result[phase]['mae_usd'],'RMSE (USD)':result[phase]['rmse_usd'],'MAE 95% bootstrap interval (USD)':'–'.join(f'{n:,.0f}' for n in result[phase].get('mae_bootstrap_95pct_usd',[])),'Scope':result['evaluation_scope']})
        if comparison:st.dataframe(comparison,hide_index=True)
        if audit['status']!='complete':st.caption('Recomputation is running; partial results are not a completed benchmark.')
        st.caption('Models refitted on all rows have training overlap here. Their audit errors cannot be ranked as held-out performance. New out-of-sample comparison of those architectures requires retraining every component on a common new training split.')
        st.download_button('Download recomputed model audit',json.dumps(audit,indent=2),'studio-audit.json','application/json')
