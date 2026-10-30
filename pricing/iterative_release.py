"""Release status comes from completed artifacts, never the marketing banner."""
import json
from pathlib import Path
from html import escape

FOLDERS=('ask','ask-quantile','bid','bid-aft')


def completed(root):
    folder=Path(root)/'models/iterative-catboost'
    try:
        summary=json.loads((folder/'summary.json').read_text())
        if summary.get('status')!='complete':return False
        for name in FOLDERS:
            manifest=json.loads((folder/name/'manifest.json').read_text())
            if manifest.get('status')!='ready' or manifest.get('outlier_filter',{}).get('stage_count')!=5:return False
            file=folder/name/manifest['model_file']
            # Chunked repository storage has its own verified index.
            if not file.exists() and not file.with_name(file.name+'.parts.json').exists():return False
        return True
    except (OSError,ValueError,KeyError):return False


def render_release(root):
    import streamlit as st
    active=completed(root)
    title='Five-stage formula. Now the default.' if active else 'Five-stage experiment: the results are in.'
    badge='LIVE · 5 × 5%' if active else 'EXPERIMENT · COMPLETED'
    text=('Both markets now use iterative CatBoost residual cleaning before final training. '
          'MiniLM still reads your asking-price description. More scrutiny; no crystal-ball warranty.') if active else (
          'The five-stage CatBoost experiment finished on Kaggle. Its weights are not installed here; estimates still use the previous models. '
          'Open the Outlier lab tab for the full-test results and trade-offs. No increased accuracy claim is made.')
    st.markdown(f'<aside class="formula-release" aria-label="Model release status"><div class="formula-badge">{badge}</div><div><h2>{title}</h2><p>{text}</p></div><div class="formula-number" aria-hidden="true">5<span>pit stops</span></div></aside>',unsafe_allow_html=True)
    if active:st.caption('Active across the calculator, two-market garage, time scenarios and Deal Detective. Historical full-test MAE: asking AFT $3,402; auction median $2,235. Both are worse than the equivalent unfiltered-training baselines. This is a requested experimental default, not an accuracy-based promotion.')
    with st.expander('What changes in the formula? · plain English + technical receipt'):
        st.write('A separate CatBoost cleaner is refitted five times. Each pass removes the highest absolute log-price residuals: ceil(5% of the remaining cars), separately in train, validation and test. Roughly 77.4% survive, with small rounding differences. The bouncer does five rounds; the final estimator starts fresh afterward.')
        st.caption('Training residuals use three-fold cross-fitting. Held-out residuals come from a training-only cleaner. The cleaner uses vehicle facts; asking final fits additionally use verified MiniLM OOF predicted_text_price. No Isolation Forest, ExtraTrees or fabricated IMR. Both AFT and quantile 0.5 are fitted for both markets. Asking defaults to AFT; auction defaults to quantile 0.5 after all four bundles complete.')
        st.warning('Filtering validation and test by their actual prices selects easier cars. Those scores are optimistic. The original test remains intact for comparison, but has informed earlier development. Large residuals may be unusual genuine cars, not mistakes. No guaranteed increase in accuracy is claimed.')
        if active:
            summary=json.loads((Path(root)/'models/iterative-catboost/summary.json').read_text())
            import pandas as pd
            rows=[]
            for name,e in summary['models'].items():
                original=e['assessment']['original_unfiltered_test'];baseline=e['unfiltered_baseline']['assessment']['original_unfiltered_test']
                rows.append({'Model':name,'Original test cars':original['rows'],'Unfiltered-fit MAE ($)':baseline['mae_usd'],
                             'New-fit MAE ($)':original['mae_usd'],'MAE improvement (%)':e['original_test_mae_improvement_pct'],
                             'Selected filtered-test MAE ($)':e['assessment']['filtered_test']['mae_usd']})
            st.dataframe(pd.DataFrame(rows),hide_index=True)
        st.markdown('[Training method and limitations](https://github.com/botsandbolts/Pricing/blob/main/ITERATIVE_CLEANING.md)')
