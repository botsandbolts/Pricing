"""Clearly separated experiment; no silent replacement of production."""
import json
from pathlib import Path
import streamlit as st
def render(root):
    st.header('Experimental pricing · The extra brain cell')
    st.write('The same Lemon cohort, XGBoost AFT asking model and Bayesian residual hierarchy, with extra BGE description coordinates. Wholesale pricing remains the existing model because auction records have no seller descriptions.')
    st.caption('BGE encoding is divided into 20 jobs. PCA is fitted only on each outer training fold; validation and test descriptions never fit PCA. Explicit money amounts are redacted from the BGE text.')
    p=Path(root)/'reports/bge-experiment/comparison.json'
    if not p.exists():
        st.info('Training and evaluation pending. No BGE accuracy gain has been established and no experimental estimate is presented as a trained result.')
    else:
        report=json.loads(p.read_text())
        st.dataframe(report['comparison'],hide_index=True)
        st.caption(report['limits'])
        with st.expander('Business and technical receipt'):st.json(report)
    st.markdown('### Your current pricing model')
    st.write('The regular calculator remains available in Price my car. This panel will show the completed head-to-head assessment; a winning research model still needs serving-memory and inference checks before release.')
