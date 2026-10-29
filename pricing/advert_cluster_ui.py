"""Optional result card; cluster failures must never suppress a dollar estimate."""
import logging
import hashlib
import json
from pathlib import Path

import streamlit as st

FOLDER = Path(__file__).resolve().parents[1]/'models/advert-clusters'


@st.cache_resource(show_spinner=False)
def _load(folder, stamp):
    from pricing.advert_clusters import load_bundle
    return load_bundle(folder)


def render_advert_type(text, fields):
    st.markdown('#### 🧩 Your advert’s tribe')
    manifest = FOLDER/'manifest.json'
    if not manifest.exists():
        st.caption('Advert grouping is unavailable until its trained bundle is installed. Your price is unaffected.')
        return
    try:
        with st.spinner('Finding its parking tribe…'):
            stamp = manifest.stat().st_mtime_ns
            signature = hashlib.sha256(json.dumps([text[:8000], fields, stamp], sort_keys=True, default=str).encode()).hexdigest()
            cached = st.session_state.get('_advert_cluster_result', {})
            if cached.get('signature') != signature:
                from pricing.advert_clusters import classify
                result = classify(_load(str(FOLDER), stamp), text[:8000], fields)
                st.session_state['_advert_cluster_result'] = {'signature': signature, 'result': result}
            else:
                result = cached['result']
        st.markdown('**'+result['name']+'**')
        if result['cluster_id'] < 0:
            st.caption(result['reason'])
        else:
            st.caption('Group keywords: '+', '.join(result['keywords']))
            st.caption(f"Match strength: {result['strength']:.0%} · {result['training_members']:,} reference adverts. A language club, not a vehicle inspection.")
            if result['strength'] < .30:
                st.caption('Weak association: included for broader coverage. Treat this group as tentative.')
        with st.expander('How this advert type works'):
            st.write('Descriptions and all 11 vehicle category fields share a TF-IDF vocabulary. Rare categorical values—including singletons—are retained as field-qualified tokens. There is no one-hot block. Values never observed during fitting are ignored by the fixed vocabulary. Year and mileage use training-based percentile ranks. Weighted Hellinger distance combines 85% TF-IDF and 15% numbers before UMAP and HDBSCAN.')
            st.write('UMAP and HDBSCAN settings are selected using validation assignment coverage, with checks against a collapsed partition. The final test sample is evaluated after selection. The name comes from distinctive TF-IDF terms, including categorical terms where useful. Native HDBSCAN assignments are retained even when weak; density noise stays unclassified. There is no forced nearest-cluster assignment. Match strength is not a calibrated probability, and coverage is not classification accuracy.')
            st.caption('Exploratory historical Craigslist advert types—not verified dealer/private classifications, vehicle quality checks or pricing features. The AFT/MiniLM price model is unchanged. Grouping uses neither prices nor price residuals.')
            report_file=FOLDER/'report.json'
            if report_file.exists():
                report=json.loads(report_file.read_text())
                held=report.get('holdouts',{})
                if all('coverage' in held.get(phase,{}) for phase in ('validation','test')):
                    st.caption(f"Historical assignment coverage: validation {held['validation']['coverage']:.1%}, test {held['test']['coverage']:.1%}. These rates measure how many adverts receive a group—not whether that group is correct.")
    except Exception:
        logging.getLogger(__name__).exception('Advert clustering unavailable')
        st.caption('Advert grouping is temporarily unavailable. Your price estimate is unaffected.')
