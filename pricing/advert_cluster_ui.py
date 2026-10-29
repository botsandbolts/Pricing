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
        with st.expander('How this advert type works'):
            st.write('TF-IDF captures words and phrases. Common vehicle categories enter as one-hot fields; rare and unseen values are ignored. Model year and mileage become training-based percentile ranks. Weighted Hellinger distance combines 65% description, 20% categories and 15% numbers, then UMAP maps to 10 dimensions and HDBSCAN discovers groups. Missing blocks redistribute weight to the available information.')
            st.write('The name comes from distinctive, well-supported TF-IDF phrases in the group. New adverts use the frozen UMAP transform and approximate HDBSCAN assignment. Weak matches stay unclassified. Match strength is not a calibrated probability.')
            st.caption('Exploratory historical Craigslist advert types—not verified dealer/private classifications, vehicle quality checks or pricing features. The AFT/MiniLM price model is unchanged. Grouping uses neither prices nor price residuals.')
    except Exception:
        logging.getLogger(__name__).exception('Advert clustering unavailable')
        st.caption('Advert grouping is temporarily unavailable. Your price estimate is unaffected.')
