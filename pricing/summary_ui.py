"""Session-local narrative receipt for the supplied car and description."""
import logging
import streamlit as st
from pricing.vehicle_summary import vehicle_story,input_fingerprint,summarize_vehicle


@st.cache_resource(show_spinner=False)
def get_summary_engine():
    from pricing.summary_engines import CpuSummaryEngine
    return CpuSummaryEngine()


def render_vehicle_summary(fields,description):
    st.markdown('#### 📖 The car, without the sales pitch')
    signature=input_fingerprint(vehicle_story(fields,description))
    saved=st.session_state.get('_vehicle_summary',{})
    if saved.get('signature')!=signature:
        try:
            with st.spinner('Reading the backstory. Skipping the dealership poetry…'):
                result=summarize_vehicle(fields,description,get_summary_engine())
            saved=dict(signature=signature,result=result)
            st.session_state['_vehicle_summary']=saved
        except Exception:
            logging.getLogger(__name__).exception('Vehicle summarization unavailable')
            st.caption('The AI summary is temporarily unavailable. Your dollar estimate is ready.')
            return
    result=saved['result']
    if result['status']=='ready':
        st.write(result['summary'])
    else:
        st.caption('This draft needs review and has been withheld. Inspect the supplied story below.')
    st.caption('AI summary of unverified listing information. Claims are not inspection findings or reasons for the predicted price.')
    with st.expander('Read the full supplied story · model & limitations'):
        st.text(result['story'])
        st.caption('Qwen2.5-1.5B-Instruct · Q4_K_M · runs locally on the application server. No Kaggle job or external AI API is used. The same summarize_vehicle function processes every estimate.')
        st.caption('All available car facts and the stored description are considered. Long stories are summarized in chunks and merged. No price targets or residuals enter the prompt. A numeric consistency check catches some invented numbers; it cannot prove factual correctness. The price model continues to read the original description.')
