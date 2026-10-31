"""Single landing-page navigation, including callback and shared-link routing."""
import streamlit as st
PAGES=['🚗 Price my car','↔ Two-market garage','📅 Value over time','🧠 The science','📊 Evaluation','🧪 Model lab','🕵️ Deal Detective','🧹 Outlier lab','🌳 Hierarchical model']
def render_navigation():
    pending=st.session_state.pop('_navigate_to',None)
    if pending in PAGES:
        st.session_state['landing_studio_workspace']=pending
    elif 'landing_studio_workspace' not in st.session_state:
        st.session_state['landing_studio_workspace']='🌳 Hierarchical model' if st.query_params.get('hierarchy') else '🕵️ Deal Detective' if st.query_params.get('garage') else PAGES[0]
    st.markdown('### Explore the studio')
    page=st.pills('Studio panels',PAGES,selection_mode='single',key='landing_studio_workspace',label_visibility='collapsed') or PAGES[0]
    st.session_state['studio_tabs']=page
    return page
