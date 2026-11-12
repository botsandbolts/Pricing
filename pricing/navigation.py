"""Single landing-page navigation, including callback and shared-link routing."""
import streamlit as st
from functools import partial
from pricing.i18n import tr, localized_panel, localized_panel_for, locale
from pricing.i18n_story import story, story_html
PAGES=['🚗 Price my car','🔎 Listing Inspector','🚦 Pick your garage','↔ Two-market garage','📅 Value over time','🧠 The science','📊 Evaluation','🧪 Model lab','🕵️ Deal Detective','🧹 Outlier lab','🏪 Merchant lens']
def prepare_navigation_state():
    pending=st.session_state.pop('_navigate_to',None)
    if pending in ('🌳 Hierarchical model','🧬 Experimental models','🧬 Experimental pricing'):pending=PAGES[0]
    if st.session_state.get('landing_studio_workspace') in ('🌳 Hierarchical model','🧬 Experimental pricing','🧬 Experimental models'):st.session_state['landing_studio_workspace']=PAGES[0]
    if pending in PAGES:
        st.session_state['landing_studio_workspace']=pending
    elif 'landing_studio_workspace' not in st.session_state:
        st.session_state['landing_studio_workspace']='🕵️ Deal Detective' if st.query_params.get('garage') else PAGES[0]
    # Old ?focus=pricing links enter the same normal workspace once.
    # Navigation remains available; do not hide banners or samples.
    if st.query_params.get('focus') == 'pricing':
        st.session_state['landing_studio_workspace']=PAGES[0]
        st.query_params.pop('focus',None)
    return st.session_state['landing_studio_workspace']


def render_navigation():
    prepare_navigation_state()
    st.markdown('<div id="studio-workspace" tabindex="-1"></div>',unsafe_allow_html=True)
    page=st.session_state.get("landing_studio_workspace",PAGES[0])
    # Keep the canonical panel keys and all research routes, but collapse the
    # eleven-page menu on the ordinary pricing journey.
    with st.expander("🧭 "+localized_panel("🧪 Model lab")+
                     " · "+localized_panel("📊 Evaluation"),
                     expanded=page!=PAGES[0]):
        page=st.pills("Studio panels",PAGES,selection_mode='single',
                      key='landing_studio_workspace',
                      label_visibility='collapsed',persist_state='session',
                      format_func=partial(localized_panel_for,locale())) or PAGES[0]
    st.session_state['studio_tabs']=page
    from pricing.accessibility import announce
    announce(localized_panel(page))
    return page

