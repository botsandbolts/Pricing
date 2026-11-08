"""Single landing-page navigation, including callback and shared-link routing."""
import streamlit as st
PAGES=['🚗 Price my car','🔎 Listing Inspector','🚦 Pick your garage','↔ Two-market garage','📅 Value over time','🧠 The science','📊 Evaluation','🧪 Model lab','🕵️ Deal Detective','🧹 Outlier lab','🏪 Merchant lens']
def render_navigation():
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
    st.markdown('<div id="studio-workspace" tabindex="-1"></div>',unsafe_allow_html=True)
    st.markdown('<div class="garage-nav-heading"><strong>Choose a panel</strong><span>Tap a panel to open it.</span></div>',unsafe_allow_html=True)
    page=st.pills('Studio panels',PAGES,selection_mode='single',key='landing_studio_workspace',label_visibility='collapsed',persist_state='session') or PAGES[0]
    st.session_state['studio_tabs']=page
    from pricing.accessibility import announce
    announce('Current panel: '+page)
    from html import escape
    notes={'🚗 Price my car':'Two markets · seller Ask + wholesale Bid, with historical-date controls.',
           '🔎 Listing Inspector':'Look a little closer · Compare vehicles and explore the evidence.',
           '🚦 Pick your garage':'A little car chemistry · Rate real representatives, then discover your shortlist.',
           '📅 Value over time':'Take the long view · Explore how the same car’s price scenario changes.',
           '🧠 The science':'Under the bonnet · Methods, assumptions and what the models can tell you.',
           '🏪 Merchant lens':'Dealer-level context and merchant comparisons.'}
    st.markdown('<div class="garage-route-note">'+escape(notes.get(page,'The curious corner · Explore the models, evidence and historical comparisons.'))+'</div>',unsafe_allow_html=True)
    return page

