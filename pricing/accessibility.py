"""Native, opt-in reading controls and semantic status messages. No overlay."""
from html import escape
from pathlib import Path
import streamlit as st


def install():
    css=(Path(__file__).resolve().parents[1]/'assets/accessibility.css').read_text()
    if st.session_state.get('a11y_larger_text'):
        css+='\nhtml {font-size:112.5%;}'
    if st.session_state.get('a11y_high_contrast'):
        css+='\n'+(Path(__file__).resolve().parents[1]/'assets/high-contrast.css').read_text()
    st.markdown('<style>'+css+'</style>',unsafe_allow_html=True)
    st.markdown('<a class="garage-skip-link" href="#studio-workspace">Skip to panels and tools</a>',unsafe_allow_html=True)


def reading_options():
    with st.expander('Reading options',expanded=False):
        st.checkbox('Larger text',key='a11y_larger_text',persist_state='session')
        st.checkbox('Higher contrast',key='a11y_high_contrast',persist_state='session')
        st.caption('These settings apply throughout this session. You can also use browser zoom. Tab moves between controls; arrow keys move within a rating or panel group; Enter or Space activates buttons.')


def announce(message):
    st.markdown('<div class="garage-sr-only" role="status" aria-live="polite" aria-atomic="true">'+escape(str(message))+'</div>',unsafe_allow_html=True)
