"""Lightweight real-browser landing preview; no heavyweight prediction models."""
from pathlib import Path
import sys
import streamlit as st

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
st.set_page_config(page_title="Car Price Studio · Landing QA",layout="wide",
                   initial_sidebar_state="collapsed")
from pricing.studio_design import install_style,render_hero
from pricing.navigation import render_navigation

install_style()
render_hero()
page=render_navigation()
if page=="🚗 Price my car":
    st.markdown('<div id="studio-calculator" tabindex="-1"></div>',
                unsafe_allow_html=True)
    st.header("🚗 Your pricing calculator")
    with st.form("test_car"):
        st.text_input("Make",value="Ford")
        st.number_input("Mileage (miles)",min_value=0,value=75000)
        st.form_submit_button("Calculate seller Ask + wholesale Bid")
else:
    st.header("Other studio workspace")
