"""Local accessibility QA only: illustrative fixtures, no model-serving changes."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import streamlit as st
from unittest.mock import patch
st.set_page_config(layout='wide',initial_sidebar_state='collapsed')
from pricing.studio_design import install_style,render_hero,market_guide,footer
from pricing.navigation import render_navigation
install_style();render_hero();page=render_navigation()
if page=='🚦 Pick your garage':
    import pricing.taste_garage_ui as ui
    root=Path('/tmp/garage-a11y-fixture');folder=root/'data/taste-garage';folder.mkdir(parents=True,exist_ok=True)
    (folder/'manifest.json').write_text('{}');(folder/'profiles-25.json').write_text('{}')
    def catalogue(*args):
        n=args[-1];car=dict(year=2015,manufacturer='Ford',model='Focus',price=10000,observed_date='2021-05-01',odometer=50000,type='sedan')
        return {'rows':400,'density_clusters':25,'noise_fraction':.1,'profiles':[{'prototype':car,'position':i,'name':f'Profile {i}','cars':40} for i in range(n)]},{}
    with patch.object(ui,'catalogue',catalogue),patch.object(ui,'price_view',lambda *args:8000.):ui.render(root)
else:
    market_guide()
    st.header('Your car. Both markets. One button.')
    with st.form('qa_form'):
        st.text_input('Make',value='Ford')
        st.number_input('Mileage (miles)',min_value=0,value=50000)
        st.text_area('Seller description')
        st.form_submit_button('Estimate asking + auction prices')
footer()
