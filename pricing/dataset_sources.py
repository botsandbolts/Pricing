"""Visible provenance for the two different economic price targets."""
ASK={'name':'Craigslist Used Cars','archive':'Used Cars Dataset · Tom / Zenodo','url':'https://zenodo.org/records/10457828','target':'Historical asking / advertised listing price','period':'2021 listing snapshot'}
BID={'name':'Used Car Auction Prices','archive':'Kaggle · tunguz','url':'https://www.kaggle.com/datasets/tunguz/used-car-auction-prices','target':'Reported historical auction selling price','period':'2014–2015 auction records'}


def render_sources(stacked=False):
    import streamlit as st
    a,b=(st.container(),st.container()) if stacked else st.columns(2)
    with a:
        st.markdown('**ASK · Craigslist Used Cars**')
        st.caption('Advertised asking prices · 2021 snapshot. A seller’s wish is not a completed sale.')
        st.markdown('[Dataset archive · Zenodo](https://zenodo.org/records/10457828)')
    with b:
        st.markdown('**BID · Used Car Auction Prices**')
        st.caption('Kaggle / tunguz · historical 2014–2015 reported auction sales. This is not a live bid feed.')
        st.markdown('[Dataset · Kaggle](https://www.kaggle.com/datasets/tunguz/used-car-auction-prices)')
    with st.expander('Data sources: what these two prices mean'):
        st.write('Ask models learn advertised Craigslist listing prices. Bid models learn the sellingprice field in the Used Car Auction Prices dataset; they estimate historical wholesale-auction equivalents, with the existing age and CPI transport. These datasets do not contain paired asking and final retail sale prices for the same cars.')
        st.caption('CarBids is the project name. The auction dataset used here is Kaggle / tunguz, not a verified scrape from the Cars & Bids auction platform.')
