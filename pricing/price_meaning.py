"""Shared, presentation-only definitions for the two economic price targets."""
from datetime import date
from html import escape
from math import isfinite


STYLE = """<style>
.price-meaning{margin:0 0 1.25rem;color:#241d35}
.price-meaning h3{font-size:1.25rem;margin:0 0 .7rem;letter-spacing:-.025em}
.price-meaning-grid{display:grid;grid-template-columns:1fr 1fr;gap:.8rem}
.price-meaning-card{background:#eee9ff;border:1px solid #d0c4f0;border-top:4px solid #6351e8;border-radius:15px;padding:1rem 1.15rem}
.price-meaning-card.auction{background:#edf3df;border-color:#cedbb4;border-top-color:#416529}
.price-meaning-card h4{font-size:1.05rem;margin:.4rem 0;color:#241d35}
.price-meaning-card p{font-size:.94rem;line-height:1.5;margin:.4rem 0}
.price-meaning-tag,.price-basis{font-size:.76rem;font-weight:750;letter-spacing:.025em}
.price-meaning-source{font-size:.78rem;color:#51495e;margin-top:.65rem}
.price-meaning-gap{margin-top:.8rem;padding:.85rem 1rem;background:#fff3e8;border:1px solid #ddb795;border-radius:12px;font-size:.94rem;line-height:1.5}
.price-basis{display:inline-block;background:#fff;border:1px solid #c7bed6;border-radius:8px;padding:.4rem .65rem;color:#3d334d;margin:.3rem 0 .55rem}
@media(max-width:640px){.price-meaning-grid{grid-template-columns:1fr}.price-meaning-card{padding:.85rem 1rem}.price-meaning-gap{font-size:.9rem}}
</style>"""


def date_label(value):
    """Format an explicit reference date; never substitute today's date."""
    if isinstance(value, date):
        return value.strftime('%d %b %Y')
    try:
        return date.fromisoformat(str(value)[:10]).strftime('%d %b %Y')
    except (TypeError, ValueError):
        return str(value)


def render_market_guide():
    import streamlit as st
    st.markdown(STYLE + '''<section class="price-meaning" aria-label="What the two prices mean">
<h3>Two prices. Two different questions.</h3>
<div class="price-meaning-grid">
<article class="price-meaning-card"><div class="price-meaning-tag">ASK · SELLER'S ADVERTISED PRICE</div>
<h4>What might a seller put in the advert?</h4>
<p>A seller's <strong>asking price</strong>, before negotiation. It is <strong>not a completed retail sale</strong>.</p>
<div class="price-meaning-source">Craigslist Used Cars · 2021 asking listings</div></article>
<article class="price-meaning-card auction"><div class="price-meaning-tag">AUCTION · WHOLESALE SALE ESTIMATE</div>
<h4>What might it sell for at auction?</h4>
<p>A model estimate of a <strong>wholesale-auction selling price</strong>. It is <strong>not a live bid or dealer offer</strong>.</p>
<div class="price-meaning-source">Used Car Auction Prices · 2014–2015 reported sales · Kaggle / tunguz</div></article>
</div>
<div class="price-meaning-gap"><strong>The gap is not dealer profit or a guaranteed discount.</strong> Different markets, vehicle populations and periods; no matched retail sale for the same car. A profit calculation would also need actual purchase and sale prices, fees and costs. Two price tags. No magic margin.</div>
</section>''', unsafe_allow_html=True)


def render_market_price(side, value, reference_date, *, adjusted=False, scenario=False):
    """Render an existing prediction unchanged, with its target and date attached."""
    import streamlit as st
    if side not in ('ask', 'bid'):
        raise ValueError('Unknown price target')
    asking = side == 'ask'
    st.markdown('**' + ('ASK · SELLER\'S ADVERTISED PRICE' if asking else 'AUCTION · WHOLESALE SALE ESTIMATE') + '**')
    if scenario:
        basis = 'Estimated for ' + date_label(reference_date) + ' · age + market adjusted'
    elif asking:
        basis = 'Historical asking market · May 2021'
    else:
        basis = ('Age/index-adjusted to ' if adjusted else 'Historical auction basis · ') + date_label(reference_date)
    st.markdown(STYLE + '<div class="price-basis">' + escape(basis) + '</div>', unsafe_allow_html=True)
    available = value is not None and isfinite(float(value))
    label = ('Seller asking price' if asking else 'Wholesale auction sale price') + (' · date-adjusted estimate' if scenario else ' · estimate')
    st.metric(label, f'${float(value):,.0f}' if available else 'Unavailable')
    st.write('What a seller might advertise — not what a buyer necessarily pays.' if asking else
             'What a wholesale auction sale might achieve — not a live bid, dealer offer or retail sale quote.')
    st.caption('Source: Craigslist Used Cars · 2021 asking listings.' if asking else
               'Source: Used Car Auction Prices · 2014–2015 reported sales · Kaggle / tunguz. Not the Cars & Bids website.')
    if scenario:
        st.caption('Experimental time adjustment of historical models; not validated against current individual-car sale outcomes.')


def render_price_pair(asking, auction, auction_date, *, asking_date='2021-05-01', scenario=False):
    import streamlit as st
    a, b = st.columns(2)
    with a:
        render_market_price('ask', asking, asking_date, scenario=scenario)
    with b:
        render_market_price('bid', auction, auction_date, adjusted=True, scenario=scenario)
    if not scenario and str(asking_date)[:7] != str(auction_date)[:7]:
        st.warning('Different price dates: asking is May 2021; auction is adjusted to ' + date_label(auction_date) +
                   '. Open Value over time to put both scenarios on the same date before comparing.')
    st.info('These are two market estimates, not a buy-low / sell-high profit calculation. The gap is not guaranteed negotiating room.')
