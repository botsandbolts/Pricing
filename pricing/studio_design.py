"""Presentation-only design system for Car Price Studio. No prediction logic."""
from html import escape
import json
from pathlib import Path
import streamlit as st

# Kept local so the design works without third-party fonts or image hosts.
CSS = '<style>'+(Path(__file__).resolve().parents[1]/'assets/garage.css').read_text()+'</style>'


def install_style():
    st.markdown(CSS,unsafe_allow_html=True)
    from pricing.accessibility import install
    install()


def origin_badge():
    return '<span class="garage-origin"><span class="garage-flag" role="img" aria-label="German flag"></span>Made in Germany</span>'


def render_hero():
    from pricing.garage_art import car_svg,hero_svg
    st.markdown('<div class="garage-nav"><a class="garage-brand" href="#studio-workspace" aria-label="Car Price Studio · skip to workspace"><span class="garage-mark" aria-hidden="true">'+car_svg()+'</span><span class="garage-wordmark"><strong>car price studio</strong><small>A little car love. A lot of curiosity.</small></span></a>'+origin_badge()+'</div>',unsafe_allow_html=True)
    from pricing.accessibility import reading_options
    reading_options()
    page=st.session_state.get('_navigate_to') or st.session_state.get('landing_studio_workspace','🚗 Price my car')
    # Keep the large welcome on the calculator, and give other workspaces room.
    if page!='🚗 Price my car':
        return
    st.markdown('''<section class="garage-hero" aria-label="Welcome to Car Price Studio">
<div><div class="garage-eyebrow">Your friendly neighbourhood car lab</div><h1>Big car feelings.<br><span class="garage-accent">Clearer numbers.</span></h1><p>Price the car you have. Find the kind you love.<br>A friendly garage for exploring what the data says.</p><a class="garage-cta" href="#studio-workspace">Let’s meet your car <span aria-hidden="true">↗</span></a><small class="garage-micro">Free to explore · Open source · Made with curiosity</small></div>
<div class="garage-art" aria-hidden="true">'''+hero_svg()+'''</div></section>
<div class="garage-ticker"><span><b>01</b> Meet your car</span><span><b>02</b> Explore its value</span><span><b>03</b> Find your favourites</span><span class="garage-ticker-note">Historical data. Real models. A little perspective.</span></div>''',unsafe_allow_html=True)


def page_intro(kicker,title,description,icon='🚗'):
    shapes={'💚':'<path d="M16 27L5 16C-2 7 9 0 16 9C23 0 34 7 27 16Z" fill="#207b65"/>',
            '🔎':'<circle cx="13" cy="13" r="9" fill="#dceee2" stroke="#207b65" stroke-width="3"/><path d="M20 20L29 29" stroke="#207b65" stroke-width="4" stroke-linecap="round"/>',
            '🌳':'<path d="M16 30V16" stroke="#806342" stroke-width="4"/><path d="M16 2L28 16H23L30 23H2L9 16H4Z" fill="#207b65"/>'}
    art='<svg viewBox="0 0 32 32" aria-hidden="true">'+shapes.get(icon,'<circle cx="16" cy="16" r="12" fill="#f5d478"/><path d="M12 10L22 16L12 22Z" fill="#52695b"/>')+'</svg>'
    st.markdown(f'<div class="garage-page-intro"><div><div class="garage-section">{escape(kicker)}</div><h1>{escape(title)}</h1><p>{escape(description)}</p></div><span class="garage-page-icon" aria-hidden="true">{art}</span></div>',unsafe_allow_html=True)


def journey_steps(steps):
    st.markdown('<div class="garage-steps">'+''.join(f'<div class="garage-step"><b>{i:02}</b><div><strong>{escape(title)}</strong><small>{escape(note)}</small></div></div>' for i,(title,note) in enumerate(steps,1))+'</div>',unsafe_allow_html=True)


def market_guide():
    st.markdown('''<div class="garage-market-guide"><article><strong>🏷 Seller asking price</strong><p>The price a seller might advertise, before negotiation. Not a completed retail sale.</p></article><article><strong>🔨 Wholesale auction</strong><p>A comparable car’s estimated wholesale sale value. Not a live bid or dealer offer.</p></article></div>''',unsafe_allow_html=True)


def section(kicker,title):
    st.markdown(f'<div class="garage-section">{escape(kicker)}</div><div class="garage-card-title">{escape(title)}</div>',unsafe_allow_html=True)


def empty_receipt():
    st.markdown('''<span class="garage-receipt-tag">Your price receipt</span><div class="garage-card-title">The suspense is<br>almost over.</div><div class="garage-empty-number">$ — — —</div><p class="garage-empty-note">Tell us about the car, then hit <strong>Estimate asking price</strong>.<br><br>The model does the number-crunching. You retain the right to raise an eyebrow.</p><div class="garage-science-note">You’ll get an asking estimate and a separate wholesale-auction equivalent. Different markets. Different numbers.</div>''',unsafe_allow_html=True)


def sidebar_receipt(root):
    """Read metrics from actual installed bundles; never invent a score."""
    from pricing.residual_ui import model_root,bid_folder
    with st.sidebar:
        st.markdown('### The evidence locker')
        st.caption('Recorded original, unfiltered test MAE. Historical benchmarks—not a guarantee for your car.')
        for label,folder in [('Ask · Craigslist',model_root(root)/'ask'),('Bid · auction sales',bid_folder(root))]:
            file=folder/'evaluation.json'
            if file.exists():
                report=json.loads(file.read_text())
                m=report.get('assessment',{}).get('original_unfiltered_test',{})
                if m.get('mae_usd') is not None:st.metric(label+' · MAE',f"${m['mae_usd']:,.0f}",help=f"{m['rows']:,} original test cars. This benchmark has informed earlier model choices.")
        st.caption('Residual-trimmed validation/test scores are optimistic. See Evaluation for the full cohorts and limitations.')


def footer():
    with st.expander('Accessibility & browser test report'):
        from pricing.accessibility_report import render as render_accessibility_report
        render_accessibility_report()
    st.markdown('<footer class="garage-footer"><span><strong>Car Price Studio.</strong> Built with care. Best explored with curiosity.<br>Estimates ≠ sale guarantees. Keep your mechanic.</span>'+origin_badge()+'</footer>',unsafe_allow_html=True)
    st.caption('© 2026 [Amit Steinberg](https://de.linkedin.com/in/amit-steinberg-504546445) · Open-source software · [MIT License](https://github.com/botsandbolts/Pricing/blob/main/LICENSE)')
    st.caption('Datasets, pretrained models and dependencies retain their respective licences.')

