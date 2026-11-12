"""Presentation-only design system for Car Price Studio. No prediction logic."""
from html import escape
from pricing.i18n import editorial_voice, locale
from pricing.i18n_story import story, story_html
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


def show_landing_banner():
    """Only the primary Price My Car workspace has the full welcome campaign.

    Historical shared-link routes and research panels still show the compact
    native brand bar; no full-size hero above every experimental panel.
    """
    if st.query_params.get('legal') in ('imprint', 'privacy'):
        return False
    active=st.session_state.get('landing_studio_workspace','🚗 Price my car')
    if st.query_params.get('garage') and 'landing_studio_workspace' not in st.session_state:
        return False
    return active in ('🚗 Price my car','🧬 Experimental pricing','🧬 Experimental models')


def landing_banner_html():
    """One short, fully localized call to action; no ad-like hero stack."""
    return f"""
<section class="garage-home-quick" aria-labelledby="garage-home-title"
 style="border:1px solid #a8c6bd;border-radius:15px;padding:12px 16px;
 margin:4px 0 9px;background:linear-gradient(115deg,#f4fbf8,#ecf6f0)">
 <h1 id="garage-home-title" style="font-size:clamp(1.18rem,2.5vw,1.65rem);
 color:#173d40;line-height:1.2;margin:0 0 3px">{story_html('us_hero')}</h1>
 <p style="font-size:.91rem;color:#294b50;line-height:1.4;margin:0 0 9px">
 {story_html('us_pitch')}</p>
 <a class="garage-home-cta" href="#studio-calculator" target="_self"
 aria-label="{story_html('us_cta')}"
 style="display:inline-block;padding:9px 15px;border-radius:10px">🚗 {story_html('us_cta')} →</a>
 <a href="https://github.com/botsandbolts/Pricing" target="_blank"
 rel="noopener noreferrer" style="margin-left:12px;color:#20665c;font-size:.86rem">
 {story_html('source')} ↗</a>
 <span style="display:block;font-size:.77rem;margin-top:6px;color:#46625a">
 {story_html('independent')}</span>
</section>
"""

def render_hero():
    from pricing.garage_art import car_svg
    tagline = story_html("us_hero" if st.query_params.get("market")!="eu" else "eu_hero")
    st.markdown('<div class="garage-nav"><a class="garage-brand" href="#studio-workspace" aria-label="Car Price Studio"><span class="garage-mark" aria-hidden="true">'+car_svg()+'</span><span class="garage-wordmark"><strong>car price studio</strong><small>'+tagline+'</small></span></a>'+origin_badge()+'</div>',unsafe_allow_html=True)
    from pricing.accessibility import reading_options
    reading_options()
    if show_landing_banner():
        st.markdown(landing_banner_html(),unsafe_allow_html=True)


def page_intro(kicker,title,description,icon='🚗'):
    shapes={'💚':'<path d="M16 27L5 16C-2 7 9 0 16 9C23 0 34 7 27 16Z" fill="#207b65"/>',
            '🔎':'<circle cx="13" cy="13" r="9" fill="#dceee2" stroke="#207b65" stroke-width="3"/><path d="M20 20L29 29" stroke="#207b65" stroke-width="4" stroke-linecap="round"/>',
            '🌳':'<path d="M16 30V16" stroke="#806342" stroke-width="4"/><path d="M16 2L28 16H23L30 23H2L9 16H4Z" fill="#207b65"/>'}
    art='<svg viewBox="0 0 32 32" aria-hidden="true">'+shapes.get(icon,'<circle cx="16" cy="16" r="12" fill="#f5d478"/><path d="M12 10L22 16L12 22Z" fill="#52695b"/>')+'</svg>'
    st.markdown(f'<div class="garage-page-intro"><div><div class="garage-section">{escape(kicker)}</div><h1>{escape(title)}</h1><p>{escape(description)}</p></div><span class="garage-page-icon" aria-hidden="true">{art}</span></div>',unsafe_allow_html=True)


def journey_steps(steps):
    st.markdown('<div class="garage-steps">'+''.join(f'<div class="garage-step"><b>{i:02}</b><div><strong>{escape(title)}</strong><small>{escape(note)}</small></div></div>' for i,(title,note) in enumerate(steps,1))+'</div>',unsafe_allow_html=True)


def market_guide():
    """Avoid confusing the U.S. wholesale auction with the European market."""
    st.caption(story("us_pitch"))


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
        st.caption('Original full-cohort validation is reported in Evaluation. Historical test data were not used to choose the new model.')


def footer():
    if locale() not in ("en-GB", "en-US"):
        st.markdown('<footer class="garage-footer"><span><strong>Car Price Studio.</strong> '+
                    story_html("disclaimer")+'</span><nav class="garage-legal-links">'+
                    '<a href="?legal=imprint">ⓘ</a><span aria-hidden="true"> · </span>'+
                    '<a href="?legal=privacy">🛡️</a><span aria-hidden="true"> · </span>'+
                    '<a href="https://github.com/botsandbolts/Pricing">'+story_html("source")+'</a></nav>'+
                    origin_badge()+'</footer>',unsafe_allow_html=True)
        st.caption(story("science_notice"))
        return
    with st.expander('Accessibility & browser test report'):
        from pricing.accessibility_report import render as render_accessibility_report
        render_accessibility_report()
    st.markdown('<footer class="garage-footer"><span><strong>Car Price Studio.</strong> Built with care. Best explored with curiosity.<br>Estimates ≠ sale guarantees. Keep your mechanic.</span><nav class="garage-legal-links" aria-label="Legal information"><a href="?legal=imprint">Imprint</a><span aria-hidden="true">·</span><a href="?legal=privacy">Privacy Policy</a></nav>'+origin_badge()+'</footer>',unsafe_allow_html=True)
    st.caption('© 2026 [Amit Steinberg](https://de.linkedin.com/in/amit-steinberg-504546445) · Open-source software · [MIT License](https://github.com/botsandbolts/Pricing/blob/main/LICENSE)')
    st.caption('Datasets, pretrained models and dependencies retain their respective licences.')

