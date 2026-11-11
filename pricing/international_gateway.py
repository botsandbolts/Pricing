"""One studio, two explicit markets. The gateway is locale-aware.

No browser geolocation, language-based redirection or tracking cookie.
Existing American routes remain the default and retain their original IDs.
"""
from __future__ import annotations
import streamlit as st
from html import escape
from pricing.i18n import tr, editorial_voice
from pricing.i18n_story import story, story_html

MARKETS={"us":"🇺🇸 United States","eu":"🇪🇺 Europe"}

def _resolve():
    request=str(st.query_params.get("market") or "").lower()
    if request in MARKETS:
        st.session_state["cps_market"]=request
        return request
    stored=st.session_state.get("cps_market")
    return stored if stored in MARKETS else "us"

def _choose(choice):
    if choice not in MARKETS:
        raise ValueError("Unknown market")
    st.session_state["cps_market"]=choice
    st.query_params["market"]=choice
    st.rerun()

def render_market_gateway():
    """Prominent two-market landing without breaking old US shared links."""
    selected=_resolve()
    st.markdown("""<style>
    .cps-globe{background:linear-gradient(125deg,#11273d,#204a55 65%,#355b56);
       color:#fff;border:1px solid #83afa8;border-radius:22px;
       box-shadow:0 14px 36px #13253525;padding:clamp(15px,3vw,28px);margin:3px 0 14px}
    .cps-globe strong,.cps-globe h1{color:#fff!important}
    .cps-globe .pre{font-size:.75rem;letter-spacing:.16em;font-weight:900;
       color:#d6dfab;margin:0 0 6px}
    .cps-globe h1{font-size:clamp(1.4rem,3vw,2.4rem);line-height:1.15;margin:0 0 6px}
    .cps-globe p{color:#e2ebef;margin:0;font-size:.93rem;line-height:1.5}
    .cps-market-card{border:1px solid #aebdbf;border-radius:12px;
       padding:13px 15px;margin:4px 0 8px;background:rgba(55,92,95,.045);
       min-height:89px}
    .cps-market-card b{font-size:1.08rem}
    .cps-market-card small{display:block;color:#61727b;line-height:1.4;margin-top:5px}
    </style>""",unsafe_allow_html=True)
    # The top-level HTML block must START at column zero. When indented
    # beneath a CSS block, Markdown treats <section> as a literal code block.
    landing_html=f"""<section class="cps-globe" aria-label="{story_html('edition')}">
    <div class="pre">CAR PRICE STUDIO · {story_html('edition')}</div>
    <h1>{escape(tr('market_title'))}</h1>
    <p>{escape(tr('market_intro'))}</p>
    </section>"""
    st.markdown(landing_html,unsafe_allow_html=True)
    headline, invitation = editorial_voice()
    st.markdown('<div class="cps-market-card" role="note"><b>'+escape(headline)+'</b>'
                '<small>'+escape(invitation)+'</small></div>',
                unsafe_allow_html=True)
    left,right=st.columns(2,gap="medium")
    with left:
        st.markdown('<div class="cps-market-card"><b>🇺🇸 '+story_html("us_name")+'</b>'
                    '<small>'+story_html("us_desc")+'</small></div>',
                    unsafe_allow_html=True)
        if st.button("🇺🇸 "+("✓ " if selected=="us" else "")+tr("open_us"),
                     key="cps_market_us",use_container_width=True,
                     type="primary" if selected=="us" else "secondary"):
            _choose("us")
    with right:
        st.markdown('<div class="cps-market-card"><b>🇪🇺 '+story_html("eu_name")+'</b>'
                    '<small>'+story_html("eu_desc")+'</small></div>',
                    unsafe_allow_html=True)
        if st.button("🇪🇺 "+("✓ " if selected=="eu" else "")+tr("open_eu"),
                     key="cps_market_eu",use_container_width=True,
                     type="primary" if selected=="eu" else "secondary"):
            _choose("eu")
    st.caption(story("disclaimer"))
    return selected
