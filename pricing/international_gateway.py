"""One studio, two explicit markets. The gateway is locale-aware.

No browser geolocation, language-based redirection or tracking cookie.
Existing American routes remain the default and retain their original IDs.
"""
from __future__ import annotations
import streamlit as st
from html import escape
from pricing.i18n import tr
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
    """Compact, keyboard-accessible market choice in all 36 site locales."""
    selected=_resolve()
    st.markdown("""<style>
    .cps-quick-market{padding:10px 13px;margin:0 0 6px;border-radius:12px;
       background:#102b39;color:#fff}
    .cps-quick-market h2{color:#fff!important;font-size:1.1rem;margin:0 0 3px}
    .cps-quick-market p{margin:0;color:#e6f1ef;font-size:.87rem}
    .cps-quick-market-card{border-left:3px solid #267d6c;padding:3px 8px;
       margin:0 0 3px;min-height:0}
    .cps-quick-market-card small{display:block;line-height:1.34;
       color:inherit;font-size:.81rem}
    </style>""",unsafe_allow_html=True)
    st.markdown('<section class="cps-quick-market" role="region" aria-label="'+
                story_html("edition")+'"><h2>'+escape(tr("market_title"))+
                '</h2><p>'+escape(tr("market_intro"))+'</p></section>',
                unsafe_allow_html=True)
    left,right=st.columns(2,gap="small")
    with left:
        st.markdown('<div class="cps-quick-market-card"><strong>🇺🇸 '+
                    story_html("us_name")+'</strong><small>'+
                    story_html("us_desc")+'</small></div>',unsafe_allow_html=True)
        if st.button("🇺🇸 "+("✓ " if selected=="us" else "")+tr("open_us"),
                     key="cps_market_us",use_container_width=True,
                     type="primary" if selected=="us" else "secondary"):
            _choose("us")
    with right:
        st.markdown('<div class="cps-quick-market-card"><strong>🇪🇺 '+
                    story_html("eu_name")+'</strong><small>'+
                    story_html("eu_desc")+'</small></div>',unsafe_allow_html=True)
        if st.button("🇪🇺 "+("✓ " if selected=="eu" else "")+tr("open_eu"),
                     key="cps_market_eu",use_container_width=True,
                     type="primary" if selected=="eu" else "secondary"):
            _choose("eu")
    return selected
