"""Presentation-only design system for Car Price Studio. No prediction logic."""
from html import escape
import json
from pathlib import Path
import streamlit as st

CSS = """
<style>
:root { --garage-ink:#241d35; --garage-purple:#6351e8; --garage-lime:#d9fb77; --garage-coral:#ff997d; --garage-paper:#f7f5ef; }
[data-testid="stAppViewContainer"] { background:radial-gradient(ellipse at 95% 0%,#e9e0ff 0%,transparent 35%),radial-gradient(ellipse at 0% 30%,#ecf7d2 0%,transparent 30%),var(--garage-paper); color:var(--garage-ink); }
[data-testid="stHeader"] { background:rgba(247,245,239,.94); }
[data-testid="stMainBlockContainer"] { max-width:1320px; padding:2rem 2.5rem 4rem; }
[data-testid="stSidebar"] { background:#eeebfa; border-right:1px solid #dcd6ed; }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top:1.5rem; }
h1,h2,h3,h4 { color:var(--garage-ink); letter-spacing:-.035em; }
[data-testid="stMarkdownContainer"] p { line-height:1.65; }
[data-testid="stCaptionContainer"] { color:#655f74; }
[data-testid="stWidgetLabel"] p { font-size:.85rem; font-weight:650; color:var(--garage-ink); }
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] { border-radius:12px!important; background:#fff!important; border-color:#ddd8e7!important; }
input,textarea { color:var(--garage-ink)!important; }
[data-baseweb="input"]:focus-within, [data-baseweb="select"]:focus-within, [data-baseweb="textarea"]:focus-within { outline:3px solid #d9d1ff; outline-offset:1px; }
[data-testid="stForm"] { border:0; padding:0; }
[data-testid="stButton"] button, [data-testid="stFormSubmitButton"] button, [data-testid="stDownloadButton"] button { border-radius:12px; font-weight:700; min-height:2.8rem; }
[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"], button[kind="primary"] { background:var(--garage-purple)!important; border:1px solid var(--garage-purple)!important; color:#fff!important; box-shadow:0 5px 0 #392c9e; transition:transform .15s,box-shadow .15s; }
[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"]:hover, button[kind="primary"]:hover { transform:translateY(-2px); box-shadow:0 7px 0 #392c9e; }
[data-testid="stTabs"] [role="tablist"] { gap:.5rem; border-bottom:1px solid #d9d2e5; padding-bottom:10px; overflow-x:auto; }
[data-testid="stTabs"] [role="tab"] { border-radius:100px; padding:8px 17px; height:auto; white-space:nowrap; background:#ede9f3; color:#463e56; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { background:var(--garage-ink); color:#fff; }
[data-testid="stTabs"] [role="tab"] p { font-weight:700; font-size:.88rem; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"], [data-testid="stTabs"] [data-baseweb="tab-border"] { display:none; }
[data-testid="stTabs"] [role="tabpanel"] { padding-top:1rem; }
[data-testid="stExpander"] { border-radius:15px; border-color:#ded7e8; background:#ffffff80; }
[data-testid="stMetricLabel"] p { font-weight:700; }
[data-testid="stMetricValue"] { font-variant-numeric:tabular-nums; letter-spacing:-.055em; }
[data-testid="stAlert"] { border-radius:13px; }
.st-key-vehicle_editor { background:white; padding:1.6rem; border:1px solid #e5dfeb; border-radius:24px; box-shadow:0 5px 24px #2c213708; }
.st-key-price_receipt { background:#eae4ff; padding:1.6rem; border:1px solid #d8cafa; border-radius:24px; }
.st-key-price_receipt [data-testid="stMetricValue"] { font-size:clamp(2.4rem,4vw,4rem); color:#4e3cc8; }
.st-key-price_receipt h3 { font-size:1.15rem; }
.st-key-auction_receipt { background:#e5edcf; border:1px solid #d3dfb0; border-radius:20px; padding:1.3rem; margin-top:1rem; }
.st-key-auction_receipt [data-testid="stMetricValue"] { color:#344d25; font-size:2.7rem; }
.garage-nav { display:flex; align-items:center; justify-content:space-between; gap:1rem; margin:.1rem 0 1.25rem; }
.garage-brand { display:flex; gap:.8rem; align-items:center; color:var(--garage-ink); text-decoration:none!important; }
.garage-mark { display:block; flex:0 0 56px; width:56px; height:56px; filter:drop-shadow(0 3px 0 #cfc7de); }
.garage-mark svg { display:block; width:100%; height:100%; }
.garage-wordmark { display:flex; flex-direction:column; gap:2px; line-height:1.05; }
.garage-wordmark strong { font-size:1.4rem; font-weight:850; letter-spacing:-.065em; }
.garage-wordmark small { font-size:.65rem; font-weight:750; text-transform:uppercase; letter-spacing:.17em; color:#776789; }
.garage-nav-note { font-size:.72rem; text-transform:uppercase; letter-spacing:.12em; color:#70677d; font-weight:700; }
.garage-hero { box-shadow:0 18px 50px #6351e811; display:grid; grid-template-columns:1.2fr 1fr; gap:2rem; align-items:center; padding:2.6rem 3rem; background:#e7e2fb; border:1px solid #d8d0ef; border-radius:28px; overflow:hidden; margin-bottom:1.2rem; }
.garage-eyebrow { display:flex; gap:.55rem; align-items:center; font-size:.68rem; font-weight:850; letter-spacing:.12em; text-transform:uppercase; margin-bottom:1rem; }
.garage-eyebrow b { background:var(--garage-ink); color:var(--garage-lime); padding:6px 10px; border-radius:40px; }
.garage-hero h1 { font-family:Arial,Helvetica,sans-serif; font-size:clamp(2.5rem,5.3vw,4.8rem); line-height:.98; font-weight:900; letter-spacing:-.07em; margin:0 0 1rem; padding:0; }
.garage-hero h1 .garage-accent { color:#6351d7; }
.st-key-story_receipt { background:linear-gradient(120deg,#ffffff,#f0edff);border:1px solid #d8ccef;border-radius:24px;padding:1.7rem 2rem;margin:1.5rem 0;box-shadow:0 12px 30px #3a285208; }
.formula-release { display:flex;gap:1.25rem;align-items:center;background:#271e3b;color:#faf7ff;border-radius:22px;padding:1.5rem 1.8rem;margin:.3rem 0 1.2rem;position:relative;overflow:hidden; }
.formula-release h2 { color:#e1fc9b;font-size:1.6rem;line-height:1.1;margin:0 0 .55rem; }
.formula-release p { font-size:.86rem;line-height:1.5;margin:0;color:#e0d8ec;max-width:750px; }
.formula-badge { font-size:.65rem;letter-spacing:.1em;font-weight:800;padding:8px 12px;border:1px solid #a9a0c2;border-radius:50px;white-space:nowrap;align-self:flex-start; }
.formula-number { font-size:4rem;font-weight:900;line-height:.85;color:#c7b6ff;margin-left:auto;text-align:center; }
.formula-number span { display:block;font-size:.65rem;letter-spacing:.12em;text-transform:uppercase;margin-top:.6rem;white-space:nowrap; }
@media(max-width:760px) { .formula-release { flex-wrap:wrap;padding:1.3rem;gap:.8rem; }.formula-release h2 {font-size:1.45rem}.formula-number {display:none}.st-key-story_receipt {padding:1.2rem} }
.garage-hero p { color:#534a63; font-size:1rem; max-width:420px; line-height:1.6; margin:0 0 1.4rem; }
.garage-cta { display:inline-block; background:var(--garage-ink); color:#fff!important; padding:13px 21px; border-radius:12px; font-size:.9rem; font-weight:750; text-decoration:none!important; box-shadow:0 4px 0 #b7a7db; }
.garage-cta:hover { background:#45346b; }
.garage-cta:focus-visible { outline:3px solid #6351e8; outline-offset:4px; }
.garage-art { position:relative; min-width:0; }
.garage-art svg { display:block; width:100%; height:auto; }
.garage-sticker { position:absolute; right:0; top:0; transform:rotate(7deg); background:var(--garage-coral); border:2px solid var(--garage-ink); border-radius:14px; padding:10px 14px; font-size:.76rem; font-weight:850; box-shadow:3px 3px 0 var(--garage-ink); text-align:center; line-height:1.3; }
.garage-ticker { display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem; padding:11px 6px 18px; color:#625970; font-size:.76rem; }
.garage-ticker strong { color:var(--garage-ink); }
.garage-section { font-size:.7rem; letter-spacing:.13em; text-transform:uppercase; font-weight:800; margin:0 0 .6rem; color:#75658a; }
.garage-card-title { font-size:1.6rem; font-weight:850; line-height:1.12; letter-spacing:-.055em; margin:0 0 .5rem; }
.garage-empty-number { font-size:4.4rem; font-weight:850; letter-spacing:-.08em; line-height:1.15; color:#8678ba; padding:1.5rem 0 1rem; }
.garage-empty-note { font-size:.92rem; color:#605177; line-height:1.65; }
.garage-receipt-tag { display:inline-block; padding:5px 9px; border-radius:50px; font-weight:750; font-size:.67rem; letter-spacing:.08em; text-transform:uppercase; background:#fff8; color:#584294; margin-bottom:1rem; }
.garage-science-note { background:#eee9fa; border-left:4px solid #9582df; padding:1rem 1.2rem; border-radius:0 14px 14px 0; margin:1rem 0; font-size:.86rem; color:#534762; }
.garage-footer { border-top:1px solid #dcd5e6; margin-top:2.5rem; padding-top:1.3rem; display:flex; gap:1rem; flex-wrap:wrap; justify-content:space-between; font-size:.75rem; color:#766b81; }
.garage-footer strong { color:#463750; }
@media(max-width:900px) { [data-testid="stMainBlockContainer"] { padding:1.2rem 1.1rem 3rem; } .garage-hero { gap:1rem; padding:2rem; } .garage-hero h1 { font-size:3.4rem; } .garage-nav-note { display:none; } }
@media(max-width:640px) { .garage-hero { grid-template-columns:1fr; padding:1.7rem; gap:.5rem; border-radius:22px; } .garage-hero h1 { font-size:3.3rem; } .garage-art { max-width:360px; width:100%; margin:auto; } .garage-sticker { padding:7px 10px; font-size:.68rem; } .garage-ticker { gap:.7rem; justify-content:flex-start; } .st-key-vehicle_editor,.st-key-price_receipt { padding:1.15rem; } [data-testid="stTabs"] [role="tab"] { padding:8px 13px; } }
@media(prefers-reduced-motion:reduce) { *,*:before,*:after { animation:none!important; transition:none!important; scroll-behavior:auto!important; } }
</style>
"""


def install_style():
    st.markdown(CSS,unsafe_allow_html=True)


def render_hero():
    mark=(Path(__file__).resolve().parents[1]/'assets/pricing-logo.svg').read_text()
    st.markdown('''<div class="garage-nav"><a class="garage-brand" href="#studio-calculator" aria-label="Car Price Studio · go to calculator"><span class="garage-mark">'''+mark+'''</span><span class="garage-wordmark"><strong>car price studio</strong><small>Powered by curiosity + CarBids</small></span></a><div class="garage-nav-note">The open garage for curious car people</div></div>
<section class="garage-hero" aria-label="Car Price Studio introduction">
<div><div class="garage-eyebrow"><b>Free to try</b> Built for your inner car nerd</div><h1>Less haggle.<br><span class="garage-accent">More signal.</span></h1><p>A car, a few details, a little machine learning.<br>Get a price estimate. Then interrogate the maths.</p><a class="garage-cta" href="#studio-calculator">Let’s price your car ↗</a></div>
<div class="garage-art"><div class="garage-sticker">“I know what<br>I’ve got.”<br>Let’s check.</div>
<svg viewBox="0 0 500 290" role="img" aria-label="Playful lime-green car on a violet test track">
<ellipse cx="256" cy="248" rx="205" ry="20" fill="#c6b9e7"/>
<path d="M35 241H475" stroke="#382b50" stroke-width="3" stroke-linecap="round"/>
<path d="M48 128H121M25 147H103M59 167H100" stroke="#a193c2" stroke-width="5" stroke-linecap="round"/>
<path d="M104 172L128 130Q138 105 165 104H284Q304 104 325 130L355 164L404 174Q427 179 430 201L429 220H99L92 201Z" fill="#d9fb77" stroke="#30243d" stroke-width="4"/>
<path d="M149 135Q155 121 172 121H210V161H135Z" fill="#fff9e8" stroke="#30243d" stroke-width="3"/>
<path d="M228 121H279Q289 121 303 138L321 161H228Z" fill="#fff9e8" stroke="#30243d" stroke-width="3"/>
<path d="M222 173V212M242 180H258" stroke="#30243d" stroke-width="3" stroke-linecap="round"/>
<rect x="394" y="182" width="25" height="13" rx="6" fill="#fff9e8" stroke="#30243d" stroke-width="3"/>
<rect x="98" y="184" width="17" height="10" rx="3" fill="#ff997d" stroke="#30243d" stroke-width="3"/>
<circle cx="160" cy="218" r="30" fill="#30243d"/><circle cx="160" cy="218" r="16" fill="#e9e2fa"/><circle cx="160" cy="218" r="5" fill="#30243d"/>
<circle cx="365" cy="218" r="30" fill="#30243d"/><circle cx="365" cy="218" r="16" fill="#e9e2fa"/><circle cx="365" cy="218" r="5" fill="#30243d"/>
<path d="M172 83L186 66M202 79L208 57M232 80L230 62" stroke="#6351e8" stroke-width="4" stroke-linecap="round"/>
<path d="M50 67L55 55L60 67L73 72L60 77L55 90L50 77L37 72Z" fill="#ff997d"/>
<text x="192" y="276" fill="#6b5e7d" font-size="10" font-family="Arial,sans-serif" letter-spacing="1.4">SMALL CAR. BIG DATA ENERGY.</text>
</svg></div></section>
<div class="garage-ticker"><span><strong>01</strong> Pick a car</span><span><strong>02</strong> Tell its story</span><span><strong>03</strong> Get the numbers</span><span>✦ Historical data. Real models. No crystal ball.</span></div>''',unsafe_allow_html=True)


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
    st.markdown('''<footer class="garage-footer"><span><strong>Car Price Studio.</strong> Serious maths. Zero tyre-kicking.</span><span>Estimates ≠ sale guarantees. Keep your mechanic.</span></footer>''',unsafe_allow_html=True)
    st.caption('© 2026 [Amit Steinberg](https://de.linkedin.com/in/amit-steinberg-504546445) · Open-source software · [MIT License](https://github.com/botsandbolts/Pricing/blob/main/LICENSE)')
    st.caption('Datasets, pretrained models and dependencies retain their respective licences.')

