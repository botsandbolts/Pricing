"""Small, local SVG illustrations. Decorative vehicles are never listing photos."""
from html import escape


def car_svg(body='sedan', color='mint', *, label='Illustrated car'):
    colors={'mint':'#8cdbbe','coral':'#f5a48e','blue':'#8abadd','yellow':'#f5cc72',
            'red':'#df7c73','green':'#8cbd9b','white':'#fffdf6','black':'#5c6866',
            'grey':'#b6c3c0','silver':'#cbd7d5','brown':'#ba987a','orange':'#e9ac72',
            'purple':'#b8a4d7','custom':'#b8a4d7'}
    paint=colors.get(str(color).lower(), '#8cdbbe')
    body=str(body).lower()
    if body in ('truck','pickup'):
        shell='M38 115L60 110L78 60Q83 51 96 51H158Q170 51 176 66L190 111H276V93H306V147H37Z'
        windows='M91 65H116V105H77ZM128 65H153Q158 65 162 76L174 105H128Z'
    elif body in ('suv','van','wagon'):
        shell='M36 116L49 66Q52 53 67 53H221Q234 53 247 72L269 108L295 115Q308 120 308 137V147H35Z'
        windows='M66 67H114V104H57ZM127 67H172V104H127ZM185 67H218Q225 67 233 80L249 104H185Z'
    else:
        shell='M35 119L65 108L94 68Q102 57 119 57H198Q215 57 229 73L259 108L290 116Q306 122 306 138V147H34Z'
        windows='M109 71H151V105H84ZM164 71H197Q205 71 217 85L235 105H164Z'
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 340 190" role="img" aria-label="{escape(label,quote=True)}">
<ellipse cx="170" cy="163" rx="145" ry="12" fill="#193f3710"/>
<path d="M22 159H320" stroke="#b0c8be" stroke-width="2" stroke-linecap="round"/>
<path d="{shell}" fill="{paint}" stroke="#204b43" stroke-width="3" stroke-linejoin="round"/>
<path d="{windows}" fill="#f6fbf5" stroke="#204b43" stroke-width="2.5" stroke-linejoin="round"/>
<path d="M157 115V141M169 120H181" stroke="#204b43" stroke-width="2.5" stroke-linecap="round"/>
<rect x="282" y="122" width="17" height="9" rx="4" fill="#fff1b2"/>
<rect x="36" y="124" width="11" height="8" rx="3" fill="#dc7964"/>
<circle cx="88" cy="148" r="23" fill="#204b43"/><circle cx="88" cy="148" r="12" fill="#fffdf5"/><circle cx="88" cy="148" r="4" fill="#9ab7ac"/>
<circle cx="253" cy="148" r="23" fill="#204b43"/><circle cx="253" cy="148" r="12" fill="#fffdf5"/><circle cx="253" cy="148" r="4" fill="#9ab7ac"/>
<path d="M269 136Q276 141 283 136" fill="none" stroke="#204b43" stroke-width="2" stroke-linecap="round"/>
</svg>'''


def hero_svg():
    car=car_svg(label='A friendly mint car in an illustrated garage')
    # Nest the car in a larger scene; pure SVG keeps the hero crisp and local.
    car=car.replace('viewBox="0 0 340 190"', 'x="56" y="113" width="358" height="200" viewBox="0 0 340 190"')
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 470 340" role="img" aria-label="A happy car, a price receipt and a little love for the road">
<circle cx="241" cy="166" r="139" fill="#dceee2"/>
<path d="M64 284H424" stroke="#bed4c7" stroke-width="2" stroke-linecap="round"/>
<path d="M111 223V115Q111 103 123 103H339Q351 103 351 115V223" fill="#f5f8ed" stroke="#9abaac" stroke-width="2"/>
<path d="M128 124H334M128 139H334M128 154H334" stroke="#d4e2d2" stroke-width="3"/>
<path d="M91 103L229 44L370 103Z" fill="#fffaf0" stroke="#9abaac" stroke-width="2" stroke-linejoin="round"/>
<rect x="183" y="68" width="94" height="27" rx="13" fill="#204b43"/>
<text x="230" y="86" text-anchor="middle" font-size="11" font-weight="700" letter-spacing="2" fill="#fffdf4" font-family="sans-serif">THE GARAGE</text>
<g transform="rotate(9 374 139)"><rect x="332" y="76" width="98" height="123" rx="13" fill="#fffdf5" stroke="#d8d9c8" stroke-width="2"/>
<rect x="345" y="90" width="28" height="6" rx="3" fill="#91bba8"/><path d="M346 109H415M346 119H401" stroke="#dce5d8" stroke-width="4" stroke-linecap="round"/>
<path d="M347 159L364 145L380 151L408 130" fill="none" stroke="#207b65" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
<circle cx="407" cy="130" r="5" fill="#207b65"/><text x="346" y="183" fill="#547367" font-size="10" font-family="sans-serif">A little more clarity.</text></g>
{car}
<g transform="rotate(-10 77 135)"><rect x="33" y="105" width="88" height="57" rx="18" fill="#f8b39c"/><path d="M64 125C58 116 46 124 53 132L65 144L78 132C85 123 73 116 67 125Z" fill="#8c3d31"/><path d="M87 123H108M87 132H103" stroke="#a85947" stroke-width="3" stroke-linecap="round"/></g>
<path d="M393 242L398 229L403 242L416 247L403 252L398 265L393 252L380 247Z" fill="#e1b64f"/>
<path d="M73 71L76 63L79 71L87 74L79 77L76 85L73 77L65 74Z" fill="#e1b64f"/>
<path d="M158 307Q231 324 302 307" fill="none" stroke="#9abaac" stroke-width="2" stroke-dasharray="5 7" stroke-linecap="round"/>
</svg>'''


def car_art(car, *, compact=False):
    """Render a body-style sketch with an explicit illustration label."""
    import streamlit as st
    body=str(car.get('type') or 'sedan')
    color=str(car.get('paint_color') or 'mint')
    st.markdown(f'<div class="garage-car-art{" garage-car-art-small" if compact else ""}">{car_svg(body,color,label=body.title()+" illustration")}<span>Body-style illustration · not a listing photo</span></div>',unsafe_allow_html=True)
