"""Decision tools using the existing historical estimate; no new accuracy claims."""
import html
import json
from datetime import date
import streamlit as st


def render():
    st.markdown('## 🕵️ Deal Detective')
    st.caption('Bring an advert. Question the price. Keep your mechanic.')
    saved = st.session_state.get('last_explained_price')
    if not saved:
        st.info('Estimate a car in Price my car first. Your detective needs a suspect.')
        return
    model, prediction, fields, description = saved
    estimate = float(prediction['price'])
    title = f"{fields.get('year')} {fields.get('manufacturer')} {fields.get('model')}"
    st.markdown(f'### {title}')
    st.warning('Historical US asking-price estimate, referenced to May 2021. A seller price from today needs the separate Value over time analysis. This is not a live fair-market valuation.')
    seller = st.number_input('Seller asking price in US dollars', min_value=0., step=100., key='detective_seller')
    period = st.checkbox('I am comparing prices at the same May 2021 reference date')
    a,b,c = st.columns(3)
    a.metric('Historical asking estimate', f'${estimate:,.0f}')
    if seller and period:
        difference = seller-estimate
        b.metric('Advert minus estimate', f'${difference:+,.0f}')
        c.metric('Relative difference', f'{difference/estimate:+.1%}')
        st.write('Seller optimism: comes with ambition as standard.' if difference>estimate*.15 else 'A conversation starter. The inspection still gets the final word.')
    elif seller:
        st.info('Use Value over time to align market dates before interpreting a price gap.')
    st.caption('The gap is a model comparison, not guaranteed negotiating room or profit. No calibrated individual prediction interval is available.')
    tabs=st.tabs(['📤 Share receipt','🚗 My shortlist','🔎 Buyer briefing','📝 Advert workbench'])
    with tabs[0]:
        public={k:fields.get(k) for k in ['year','manufacturer','model','odometer','type','fuel','transmission']}
        receipt={'vehicle':public,'asking_estimate_usd':round(estimate,2),'reference_date':'2021-05-01','generated_on':date.today().isoformat(),'model':model,'warning':'Historical estimate, not a sale guarantee. No validated individual interval.'}
        st.write('Your share card excludes seller descriptions, contacts, visitor identifiers and tracking data.')
        card=f'''<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Deal Detective receipt</title><body style="font-family:system-ui;background:#241d35;color:#fff;padding:40px"><main style="max-width:600px;margin:auto"><p style="color:#d9fb77">CAR PRICE STUDIO · DEAL DETECTIVE</p><h1>{html.escape(title)}</h1><p>Historical asking estimate</p><strong style="font-size:64px;color:#d9fb77">${estimate:,.0f}</strong><p>May 2021 reference market · US dollars</p><p>Serious maths. Zero tyre-kicking.</p><p>Estimate ≠ sale guarantee. No validated individual prediction interval.</p><a style="color:#c9bcff" href="https://pricing-5a4k2zenhw2nte9xpshxs3.streamlit.app/">Investigate your own car →</a></main></body></html>'''
        st.download_button('Download share card · HTML',card,'car-price-receipt.html','text/html')
        st.download_button('Download valuation facts · JSON',json.dumps(receipt,indent=2),'valuation.json','application/json')
        st.code(f'{title}: historical asking estimate ${estimate:,.0f} (May 2021 reference). What would you pay?\nhttps://pricing-5a4k2zenhw2nte9xpshxs3.streamlit.app/',language=None)
    with tabs[1]:
        garage=st.session_state.setdefault('detective_garage',[])
        st.caption('Session-only shortlist: closes with this session. All estimates share the May 2021 reference date.')
        if st.button('Park this estimate in my shortlist'):
            entry={'Car':title,'Miles':fields.get('odometer'),'Historical ask USD':round(estimate),'Seller price USD':seller or None,'Model':model}
            if entry not in garage: garage.append(entry)
        if garage:
            st.dataframe(garage,use_container_width=True)
            st.download_button('Export shortlist',json.dumps(garage,indent=2),'shortlist.json','application/json')
            if st.button('Clear shortlist'): garage.clear();st.rerun()
    with tabs[2]:
        st.markdown('**Questions before you buy**')
        questions=['Can I see the title and maintenance records?','Has it had accident, flood or structural repairs?','May an independent mechanic inspect it?','What fees are added to the advertised price?','Which major services are due, and when were tyres and brakes replaced?']
        if fields.get('title_status')!='clean': questions.insert(0,'Please explain the title status and provide its documentation.')
        for q in questions: st.write('• '+q)
        st.download_button('Take the checklist with me','\n'.join(questions),'buyer-checklist.txt')
        st.caption('These are questions, not findings about this particular vehicle. Model contributions describe associations, not causal repair discounts.')
    with tabs[3]:
        st.text_area('Paste advert text for your own review',key='detective_advert',max_chars=8000)
        st.caption('This draft stays in session memory and is not saved to analytics. Copy the reviewed description into Price my car to include it in a prediction. Automated screenshot extraction is not enabled here.')
