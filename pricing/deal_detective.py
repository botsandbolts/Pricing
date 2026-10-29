"""Deal Detective: decision support and explicit sharing; no current-value claims."""
import base64
import hashlib
import html
import io
import json
import uuid
from datetime import date, timedelta
from pathlib import Path

import streamlit as st
from pricing.deal_tools import (APP_URL,REFERENCE_DATE,safe_fields,car_title,share_link,read_share,
    caption,social_links,parse_advert,ocr_image,card_png,reference_cases,comparables,daily_case)

STYLE='''<style>
.deal-hero{padding:26px 30px;border-radius:24px;background:linear-gradient(115deg,#241d35,#514079);color:white;margin-bottom:20px}
.deal-hero h2{color:#d9fb77!important;font-size:2.4rem!important;margin:0}.deal-hero p{color:#eee9ff;margin:5px 0}
.deal-kicker{letter-spacing:.12em;font-size:.7rem;font-weight:800;text-transform:uppercase;color:#d9fb77}
.deal-note{padding:16px 20px;border-radius:14px;background:#f0edff;border-left:5px solid #6351e8;color:#33264d}
@media(max-width:600px){.deal-hero{padding:20px}.deal-hero h2{font-size:1.8rem!important}}
</style>'''


def apply_vehicle(fields,text='',seller=None,tab='🚗 Price my car'):
    """Widget callback runs before the calculator constructs its form."""
    f=safe_fields(fields)
    enums={'type':['sedan','SUV','truck','pickup','hatchback','coupe','wagon','van','convertible','other','unknown'],
      'condition':['unknown','excellent','good','fair','like new','new','salvage'],
      'fuel':['gas','diesel','hybrid','electric','other','unknown'], 'transmission':['automatic','manual','other','unknown'],
      'drive':['unknown','fwd','rwd','4wd'],'title_status':['clean','rebuilt','salvage','lien','missing','parts only','unknown'],
      'cylinders':['unknown','3 cylinders','4 cylinders','5 cylinders','6 cylinders','8 cylinders','10 cylinders','12 cylinders','other'],
      'paint_color':['unknown','white','black','silver','grey','blue','red','green','brown','yellow','orange','purple','custom']}
    for k,values in enums.items():
        v=f.get(k,'unknown');v='SUV' if k=='type' and str(v).lower()=='suv' else str(v).lower()
        f[k]=v if v in values else 'unknown'
    for k,v in f.items():st.session_state['vehicle_'+k]=v
    st.session_state['seller_description']=text[:8000]
    if seller is not None:st.session_state['detective_seller']=float(seller)
    st.session_state.pop('last_explained_price',None)
    st.session_state.pop('pricing_visual_explanation',None)
    st.session_state['studio_tabs']=tab


def move_to(tab):st.session_state['studio_tabs']=tab

def open_share():
    st.session_state['studio_tabs']='🕵️ Deal Detective'
    st.session_state['detective_tabs']='📤 Share & go viral'


def draft_parse():
    draft=parse_advert(st.session_state.get('deal_advert_text',''))
    st.session_state['deal_draft']=draft
    for k in ('year','manufacturer','model','odometer'):
        st.session_state['review_'+k]=draft['fields'].get(k,2015 if k=='year' else 0 if k=='odometer' else '')
    st.session_state['review_price']=float(draft['seller_price'] or 0)
    st.session_state['review_description']=draft['text']


def apply_review():
    f=dict((st.session_state.get('deal_draft') or {}).get('fields',{}))
    for k in ('year','manufacturer','model','odometer'):f[k]=st.session_state.get('review_'+k)
    apply_vehicle(f,st.session_state.get('review_description',''),st.session_state.get('review_price'))


def render_shared():
    token=st.query_params.get('garage')
    if not token:return
    try:cars=read_share(token)
    except ValueError as exc:st.warning(str(exc));return
    st.markdown('### 👋 A friend has opened the garage for you')
    st.caption('These are shared vehicle inputs, not trusted price predictions. Seller text is omitted. Recalculate with the installed model; estimates can differ from the sender’s text-based result.')
    for i,f in enumerate(cars):
        with st.container(border=True):
            st.markdown('**'+car_title(f)+'**')
            st.caption(f"{f.get('odometer',0):,} miles · {f.get('state','state unspecified')}")
            st.button('Load this car into the calculator',key=f'load_shared_{i}',on_click=apply_vehicle,args=(f,))


def render_importer():
    st.markdown('### 📸 From advert to investigation')
    st.caption('Paste text or read a screenshot locally on this server. No external AI service receives the image. Review the extraction before loading the calculator.')
    left,right=st.columns([1.35,1])
    with left:
        st.text_area('Advert text',key='deal_advert_text',height=140,max_chars=16000,placeholder='2015 Toyota Corolla SE · 95,000 miles · $13,500…')
        st.button('Extract vehicle details',on_click=draft_parse,type='primary')
    with right:
        image=st.file_uploader('Advert screenshot · PNG/JPEG, up to 8 MB',type=['png','jpg','jpeg'],key='deal_advert_image')
        if st.button('Read screenshot',disabled=image is None):
            try:
                with st.spinner('Reading the small print. Finally, a volunteer…'):
                    text=ocr_image(image.getvalue())
                st.session_state['deal_ocr_pending']=text
                # text widget was already mounted; stage its new value for the next run.
                st.rerun()
            except (ValueError,RuntimeError,OSError) as exc:st.warning('Could not read the screenshot. Try a tighter crop or paste its text. '+str(exc)[:120])
        st.caption('Crop to the advert. Screenshots and OCR drafts stay in session memory; they are not added to visitor analytics.')
    draft=st.session_state.get('deal_draft')
    if draft:
        with st.container(border=True):
            st.markdown('**Review first. The robot sometimes reads “miles” as “smiles”.**')
            a,b=st.columns(2)
            a.number_input('Review year',min_value=1900,max_value=2021,key='review_year')
            b.number_input('Review mileage · miles',min_value=0,max_value=2000000,key='review_odometer')
            a.text_input('Review make',key='review_manufacturer',max_chars=80)
            b.text_input('Review model / trim',key='review_model',max_chars=80)
            st.number_input('Review seller price · USD',min_value=0.,key='review_price')
            st.text_area('Review description',key='review_description',max_chars=8000,height=100)
            st.caption('Kilometres are converted to miles when detected. OCR may miss trim, fees or currency. This release values US vehicles in USD, model years through 2021.')
            st.button('Use reviewed details → Price my car',on_click=apply_review,type='primary',disabled=not st.session_state.get('review_manufacturer') or not st.session_state.get('review_model'))
    with st.expander('Clear my advert draft'):
        st.button('Clear text and extracted details',on_click=clear_draft)


def clear_draft():
    for key in list(st.session_state):
        if key.startswith(('review_','deal_draft','deal_advert_text','deal_ocr_pending')):st.session_state.pop(key,None)


def social_component(url,text,png,title):
    import streamlit.components.v2 as components
    folder=Path(__file__).parent/'social'
    widget=components.component('deal_social',html=(folder/'share.html').read_text(),css=(folder/'share.css').read_text(),js=(folder/'share.js').read_text())
    widget(key='deal_social_buttons',data={'url':url,'caption':text,'png':base64.b64encode(png).decode(),'title':title})


def render_share(fields,estimate,model):
    st.markdown('### 📣 Send the verdict to the group chat')
    tone=st.radio('Caption personality',['Playful','Professional','Challenge'],horizontal=True,key='share_tone')
    include=st.checkbox('Include these vehicle details in a public link',key='share_include',help='Year, make/model, mileage and selected vehicle characteristics. Anyone with the link can read them. No seller description, price prediction, visitor ID or location tracking is included.')
    with st.expander('Preview exactly what goes in the public link'):
        st.json(safe_fields(fields) if include else {'destination':'Pricing home page only'})
    url=share_link([fields]) if include else APP_URL
    text=caption(fields,estimate,tone)
    portrait=st.radio('Card format',['Landscape · 1200 × 630','Portrait · 1080 × 1350'],horizontal=True,key='share_format').startswith('Portrait')
    signature=hashlib.sha256(json.dumps([safe_fields(fields),estimate,url,portrait],sort_keys=True).encode()).hexdigest()
    cached=st.session_state.get('deal_card_cache',{})
    if cached.get('signature')!=signature:
        cached={'signature':signature,'png':card_png(fields,estimate,url,portrait)};st.session_state['deal_card_cache']=cached
    png=cached['png']
    preview,controls=st.columns([1.2,1])
    with preview:st.image(png,caption='Ready for social media. Reference date stays on the image.',width='stretch')
    with controls:
        st.download_button('⬇ Download image card · PNG',png,'car-price-studio.png','image/png',type='primary')
        social_component(url,text,png,car_title(fields))
        links=social_links(url,text)
        a,b=st.columns(2)
        for i,(network,link) in enumerate(links.items()):
            (a if i%2==0 else b).link_button(network,link,width='stretch')
        st.caption('LinkedIn/Facebook share the link; paste the caption and attach the PNG if wanted. For Instagram, use your phone’s image-share menu or download and attach the portrait card. You choose the audience and confirm posting.')
    st.text_area('Caption · select/copy if your browser blocks the copy button',value=text+'\n'+url,height=150,key='caption_'+signature[:12]+'_'+tone)
    st.text_input('Share link · select/copy',value=url,key='url_'+signature[:12])
    receipt={'vehicle':safe_fields(fields),'asking_estimate_usd':round(estimate,2),'reference_date':REFERENCE_DATE,'generated_on':date.today().isoformat(),'model':model,'notes':'Historical asking estimate. Seller description excluded; facts-only links may recompute differently. No validated individual prediction interval.'}
    st.download_button('Valuation facts · JSON',json.dumps(receipt,indent=2),'valuation.json','application/json')


def park(fields,text,estimate,model,seller):
    garage=st.session_state.setdefault('detective_garage',[])
    identity=hashlib.sha256(json.dumps([safe_fields(fields),text,model],sort_keys=True).encode()).hexdigest()[:16]
    existing=next((i for i,c in enumerate(garage) if c['id']==identity),None)
    entry={'id':identity,'fields':safe_fields(fields),'text':text,'estimate':estimate,'model':model,'seller_price':seller or None}
    if existing is not None:garage[existing]=entry
    elif len(garage)<3:garage.append(entry)
    else:st.session_state['garage_full']=True


def remove_car(identity):
    st.session_state['detective_garage']=[c for c in st.session_state.get('detective_garage',[]) if c['id']!=identity]
    st.session_state.pop('garage_full',None)


def render_garage(fields,text,estimate,model,seller,estimate_fn,bid_fn):
    st.markdown('### 🚗 Three cars enter. Your spreadsheet retires.')
    st.caption('A session-only shortlist. Every displayed estimate is anchored to May 2021. Seller quotes may refer to another date; they are not treated as model targets.')
    st.button('Park this estimate',on_click=park,args=(fields,text,estimate,model,seller),type='primary')
    if st.session_state.get('garage_full'):st.info('Your three-car garage is full. Remove a car to park another.')
    garage=st.session_state.get('detective_garage',[])
    if not garage:st.info('Park a car, change its details in the calculator, then park another. Up to three.')
    for col,car in zip(st.columns(max(1,len(garage))),garage):
        with col,st.container(border=True):
            st.markdown('**'+car_title(car['fields'])+'**')
            st.metric('Saved historical asking estimate',f"${car['estimate']:,.0f}")
            st.caption(f"{car['fields'].get('odometer',0):,} miles")
            st.button('Inspect this car',key='inspect_'+car['id'],on_click=apply_vehicle,args=(car['fields'],car['text'],car['seller_price']))
            st.button('Value through time',key='timeline_'+car['id'],on_click=apply_vehicle,args=(car['fields'],car['text'],car['seller_price'],'📅 Value over time'))
            st.button('Remove',key='remove_'+car['id'],on_click=remove_car,args=(car['id'],))
    if garage and estimate_fn and st.button('Compare all with the selected model',type='primary'):
        rows=[]
        with st.spinner('Assembling the committee of tiny decision trees…'):
            for car in garage:
                f=car['fields'];row={'Car':car_title(f),'Mileage':f.get('odometer'),'Ask · May 2021 USD':None,'Bid · May 2021 USD':None,'Model':model}
                try:row['Ask · May 2021 USD']=round(float(estimate_fn(car['text'],f)['price']),2)
                except Exception:row['Ask status']='Unsupported or unavailable'
                if f.get('year',2022)>2015:row['Bid status']='Outside auction-era model years'
                elif bid_fn:
                    try:row['Bid · May 2021 USD']=round(float(bid_fn(f,REFERENCE_DATE)),2)
                    except Exception:row['Bid status']='Unsupported or unavailable'
                rows.append(row)
        st.session_state['garage_comparison']={'signature':json.dumps([(c['id'],c['seller_price']) for c in garage])+model,'rows':rows}
    compared=st.session_state.get('garage_comparison')
    if compared and compared['signature']==json.dumps([(c['id'],c['seller_price']) for c in garage])+model:
        st.dataframe(compared['rows'],hide_index=True,width='stretch')
        import pandas as pd
        chart=pd.DataFrame(compared['rows']).set_index('Car')[['Ask · May 2021 USD','Bid · May 2021 USD']]
        st.bar_chart(chart,color=['#6351e8','#72a23d'])
        st.caption('Ask and Bid are different markets. The spread is not guaranteed profit. No intervals are implied by these bars.')
    if garage:
        if st.checkbox('Create a public comparison link containing only these cars’ vehicle facts',key='share_garage_consent'):
            url=share_link(garage);st.text_input('Comparison link',url)
            st.link_button('Share comparison on WhatsApp',social_links(url,'Which would you buy? Three cars. One group chat. No peace.')['WhatsApp'])
        public=[{'fields':c['fields'],'historical_asking_estimate_usd':c['estimate'],'reference_date':REFERENCE_DATE,'model':c['model']} for c in garage]
        st.download_button('Export shortlist · no seller text',json.dumps(public,indent=2),'car-shortlist.json','application/json')
    with st.expander('Ownership budget · editable assumptions, not a model forecast'):
        annual=st.number_input('Annual driving · miles',0,50000,12000,step=1000)
        fuel=st.number_input('Fuel economy · miles per US gallon',1.,200.,30.)
        gallon=st.number_input('Fuel price · USD per US gallon',0.,30.,4.)
        other=st.number_input('Annual maintenance + insurance budget · USD',0.,100000.,2000.,step=100.)
        st.metric('Illustrative annual running budget',f'${annual/fuel*gallon+other:,.0f}')
        st.caption('Excludes financing, tax, parking and depreciation. Use Value over time for explicitly modelled depreciation scenarios.')


def render_evidence(fields):
    data=reference_cases();rows=comparables(fields,data['cases'])
    st.markdown('### 🔎 Show me the receipts. Historical ones.')
    st.caption('Nearest make/model examples from a small historical wholesale-auction sample. These are provider-reported sales from 2014–2015, not contemporary retail listings or independently verified settlements.')
    if rows:
        st.dataframe([{'Car':car_title(c),'Miles':c['odometer'],'Sale date':c['sale_date'],'Reported auction USD':c['price'],'Year gap':c['year_difference'],'Mileage gap':c['mileage_difference']} for c in rows],hide_index=True,width='stretch')
        st.caption('Match distance = absolute year gap + mileage gap / 25,000 + 2 for a partial model-name match. Same make required; model text must match or overlap; at most three model years apart. No hidden price-based ranking.')
    else:st.info('No close match in this small gallery. We would rather leave a space than put a Ferrari in your Corolla’s family tree.')
    with st.expander('Evidence provenance and limitations'):
        st.write(data['selection']);st.write(data['purpose']);st.write(data['modifications'])
        st.markdown('[Used Car Auction Prices · Bojan Tunguz](https://www.kaggle.com/datasets/tunguz/used-car-auction-prices) · CC0: Public Domain.')
        st.caption('This gallery may overlap the fitted models’ training data. It is illustrative evidence, not new evaluation data. Prices are not inflation-adjusted here.')


def render_game(bid_fn):
    data=reference_cases();cases=data['cases'];round_no=st.session_state.get('challenge_round',0)
    challenge=daily_case(cases,date.today()+timedelta(days=round_no));identity=challenge['id']+':'+date.today().isoformat()
    st.markdown('### 🎯 Beat the model. Keep your day job.')
    st.caption('Guess a historical wholesale sale price, in nominal USD at the date shown. A fresh daily case, plus practice rounds. This game is entertainment: the model may have trained on these cars.')
    st.markdown('**'+car_title(challenge)+'**')
    a,b,c=st.columns(3);a.metric('Mileage',f"{challenge['odometer']:,}");b.metric('Reported sale date',challenge['sale_date']);c.metric('Market','Wholesale auction')
    with st.form('guess_'+identity):
        guess=st.number_input('Your price guess · USD',min_value=1.,max_value=1000000.,value=10000.,step=250.)
        reveal=st.form_submit_button('Lock my guess & reveal',type='primary')
    if reveal:
        model_price=None
        if bid_fn:
            try:model_price=float(bid_fn(challenge,challenge['sale_date']))
            except Exception:pass
        result={'id':identity,'car':car_title(challenge),'guess':guess,'actual':challenge['price'],'model':model_price}
        previous=st.session_state.setdefault('challenge_scores',[])
        if not any(r['id']==identity for r in previous):previous.append(result)
        st.session_state['challenge_result']=next(r for r in previous if r['id']==identity)
    result=st.session_state.get('challenge_result')
    if result and result['id']==identity:
        a,b,c=st.columns(3);a.metric('Provider-reported sale',f"${result['actual']:,.0f}");b.metric('Your absolute error',f"${abs(result['guess']-result['actual']):,.0f}")
        if result['model'] is not None:
            c.metric('Model absolute error',f"${abs(result['model']-result['actual']):,.0f}")
            st.write('Human wins this round. The trees are requesting a recount.' if abs(result['guess']-result['actual'])<abs(result['model']-result['actual']) else 'The trees take this round. Rematch?')
        else:st.caption('Model prediction unavailable for this case; no invented model score.')
        st.caption('Source: Used Car Auction Prices. A reported selling price is not independent proof of settlement. This one-case comparison is not a validation metric.')
    st.button('Another practice car',on_click=next_round)
    scores=st.session_state.get('challenge_scores',[])
    if scores:
        st.dataframe([{'Car':r['car'],'Your error USD':round(abs(r['guess']-r['actual'])),'Model error USD':round(abs(r['model']-r['actual'])) if r['model'] is not None else None} for r in scores],hide_index=True,width='stretch')
        st.caption('Your session scoreboard is private and disappears with the session. First guess per case counts.')


def next_round():st.session_state['challenge_round']=st.session_state.get('challenge_round',0)+1


def render_briefing(fields,description):
    questions=['Can I see the title and maintenance records?','Has it had accident, flood or structural repairs?','May an independent mechanic inspect it?','What fees are added to the advertised price?','Which major services are due, and when were tyres and brakes replaced?']
    if fields.get('title_status')!='clean':questions.insert(0,'Please explain the title status and provide documentation.')
    if fields.get('condition','unknown')=='unknown':questions.insert(0,'How would you document its condition, including faults and warning lights?')
    if not description.strip():questions.append('Could you provide a detailed advert and service history? The text branch currently has no backstory.')
    st.markdown('### 🧰 Your buyer briefing')
    for q in questions:st.write('• '+q)
    st.download_button('Take the checklist with me','\n'.join(questions),'buyer-checklist.txt')
    st.caption('Questions, not findings about this car. Seller claims and OCR results are unverified. Model contributions describe associations, not causal repair discounts.')


def render_feedback(fields,estimate,model):
    from pricing.sale_feedback import validate_outcome,submit,delete_own
    from pricing.analytics import configured,_setting
    st.markdown('### 🧾 What did it actually sell for?')
    st.write('Help the studio learn from reality. Reality has declined to use our contact form, so we need you.')
    st.caption('Optional private feedback. All submissions start unverified and are excluded from automatic training. Market/date and documentary evidence require operator review. No submissions are shown publicly.')
    with st.form('sale_feedback'):
        a,b=st.columns(2)
        sold=a.number_input('Actual completed sale price · USD',min_value=1.,max_value=1000000.,value=10000.,step=100.)
        when=b.date_input('Completed sale date',date.today(),max_value=date.today())
        market=st.selectbox('Sale market',['private_sale','dealer_retail','wholesale_auction'],format_func=lambda v:v.replace('_',' ').title())
        evidence=st.file_uploader('Optional REDACTED sale receipt · PNG/JPEG, under 4 MB',type=['png','jpg','jpeg'],key='sale_receipt')
        st.caption('Remove names, addresses, phone numbers, signatures, plates, VIN and payment details before attaching evidence. Evidence is stored privately in Turso for up to 90 days; image metadata is removed. Never attach an unredacted bill.')
        confirmed=st.checkbox('I confirm this is a completed sale, the vehicle facts above are correct, and any evidence is redacted.')
        consent=st.checkbox('I agree to submit these vehicle/sale facts and optional evidence privately to the studio operator for review, with 90-day retention. This is separate from visitor analytics.')
        send=st.form_submit_button('Submit privately for review',type='primary',disabled=not configured())
    if send:
        if not confirmed or not consent:st.warning('Please confirm the completed sale and explicit submission permission first.')
        else:
            try:
                validate_outcome(fields,sold,when,market)
                signature=hashlib.sha256(json.dumps([safe_fields(fields),sold,str(when),market]).encode()).hexdigest()
                if st.session_state.get('sale_sent_signature')==signature:st.info('This outcome has already been submitted in this session.')
                else:
                    receipt=submit(fields,sold,when,market,estimate,model,evidence.getvalue() if evidence else None)
                    st.session_state['sale_submission_receipt']=receipt;st.session_state['sale_sent_signature']=signature
                    st.success('Saved privately for review. “Submitted” is not “verified”.')
            except ValueError as exc:st.warning(str(exc))
            except Exception:st.error('Submission could not be confirmed. No success is claimed. Try later; reused evidence cannot be submitted twice.')
    if not configured():st.info('Private feedback storage is unavailable until the deployment’s Turso settings are configured.')
    receipt=st.session_state.get('sale_submission_receipt')
    if receipt:
        st.download_button('Save private submission/deletion receipt',json.dumps(receipt,indent=2),'private-sale-receipt.json','application/json')
        st.caption('Keep this receipt privately. Its deletion key is not included in social links or cards.')
    with st.expander('Delete my submitted feedback'):
        st.caption('Use the ID and deletion key from your private receipt. This does not delete visitor analytics.')
        with st.form('delete_feedback'):
            sid=st.text_input('Submission ID',max_chars=40);key=st.text_input('Deletion key',type='password',max_chars=100)
            delete=st.form_submit_button('Delete my feedback and evidence')
        if delete:
            try:delete_own(sid,key);st.success('Deletion request processed. A matching submission and evidence have been removed. Invalid credentials do not reveal whether a submission exists.')
            except Exception:st.error('Deletion could not be confirmed. Contact the studio operator privately.')
    st.caption('Operator: '+str(_setting('ANALYTICS_CONTROLLER_NAME','Studio operator'))+'. Contact: '+str(_setting('ANALYTICS_CONTROLLER_CONTACT','Use the privacy notice')))


def render(estimate_fn=None,bid_fn=None,model_id=''):
    st.markdown(STYLE,unsafe_allow_html=True)
    st.markdown('<div class="deal-hero"><span class="deal-kicker">The group chat’s unofficial car expert</span><h2>Deal Detective</h2><p>US used cars · USD &amp; miles. Bring an advert. Keep your mechanic.</p></div>',unsafe_allow_html=True)
    # Apply OCR output before constructing the text widget on this run.
    if 'deal_ocr_pending' in st.session_state:
        st.session_state['deal_advert_text']=st.session_state.pop('deal_ocr_pending');draft_parse()
    render_shared()
    saved=st.session_state.get('last_explained_price')
    tabs=st.tabs(['📸 Import advert','🕵️ Price verdict','📤 Share & go viral','🚗 Compare cars','🔎 Evidence','🎯 Beat the model','🧰 Buyer briefing','🧾 Sale feedback'],key='detective_tabs')
    with tabs[0]:render_importer()
    with tabs[5]:render_game(bid_fn)
    if not saved:
        for tab in [tabs[1],tabs[2],tabs[3],tabs[4],tabs[6],tabs[7]]:
            with tab:st.info('Estimate a car in Price my car first, or import an advert here. Your detective needs a suspect.')
        return
    model,prediction,fields,description=saved;estimate=float(prediction['price'])
    with tabs[1]:
        st.subheader(car_title(fields))
        st.warning('US historical asking estimate · May 2021 reference. For today’s seller price, use Value over time to align dates first. No validated individual prediction interval is available.')
        seller=st.number_input('Seller asking price · USD',min_value=0.,step=100.,key='detective_seller')
        aligned=st.checkbox('This seller price refers to the same May 2021 market date',key='detective_aligned')
        a,b,c=st.columns(3);a.metric('Historical asking estimate',f'${estimate:,.0f}')
        if seller and aligned:
            gap=seller-estimate;b.metric('Advert minus estimate',f'${gap:+,.0f}');c.metric('Relative difference',f'{gap/estimate:+.1%}')
            st.write('Seller optimism: comes with ambition as standard.' if gap>estimate*.15 else 'The price has entered the conversation. The inspection still gets the final word.')
        elif seller:st.info('A present-day quote and a 2021 estimate are different dates. Comparing them directly would be time travel with a calculator.')
        st.caption('The price gap is descriptive, not proof of a good deal, negotiating room, mechanical quality or profit. Model differences are not prediction intervals.')
        st.button('Open Value over time →',on_click=move_to,args=('📅 Value over time',))
    with tabs[2]:render_share(fields,estimate,model)
    with tabs[3]:render_garage(fields,description,estimate,model_id or model,st.session_state.get('detective_seller',0),estimate_fn,bid_fn)
    with tabs[4]:render_evidence(fields)
    with tabs[6]:render_briefing(fields,description)
    with tabs[7]:render_feedback(fields,estimate,model)
