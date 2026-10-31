"""Complete vehicle editor for the two-market forest studio."""
import streamlit as st
from pricing.description_examples import render_examples
from marketbridge.time_engine import today


def render(*, description_sensitive=False):
    prefix='forest_car'
    previous=st.session_state.get('_forest_editor',st.session_state.get('hierarchy_sparse_result',{}))
    for k,v in previous.get('fields',{}).items():
        if k!='auction_grade':st.session_state.setdefault(prefix+'_'+k,v)
    st.session_state.setdefault('forest_live_grade',previous.get('fields',{}).get('auction_grade'))
    st.session_state.setdefault('forest_description',previous.get('notes',''))
    controls=st.session_state.get('_forest_date_controls',{})
    st.session_state.setdefault('forest_live_date_mode',controls.get('mode','Today · follows the calendar'))
    st.session_state.setdefault('forest_live_date',controls.get('date',today()))
    st.session_state.setdefault('forest_live_analogue_today_v2',previous.get('allow',True))
    for k,v in {'year':2015,'manufacturer':'Toyota','model':'Corolla','odometer':95000,'state':'ca'}.items():st.session_state.setdefault(prefix+'_'+k,v)
    st.markdown('### Your car. Both markets. One button.')
    st.caption('Start with a real advert or enter your own vehicle. Every field is editable.')
    render_examples(True,prefix=prefix,description_key='forest_description',example_key='forest_description_example')
    mode=st.radio('Valuation date',['Today · follows the calendar','Choose a date'],horizontal=True,key='forest_live_date_mode')
    date=today() if mode.startswith('Today') else st.date_input('Value my car on',today(),key='forest_live_date')
    st.caption(f'Prices for {date:%d %b %Y}. Mileage and condition mean their values on this date.')
    with st.form('forest_vehicle_form'):
        a,b,c=st.columns(3)
        year=a.number_input('Year',1900,2021,key=prefix+'_year')
        make=b.text_input('Make',key=prefix+'_manufacturer')
        model=c.text_input('Model / trim',key=prefix+'_model')
        a,b,c=st.columns(3)
        miles=a.number_input('Mileage (miles)',0,2000000,step=1000,key=prefix+'_odometer')
        state=b.text_input('US state abbreviation',max_chars=2,key=prefix+'_state')
        body=c.selectbox('Body type',['sedan','SUV','truck','pickup','hatchback','coupe','wagon','van','convertible','other','unknown'],key=prefix+'_type')
        with st.expander('Fine-tune the details · condition, engine & more'):
            a,b,c=st.columns(3)
            condition=a.selectbox('Condition',['unknown','excellent','good','fair','like new','new','salvage'],key=prefix+'_condition')
            fuel=b.selectbox('Fuel',['gas','diesel','hybrid','electric','other','unknown'],key=prefix+'_fuel')
            transmission=c.selectbox('Transmission',['automatic','manual','other','unknown'],key=prefix+'_transmission')
            a,b=st.columns(2)
            drive=a.selectbox('Drive',['unknown','fwd','rwd','4wd'],key=prefix+'_drive')
            title=b.selectbox('Title status',['clean','rebuilt','salvage','lien','missing','parts only','unknown'],key=prefix+'_title_status')
            cylinders=a.selectbox('Cylinders',['unknown','3 cylinders','4 cylinders','5 cylinders','6 cylinders','8 cylinders','10 cylinders','12 cylinders','other'],key=prefix+'_cylinders')
            color=b.selectbox('Paint color',['unknown','white','black','silver','grey','blue','red','green','brown','yellow','orange','purple','custom'],key=prefix+'_paint_color')
            grade=st.number_input('Auction condition grade · optional (1–5)',min_value=1.,max_value=5.,value=None,step=.1,key='forest_live_grade')
        notes=st.text_area('Seller description — free text',key='forest_description',height=150,max_chars=8000,placeholder='Service history, recent repairs, mysterious rattles…')
        st.caption('MiniLM reads the description; the arithmetic mean of two description-only dollar prices enters XGBoost as one feature. No TF-IDF or raw embeddings enter the trees.' if description_sensitive=='minilm-price' else 'The asking model reads this description through 384 frozen MiniLM embedding dimensions. No TF-IDF or UMAP is used; auction pricing uses structured facts.' if description_sensitive=='minilm-embeddings' else 'This hierarchical asking model reads the description through sparse TF-IDF only; no transformer is used.' if description_sensitive=='tfidf' else 'The asking model reads this description through sparse TF-IDF and a compact MiniLM price signal (first 128 tokens). Auction pricing uses the structured vehicle details.' if description_sensitive else 'Descriptions help you load and edit the car details. This forest uses structured fields only; wording does not change its estimate. Use Price my car for the MiniLM text-sensitive calculator.')
        allow=st.checkbox('Allow an auction-generation analogue for 2016–2021 cars',value=True,key='forest_live_analogue_today_v2',help='Auction training covers model years through 2015. This explicit assumption cannot recover an unseen generation.')
        submitted=st.form_submit_button('Estimate asking + auction prices',type='primary',width='stretch')
    fields=dict(year=year,manufacturer=make,model=model,odometer=miles,state=state,type=body,condition=condition,fuel=fuel,transmission=transmission,drive=drive,title_status=title,cylinders=cylinders,paint_color=color,auction_grade=grade)
    st.session_state['_forest_editor']={'fields':fields,'notes':notes,'allow':allow}
    st.session_state['_forest_date_controls']={'mode':mode,'date':date}
    return fields,notes,date,allow,submitted
