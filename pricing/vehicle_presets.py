"""Specific vehicle examples that fill the whole editable Inspector form."""
import json,hashlib
from pathlib import Path

def load(root):
    file=Path(root)/'data/vehicle-examples.json'
    if file.exists():
        data=json.loads(file.read_text())
        data['vehicles']=sorted(data['vehicles'],key=lambda r:hashlib.sha256(('specific-vehicles-v1:'+str(r['id'])).encode()).hexdigest())
        data['selection']='1,000 specific historical cars: mixed asking adverts and auction records, shuffled. Search year, car, mileage or state; selecting one loads its available details and description. Edit mileage and condition for your valuation date.'
        return data
    from scripts.build_vehicle_presets import examples
    return examples(root)

def render(root):
    import streamlit as st
    @st.cache_data(max_entries=2,show_spinner=False)
    def read(folder,version):return load(folder)
    path=Path(root)/'data/vehicle-examples.json';data=read(str(root),path.stat().st_mtime_ns if path.exists() else 0)
    choices={r['id']:r for r in data['vehicles']}
    def label(key):
        row=choices[key];miles=f"{int(row['odometer']):,} mi" if row.get('odometer') is not None else 'mileage not supplied'
        return f"{row['year']} {str(row['manufacturer']).title()} {row['model']} · {miles} · {str(row.get('state') or '').upper()} · {key[:8] if len(key)>20 else key}"
    def chosen():
        row=choices.get(st.session_state.get('inspector_example_car'))
        if row is None:return
        mapping={'manufacturer':'make','model':'model','year':'year','odometer':'mileage','state':'state','type':'body','condition':'condition','fuel':'fuel','drive':'drive','transmission':'transmission','title_status':'title','cylinders':'cylinders','paint_color':'color'}
        options={'body':['unknown','sedan','SUV','truck','pickup','hatchback','coupe','wagon','van','convertible','other'],
                 'condition':['unknown','excellent','good','fair','like new','new','salvage'],'fuel':['unknown','gas','diesel','hybrid','electric','other'],
                 'drive':['unknown','fwd','rwd','4wd'],'transmission':['unknown','automatic','manual'], 'title':['unknown','clean','salvage','rebuilt','lien'],
                 'cylinders':['unknown','3 cylinders','4 cylinders','5 cylinders','6 cylinders','8 cylinders','10 cylinders','12 cylinders','other'],
                 'color':['unknown','white','black','silver','grey','blue','red','green','brown','yellow','orange','purple','custom']}
        for field,key in mapping.items():
            value=row.get(field)
            if key=='make':value=str(value or '').title()
            elif key=='year':value=int(value)
            elif key=='mileage':value=int(value) if value is not None else None
            elif key in options:
                lookup={s.lower():s for s in options[key]};value=lookup.get(str(value or '').lower(),'unknown')
            else:value=str(value or '')
            st.session_state['inspector_'+key]=value
        st.session_state['inspector_description']=str(row.get('description') or '')[:8000]
        st.session_state['inspector_grade']=None
        st.session_state['inspector_source_id']=row['id']
        st.session_state.pop('inspector_result',None);st.session_state.pop('inspector_passport',None);st.session_state.pop('inspector_own_price',None)
    if st.session_state.get('inspector_example_car') is not None and st.session_state.get('inspector_example_car') not in choices:
        st.session_state['inspector_example_car']=None
    st.selectbox('Choose a specific car · 1,000 examples',list(choices),index=None,placeholder='Search year, car, mileage or state…',format_func=label,on_change=chosen,key='inspector_example_car',persist_state='session')
    st.caption(data['selection'])
    selected=choices.get(st.session_state.get('inspector_example_car'))
    if selected:st.caption('Selected record: '+str(selected.get('reference_period') or selected.get('sale_date') or 'Historical')+' · '+selected.get('price_kind','Historical vehicle')+'. Fields remain editable.')
