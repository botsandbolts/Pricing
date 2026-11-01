"""Searchable historical make/model presets. Manual fields remain editable."""
import json
from pathlib import Path

def load(root):
    path=Path(root)/'data/popular-vehicles.json'
    if not path.exists():return {'vehicles':[],'ranking_basis':'No presets installed.'}
    return json.loads(path.read_text())

def render(root):
    import streamlit as st
    data=load(root);rows=data['vehicles']
    if not rows:return
    choices={r['make']+'::'+r['model']:r for r in rows}
    def chosen():
        key=st.session_state.get('inspector_popular_car')
        if key in choices:
            r=choices[key]
            st.session_state['inspector_make']=r['make'].title()
            st.session_state['inspector_model']=r['model']
    st.selectbox('Find a popular make / model · 1,000 presets',list(choices),index=None,
                 placeholder='Search Ford F-150, Toyota Camry, Honda Civic…',
                 format_func=lambda k:choices[k]['make'].title()+' '+choices[k]['model'],
                 on_change=chosen,key='inspector_popular_car',persist_state='session')
    st.caption('Pick a preset to fill the make/model fields below, then edit any detail. '+data['ranking_basis'])
