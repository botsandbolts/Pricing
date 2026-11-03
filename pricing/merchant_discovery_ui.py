"""Merchant + Meaning: direct feedback discovery, separate from profile experiments."""
import hashlib,json,sqlite3,zlib
from pathlib import Path
import numpy as np,streamlit as st
from pricing.merchant_reco import recommend,seed_cars
@st.cache_resource(max_entries=1,show_spinner=False)
def assets(folder,revision):
    p=Path(folder)
    meta=json.loads((p/'manifest.json').read_text())
    semantic=np.load(p/'semantic.npy',mmap_mode='r')
    meta['seeds']=seed_cars(semantic,count=25)
    return meta,semantic,np.load(p/'inventory.npy',mmap_mode='r')
def cars(folder,positions):
    with sqlite3.connect(str(folder/'cars.sqlite')) as conn:
        return [json.loads(zlib.decompress(conn.execute('SELECT payload FROM cars WHERE position=?',(int(i),)).fetchone()[0])) for i in positions]
def render(root):
    from pricing.studio_design import page_intro
    from pricing.garage_art import car_art
    folder=Path(root)/'data/merchant-discovery-minilm'
    path=folder/'manifest.json'
    page_intro('Drive Chemistry · Merchant + Meaning','Find your type, not a cluster','Twenty-five honest reactions. Cars that speak your language.','💚')
    if not path.exists():st.info('The merchant and semantic catalogue is being built offline. Existing profile models remain available.');return
    meta,semantic,inventory=assets(str(folder),path.stat().st_mtime_ns)
    st.caption('MiniLM description matching is active. Merchant-name extraction is being processed separately; merchant inventory links are not active in this release.')
    version=hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    st.caption(meta.get('semantic_model','Text similarity'))
    st.caption(f"{meta['rows']:,} historical cars · no account · merchant identities are inferred, not verified")
    st.write('Rate these 25 actual cars, then find your ten-car shortlist. Likes pull related cars closer; dislikes reduce their scores. The model keeps several interests separate, so liking a pickup and a convertible is allowed.')
    memory=st.session_state.setdefault('merchant_ratings_'+version,{})
    seeds=meta['seeds']
    labels={1:'1 · No thanks',2:'2 · Not my type',3:'3 · I like it',4:'4 · Love it'}
    for number,(i,car) in enumerate(zip(seeds,cars(folder,seeds)),start=1):
        with st.container(border=True):
            car_art(car,compact=True)
            st.markdown(f"### {number} / {len(seeds)} · {int(car['year'])} {car.get('manufacturer','')} {car.get('model','')}")
            st.write(f"Historical asking price: ${car['price']:,.0f} · {int(car.get('odometer') or 0):,} miles")
            rating=st.pills('Your reaction · car '+str(number),list(labels),format_func=labels.get,key=f'merchant_rating_{version}_{i}')
            if rating is not None:memory[i]=rating
    st.progress(sum(i in memory for i in seeds)/len(seeds),text=f'{sum(i in memory for i in seeds)} of {len(seeds)} cars rated')
    budget=None
    if st.checkbox('Limit by historical asking price',key='merchant_budget_enabled'):
        budget=st.number_input('Maximum historical asking price · USD',min_value=1000.,max_value=150000.,value=30000.,step=1000.)
    ready=all(i in memory for i in seeds) and any(v>=3 for v in memory.values())
    if st.button('Find cars for my taste',type='primary',disabled=not ready,key='merchant_find'):
        with st.spinner('Matching vehicle descriptions…'):
            st.session_state['merchant_result_25_'+version]=recommend(semantic,inventory,memory,meta['types'],np.array(meta['prices']),budget=budget)
    result=st.session_state.get('merchant_result_25_'+version)
    if result:
        st.markdown('## Your shortlist · strongest preference first')
        for n,(pos,car) in enumerate(zip(result['positions'],cars(folder,result['positions']))):
            with st.container(border=True):
                car_art(car,compact=True)
                st.markdown(f"### {n+1}. {int(car['year'])} {car.get('manufacturer','')} {car.get('model','')}")
                st.write(f"Historical asking price: ${car['price']:,.0f} · {int(car.get('odometer') or 0):,} miles")
                st.caption('Historical example, not live inventory or a current valuation.')
                with st.expander('Why this car?'):
                    st.write({'semantic_match':round(result['semantic'][n],3),'inventory_association':round(result['inventory'][n],3),'dislike_penalty':round(result['dislike_penalty'][n],3)})
                    st.caption('These are ranking signals, not probabilities. Merchant associations describe shared vehicle types, not verified buyer preferences.')
                    st.text(car.get('description',''))
                extra=st.pills('Refine my taste · '+str(pos),list(labels),format_func=labels.get,key=f'merchant_extra_{version}_{pos}')
                if extra is not None:memory[pos]=extra
        st.caption('Rate results, then press Find cars for my taste again to refine. Previously rated cars are excluded.')
    with st.expander('How it works · business explanation'):
        st.write('MiniLM description meaning supplies the current recommendations. Twenty-five diverse real catalogue cars collect your reactions. Merchant-name extraction is a separate offline job and is not active in this release.')
        st.write('Each liked car contributes an individual similarity signal. Dislikes apply a soft penalty, not an all-or-nothing veto. Results are sorted by descending hybrid preference score, with at most two cars of an exact make/model type.')
        st.write(f"Merchant candidate groups: {meta['merchant_candidates']:,}. Linked adverts: {meta['linked_rows']:,}. These identities have not been independently verified.")
    with st.expander('Technical method and evidence limits'):
        st.write('Full-catalogue vehicle descriptions are encoded offline with all-MiniLM-L6-v2, projected to 96 dimensions with truncated SVD. Stored document vectors are normalized; no text model is loaded in the app. Inventory counts are normalized per merchant and weighted by inverse merchant frequency, then mapped through a low-rank SVD baseline. Ranking weights are 80% semantic and 20% inventory where available; dislike similarity subtracts up to 35%. No permanent user profile or learned satisfaction probability is asserted.')
        st.write('Five single-like/multiple-dislike mechanical tests must produce distinct results before publication. There is no independent relevance benchmark or audited merchant-identity precision yet. Additional session ratings are used for refinement, not reported as an unbiased test.')
