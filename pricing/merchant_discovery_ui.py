"""Merchant + Meaning: direct feedback discovery, separate from profile experiments."""
import hashlib,json,sqlite3,zlib
from pathlib import Path
import numpy as np,streamlit as st
from pricing.merchant_reco import recommend,seed_cars
from pricing.svd_discovery import recommend_svd
from pricing.dealer_signals import load_dealers,dealer_caption
@st.cache_resource(max_entries=1,show_spinner=False)
def assets(folder,revision):
    p=Path(folder)
    meta=json.loads((p/'manifest.json').read_text())
    semantic=np.load(p/'semantic.npy',mmap_mode='r')
    meta['seeds']=seed_cars(semantic,count=25)
    return meta,np.load(p/'shrunk-svd.npy',mmap_mode='r')
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
    dealer_path=folder/'dealers.json'
    dealer_revision=dealer_path.stat().st_mtime_ns if dealer_path.exists() else 0
    svd_path=folder/'shrunk-svd.npy'
    if not svd_path.exists():
        st.info('The updated SVD catalogue is being prepared. Please return shortly.');return
    meta,vectors=assets(str(folder),(path.stat().st_mtime_ns,dealer_revision,svd_path.stat().st_mtime_ns))
    dealers=load_dealers(folder)
    st.caption('MiniLM + vehicle details + confidence-weighted GLiNER dealer candidates.' if dealers else 'MiniLM + vehicle details active. Dealer signals are awaiting publication.')
    version=hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    st.caption(meta.get('semantic_model','Text similarity'))
    st.caption(f"{meta['rows']:,} historical cars · no account · merchant identities are inferred, not verified")
    st.write('Rate 25 real cars. Learn your taste from description meaning AND vehicle features: price, year, mileage, body type and more. Four means love it; one means no thanks.')
    memory=st.session_state.setdefault('merchant_ratings_'+version,{})
    seeds=meta['seeds']
    labels={1:'1 · No thanks',2:'2 · Not my type',3:'3 · I like it',4:'4 · Love it'}
    for number,(i,car) in enumerate(zip(seeds,cars(folder,seeds)),start=1):
        with st.container(border=True):
            car_art(car,compact=True)
            st.markdown(f"### {number} / {len(seeds)} · {int(car['year'])} {car.get('manufacturer','')} {car.get('model','')}")
            st.write(f"Historical asking price: ${car['price']:,.0f} · {int(car.get('odometer') or 0):,} miles")
            st.caption(dealer_caption(dealers['rows'][i]) if dealers else 'Dealer candidates pending publication')
            rating=st.pills('Your reaction · car '+str(number),list(labels),format_func=labels.get,key=f'merchant_rating_{version}_{i}')
            if rating is not None:memory[i]=rating
    st.progress(sum(i in memory for i in seeds)/len(seeds),text=f'{sum(i in memory for i in seeds)} of {len(seeds)} cars rated')
    budget=None
    if st.checkbox('Limit by historical asking price',key='merchant_budget_enabled'):
        budget=st.number_input('Maximum historical asking price · USD',min_value=1000.,max_value=150000.,value=30000.,step=1000.)
    ready=all(i in memory for i in seeds) and any(v>=3 for v in memory.values())
    if st.button('Find my 25 cars',type='primary',disabled=not ready,key='merchant_find'):
        with st.spinner('Finding similar cars and balancing the shortlist…'):
            st.session_state['merchant_svd_v3_'+version]=recommend_svd(vectors,memory,meta['types'],np.array(meta['prices']),budget=budget,k=25)
    result=st.session_state.get('merchant_svd_v3_'+version)
    if result:
        st.caption(result['method'])
        st.markdown(f"## Your shortlist · {len(result['positions'])} cars · strongest preference first")
        for n,(pos,car) in enumerate(zip(result['positions'],cars(folder,result['positions']))):
            with st.container(border=True):
                car_art(car,compact=True)
                st.markdown(f"### {n+1}. {int(car['year'])} {car.get('manufacturer','')} {car.get('model','')}")
                st.write(f"Historical asking price: ${car['price']:,.0f} · {int(car.get('odometer') or 0):,} miles")
                st.caption(dealer_caption(dealers['rows'][pos]) if dealers else 'Dealer candidates pending publication')
                st.caption('Historical example, not live inventory or a current valuation.')
                with st.expander('Why this car?'):
                    st.write('Ranked by session preferences across description meaning, vehicle details and confidence-weighted dealer candidates. Scores are relative utilities, not probabilities.')
                    if dealers and dealers['rows'][pos]:
                        st.write('Original network dealer candidates (scores are not verified accuracy):')
                        st.dataframe(dealers['rows'][pos],hide_index=True)
                    st.text(car.get('description',''))
                extra=st.pills('Refine my taste · '+str(pos),list(labels),format_func=labels.get,key=f'merchant_extra_{version}_{pos}')
                if extra is not None:memory[pos]=extra
        st.caption('Rate results, then press Find my 25 cars again to refine. Previously rated cars are excluded.')
    with st.expander('How it works · business explanation'):
        st.write('Your likes and dislikes score nearby cars in a shared vehicle space. We keep MiniLM description meaning, vehicle details and confidence-weighted dealer candidates. Weak SVD components are shrunk strongly. No preference regression is fitted from your 25 ratings.')
        st.write('A similarity penalty reduces repetitive shortlists, then selected cars appear strongest preference first. Previously rated cars are excluded; at most two of an exact make/model. An interest in one classic can still suggest classics; variety is encouraged, not guaranteed.')
    with st.expander('Technical method and evidence limits'):
        st.write('64D truncated SVD fitted offline to catalogue vehicle/dealer features. Ridge-style spectral weights s²/(s² + 0.25 × largest singular value²). Numeric median/scaling uses the catalogue without ratings; dealer sparse one-hot confidence signals remain included. Normalize the shrunk block; combine equally with normalized MiniLM; normalize the combined vector.')
        st.write('Score = strongest rating-weighted liked cosine similarity minus 0.35 × strongest disliked similarity. Select 25 greedily from the top 2000 with a 0.25 cosine redundancy penalty. This is unsupervised transductive catalogue retrieval, not the former supervised cross-validation model. No validated relevance improvement is claimed.')
    with st.expander('Dealer names and evidence'):
        st.write('Original GLiNER candidates are inferred and unverified. Repeated mentions use maximum confidence; ambiguous candidates share feature weight. No manual blocklist or fuzzy name merges. Historical adverts are not live dealer inventory.')
        if dealers:st.json(dealers['report'])
