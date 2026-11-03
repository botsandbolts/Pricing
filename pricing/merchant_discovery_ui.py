"""Merchant + Meaning: direct feedback discovery, separate from profile experiments."""
import hashlib,json,sqlite3,zlib
from pathlib import Path
import numpy as np,streamlit as st
from pricing.merchant_reco import recommend,seed_cars
from pricing.session_pairwise import catalogue_features,recommend_pairwise
from pricing.dealer_signals import load_dealers,dealer_caption
@st.cache_resource(max_entries=1,show_spinner=False)
def assets(folder,revision):
    p=Path(folder)
    meta=json.loads((p/'manifest.json').read_text())
    semantic=np.load(p/'semantic.npy',mmap_mode='r')
    meta['seeds']=seed_cars(semantic,count=25)
    return meta,semantic,np.load(p/'inventory.npy',mmap_mode='r'),catalogue_features(p)
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
    meta,semantic,inventory,(numeric,categories)=assets(str(folder),(path.stat().st_mtime_ns,dealer_revision))
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
        with st.spinner('Learning your preferences and checking unseen rated cars…'):
            st.session_state['merchant_pairwise_dealers_v2_'+version]=recommend_pairwise(semantic,numeric,categories,memory,meta['types'],np.array(meta['prices']),budget=budget,k=25) if len(set(memory.values()))>1 else {**recommend(semantic,inventory,memory,meta['types'],np.array(meta['prices']),budget=budget,k=25),'method':'Description similarity fallback: all ratings are equal'}
    result=st.session_state.get('merchant_pairwise_dealers_v2_'+version)
    if result:
        st.caption(result['method'])
        evidence=result.get('validation',{})
        if evidence.get('accuracy') is not None:
            st.metric('Held-out preference agreement',f"{evidence['accuracy']:.0%}")
            st.caption(f"Five-fold whole-car validation · {evidence['pairs']} unseen unequal-rating pairs. Small session diagnostic; 50% is chance agreement.")
        else:st.caption('Not enough unequal held-out ratings for a validation score.')
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
        st.write('Your ratings create comparisons: four should rank above two. Strongly regularized learning combines description meaning with historical price, year, mileage and vehicle details. Four means love it; one means no thanks. Ratings stay in this session. Original GLiNER dealer candidates now supply an additional soft identity signal when installed. Dealer identity does not guarantee a car suits you.')
        st.write('Highest preference first, previously rated cars excluded, at most two of an exact make/model. Budget refers to historical advertised prices, not current valuations.')
    with st.expander('Technical method and evidence limits'):
        st.write('L2 pairwise logistic regression, fixed C=1, no intercept, gap-weighted mirrored unequal-rating pairs with fixed total weight. Inputs: 96D MiniLM, year, log mileage/price, missingness indicators and 512 hashed categorical features (collisions possible). Numeric medians and scaling use training cars only, clipped at five standard deviations.')
        st.write('Five-fold shuffled whole-car validation: training pairs use training cars only; validation pairs use held-out cars only. No hyperparameter selection on these results. Ties earn half credit. Final fitting uses all session ratings. This small prototype-panel diagnostic is not an independent relevance benchmark. Equal ratings use a labelled semantic fallback.')

    with st.expander('Dealer signals · business and technical details'):
        st.write('Dealers often specialize in particular cars. Your reactions can teach the ranker an association with a named dealer alongside vehicle features. Names are original network guesses, not verified sellers; mistaken names can affect results. Unidentified adverts stay eligible.')
        st.write('Exact case/whitespace-normalized sparse one-hot identities; maximum score across repeated mentions; final feature value 0.5 × NER score / sqrt(number of candidates). No blacklist, fuzzy resolution, website requirement, hard dealer filter or dealer-only ordering. All weights are learned with the same L2 penalty and whole-car cross-validation. Dealer scores are not calibrated probabilities. This learns dealer associations from session ratings, not a verified merchant inventory graph.')
        if dealers:st.json(dealers['report'])

