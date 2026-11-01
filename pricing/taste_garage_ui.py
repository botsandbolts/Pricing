"""Ten honest judgments, a vehicle-only ranker, and a readable shortlist."""
import hashlib,json
from datetime import date
from pathlib import Path
import numpy as np
import streamlit as st
from pricing.taste_ranking import load,records,train_rank,ndcg10,adjusted_price_features

LABELS={1:'1 · Very bad',2:'2 · Bad',3:'3 · Good',4:'4 · Very good'}

@st.cache_resource(max_entries=1,show_spinner=False)
def catalogue(root,version):return load(root)

@st.cache_resource(max_entries=1,show_spinner=False)
def transport(root):
    from marketbridge.time_engine import TimeTransport
    return TimeTransport(root)

def price_view(root,car,when):
    from pricing.reference_prices import adjust_reference
    try:return adjust_reference(car,'ask',when,transport(str(root)))['price_usd']
    except (ValueError,OSError,KeyError,TypeError):return None

def car_details(root,car,when):
    mileage=f"{int(car['odometer']):,} miles" if car.get('odometer') is not None else 'mileage not supplied'
    st.write(' · '.join([mileage]+[str(car[k]).title() for k in ['type','transmission','fuel'] if car.get(k)]))
    value=price_view(root,car,when)
    if value is not None:st.markdown(f'**${value:,.0f}** · asking-price scenario for {when:%d %b %Y}')
    else:st.caption('Date-adjusted price unavailable for this historical car.')
    st.caption(f"Original asking advert: ${car['price']:,.0f} · {str(car['observed_date'])[:10]}. Not a live offer.")

def render(root):
    from marketbridge.time_engine import today
    root=Path(root);st.title('🚦 Pick your garage')
    st.write('Ten cars. Four honest reactions. Your inner car nerd gets the steering wheel.')
    st.caption('An experimental content-based recommender using your current-session ratings and vehicle features. No user ID, demographic features or duplicated adverts.')
    path=root/'data/taste-garage/manifest.json'
    if not path.exists():
        st.info('The original-data car profiles are being built offline. This panel will unlock when the tested catalogue is installed; no substitute or invented prototypes are shown.');return
    try:loaded=catalogue(str(root),(path.stat().st_mtime_ns,path.stat().st_size))
    except (ValueError,OSError,KeyError):st.warning('The recommendation catalogue could not be verified. Your pricing and comparison tools remain available.');return
    if loaded is None:return
    meta,arrays=loaded;version=hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    mode=st.radio('Shortlist price date',['Today','Choose a date'],horizontal=True,key='taste_date_mode')
    when=today() if mode=='Today' else st.date_input('Show price scenarios for',value=today(),key='taste_date')
    st.markdown('### 1 · Rate the ten representatives')
    st.write('Rate the whole car: Would you want something like this? Compare cars against your own taste, not against each other’s price alone.')
    rating_keys=[f'taste_rating_{version}_{i}' for i in range(10)]
    count=sum(st.session_state.get(k) in LABELS for k in rating_keys)
    st.progress(count/10,text=f'{count} of 10 cars rated')
    columns=st.columns(2)
    for i,p in enumerate(meta['profiles']):
        car=p['prototype']
        with columns[i%2]:
            with st.container(border=True):
                st.markdown(f"#### {i+1}. {int(car['year'])} {str(car.get('manufacturer') or 'Unspecified').title()} {car.get('model') or 'model not supplied'}")
                st.caption(f"{p['name']} · represents {p['cars']:,} historical adverts")
                car_details(root,car,when)
                st.radio('Your reaction',list(LABELS),index=None,format_func=LABELS.get,horizontal=True,key=rating_keys[i])
                with st.expander(f'Description & details · car {i+1}'):
                    st.caption('Short excerpt from the historical advert; source text is not independently verified.')
                    st.text(car.get('description') or 'Description not supplied.')
                    st.write({k:car.get(k) for k in ['condition','drive','cylinders','title_status','paint_color','state']})
    ratings=[st.session_state.get(k) for k in rating_keys]
    complete=all(r in LABELS for r in ratings)
    signature=json.dumps([version,ratings,str(when)])
    different=len(set(ratings))>1 if complete else False
    if complete and not different:st.info('Give at least two cars different ratings so the model has an ordering to learn.')
    st.markdown('### 2 · Let the trees build your shortlist')
    if st.button('Find my ten cars →',type='primary',disabled=not complete or not different,key='taste_train'):
        try:
            with st.spinner('Learning your ordering from vehicle features and prices, then scoring the catalogue…'):
                try:adjusted=adjusted_price_features(arrays['codes'],arrays['numeric'],meta['levels'],arrays['prices'],arrays['source_days'],when,transport(str(root)))
                except (ValueError,OSError,KeyError,TypeError):adjusted=np.full(meta['rows'],np.nan)
                result=train_rank(arrays['codes'],arrays['numeric'],meta['levels'],arrays['clusters'],ratings,[p['position'] for p in meta['profiles']],arrays['validation'],prices=arrays['prices'],adjusted_prices=adjusted)
                result['adjusted_price_support']=int(np.isfinite(adjusted).sum())
            st.session_state['taste_result']=(signature,result)
        except (ValueError,OSError,RuntimeError,MemoryError) as exc:
            st.warning('The ranking could not complete. Your ratings are retained; no shortlist is invented.');return
    saved=st.session_state.get('taste_result')
    if saved and saved[0]!=signature:st.info('Your reactions changed. Press Find my ten cars to refresh the shortlist.');saved=None
    if saved:
        result=saved[1];st.markdown('### Your ten · tyres optional, taste mandatory')
        st.caption('Ordered by XGBoost preference score. These are historical adverts, not available stock. The ten prototypes you already rated are excluded.')
        cars=records(root,result['positions'])
        for rank,(position,car) in enumerate(zip(result['positions'],cars),1):
            profile=int(arrays['clusters'][position]);p=meta['profiles'][profile]
            with st.container(border=True):
                st.markdown(f"#### {rank}. {int(car['year'])} {str(car.get('manufacturer') or 'Unspecified').title()} {car.get('model') or 'model not supplied'}")
                car_details(root,car,when)
                st.write(f"Related profile: {p['name']} · you rated its representative {LABELS[ratings[profile]]}.")
                if arrays['noise'][position]:st.caption('Exploratory match: HDBSCAN treated this advert as density noise; its profile was assigned by proximity.')
                with st.expander(f'Original description & specifications · pick {rank}'):
                    st.caption('Short excerpt from the historical advert; source text is not independently verified.')
                    st.text(car.get('description') or 'Description not supplied.')
                    st.write({k:car.get(k) for k in ['condition','drive','cylinders','title_status','paint_color','state']})
        st.markdown('### Model check · what this score really measures')
        a,b,c=st.columns(3)
        a.metric('Held-out NDCG@10 · inherited labels',f"{result['ndcg10']:.3f}" if result['ndcg10'] is not None else 'Not defined')
        b.metric('Random-order baseline',f"{result['random_ndcg10']:.3f}" if result['random_ndcg10'] is not None else 'Not defined')
        c.metric('Actual ratings supplied',10)
        st.info('This NDCG@10 measures agreement with ratings inherited from clusters. Thousands of adverts still come from only ten actual judgments. It is not measured customer satisfaction or proof the recommendations are good.')
        with st.expander('Optional · rate the shortlist to check it against your real taste'):
            values=[]
            feedback_id=hashlib.sha256(signature.encode()).hexdigest()[:12]
            for i,car in enumerate(cars):values.append(st.radio(f"{i+1}. {int(car['year'])} {car.get('manufacturer')} {car.get('model')}",list(LABELS),index=None,format_func=LABELS.get,horizontal=True,key=f'taste_feedback_{feedback_id}_{i}'))
            if all(v in LABELS for v in values) and len(set(values))>1:
                direct=ndcg10(np.array(values)-1,np.arange(10,0,-1))
                st.metric('NDCG@10 against your additional ratings',f'{direct:.3f}')
                st.caption('Evaluates ordering within these ten displayed cars only. It does not measure catalogue-wide recall. This feedback is not used to train the displayed list.')
        with st.expander('Technical result'):
            st.json({k:v for k,v in result.items() if k not in ['positions','scores']},expanded=False)
    with st.expander('Business guide · how the garage learns'):
        st.write('UMAP maps structured vehicle attributes into five dimensions. HDBSCAN finds dense groups. If it yields more than ten groups, nearby group centroids are merged into ten macroprofiles. Each representative is an actual advert nearest its profile centroid. Density noise gets a nearest-profile assignment and remains flagged.')
        st.write(f"Catalogue: {meta['rows']:,} original canonical asking adverts. Native density groups: {meta['density_clusters']}. Density noise assigned by proximity: {meta['noise_fraction']:.1%}. No live inventory or verified condition is implied.")
        st.write('Your rating propagates to adverts in that profile. XGBoost then learns an ordering from vehicle attributes and advertised prices. Original log price and a date-adjusted log-price scenario enter XGBoost, with an explicit indicator where date adjustment is unsupported. Cluster IDs, prototype IDs, descriptions and all user features are excluded from its predictors. The trained ranking and your reactions stay in this Streamlit session.')
    with st.expander('Technical guide · rank:ndcg, validation and limits'):
        st.code('XGBRanker(objective="rank:ndcg", eval_metric="ndcg@10",\n          tree_method="hist", enable_categorical=True,\n          max_depth=3, n_jobs=2)')
        st.write('One session is one ranking query group; its all-zero qid is a grouping instruction, never a predictor. Ratings 1–4 map to relevance 0–3. A deterministic vehicle-ID split holds out approximately 20% for early stopping; prototypes stay in training. A final model fits every advert once, then scores the full catalogue. Native categorical partitioning avoids one-hot features in XGBoost. Original and date-adjusted log asking prices are predictors; price is excluded from UMAP/HDBSCAN. No FM or Cartesian duplication is used.')
        st.write('The vocabulary and unsupervised geometry use the full catalogue: this is transductive exploration, not an external benchmark. The held-out targets are still weak labels derived from the same ten judgments. UMAP distances can distort structure; clustered cars can have different appeal. Real additional ratings are needed to test personal preference quality.')
        st.caption('Price scenarios use existing depreciation curves and the saved CPI/Holt market clock; current-car accuracy is unvalidated. Source: Craigslist Used Cars, Tom / Zenodo 10457828, CC BY 4.0.')
