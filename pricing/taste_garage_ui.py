"""Ten honest judgments, a vehicle-only ranker, and a readable shortlist."""
import hashlib,json
from datetime import date
from pathlib import Path
import numpy as np
import streamlit as st
from pricing.taste_ranking import load,records,train_rank,ndcg10,adjusted_price_features,active_catalogue_folder

LABELS={1:'1 · 😣 No thanks',2:'2 · 🙁 Not my type',3:'3 · 🙂 I like it',4:'4 · 😍 Love it'}

@st.cache_resource(max_entries=1,show_spinner=False)
def catalogue(root,version,profile_count=10):return load(root,profile_count)

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
    from pricing.studio_design import page_intro,journey_steps
    from pricing.garage_art import car_art
    root=Path(root)
    page_intro('A little car chemistry','Pick your garage','Find your type. Four honest reactions. Absolutely no awkward goodbye.','💚')
    journey_steps([('Choose your drive','10 cars or 25? You set the pace.'),('Tell us your type','Rate the whole car, from no thanks to love.'),('Meet your shortlist','Up to ten historical cars picked for your taste.')])
    tests={'Speed Date · 10 cars, no small talk':10,'The Long Drive · 25 cars, zero commitment':25}
    st.session_state.setdefault('taste_test_mode',next(iter(tests)))
    def choose_test(value):st.session_state['taste_test_mode']=value
    for col,(value,size) in zip(st.columns(2),tests.items()):
        with col:
            with st.container(border=True):
                st.markdown('**Speed Date**' if size==10 else '**The Long Drive**')
                st.caption('10 cars. No small talk.' if size==10 else '25 cars. Zero commitment.')
                st.button('Choose Speed Date · 10 cars' if size==10 else 'Choose The Long Drive · 25 cars',key=f'taste_choose_{size}',on_click=choose_test,args=(value,),type='primary' if st.session_state['taste_test_mode']==value else 'secondary',use_container_width=True)
    chosen=st.session_state['taste_test_mode']
    profile_count=tests[chosen]
    st.caption('Switch tests without losing your ratings. Both return up to ten recommendations from historical adverts, not live stock.')
    with st.expander('About your ratings & this experiment'):
        st.write('An experimental content-based recommender using your current-session ratings and vehicle features. No user ID, demographic features or duplicated adverts. Short: ten representatives. Long: twenty-five finer profiles.')
    folder=active_catalogue_folder(root);path=folder/'manifest.json'
    if not path.exists():
        st.info('The original-data car profiles are being built offline. This panel will unlock when the tested catalogue is installed; no substitute or invented prototypes are shown.');return
    variant=folder/'profiles-25.json'
    revision=(path.stat().st_mtime_ns,path.stat().st_size,variant.stat().st_mtime_ns if variant.exists() else 0)
    try:loaded=catalogue(str(root),revision,profile_count)
    except (ValueError,OSError,KeyError):st.warning('The recommendation catalogue could not be verified. Your pricing and comparison tools remain available.');return
    if loaded is None:
        st.info('The 25-car test is being prepared from the verified HDBSCAN hierarchy. Speed Date remains available.');return
    meta,arrays=loaded;version=hashlib.sha256((path.read_bytes()+(variant.read_bytes() if profile_count==25 else b'')+str(profile_count).encode())).hexdigest()[:12]
    onehot=meta.get('encoding_kind')=='common-onehot-v1'
    st.caption(('One-hot HDBSCAN · includes mileage and vehicle year · FAISS cosine comparisons.' if onehot else 'HDBSCAN profiles · FAISS cosine preference comparisons. Real representative cars, with no dimensionality reduction.'))
    mode=st.radio('Shortlist price date',['Today','Choose a date'],horizontal=True,key='taste_date_mode')
    when=today() if mode=='Today' else st.date_input('Show price scenarios for',value=today(),key='taste_date')
    st.markdown(f'## 1 · Rate your {profile_count} representatives')
    st.write('Rate the whole car: Would you want something like this? Compare cars against your own taste, not against each other’s price alone.')
    rating_keys=[f'taste_rating_{version}_{i}' for i in range(profile_count)]
    count=sum(st.session_state.get(k) in LABELS for k in rating_keys)
    st.progress(count/profile_count,text=f'{count} of {profile_count} cars rated')
    from pricing.accessibility import announce
    announce(f'{count} of {profile_count} cars rated. '+('All cars rated.' if count==profile_count else f'{profile_count-count} reactions remaining.'))
    memory=st.session_state.setdefault('taste_rating_memory',{})
    for i,p in enumerate(meta['profiles']):
        # One pair per row preserves 1,2,3... when mobile stacks columns.
        if i%2==0:columns=st.columns(2)
        car=p['prototype']
        with columns[i%2]:
            with st.container(border=True,key=f'garage_taste_card_{profile_count}_{i}'):
                car_art(car)
                st.markdown(f"### {i+1}. {int(car['year'])} {str(car.get('manufacturer') or 'Unspecified').title()} {car.get('model') or 'model not supplied'}")
                st.caption(f"Profile {i+1} · {p['cars']:,} historical adverts · real representative")
                car_details(root,car,when)
                if rating_keys[i] not in st.session_state and rating_keys[i] in memory:st.session_state[rating_keys[i]]=memory[rating_keys[i]]
                st.markdown('**Your reaction**')
                reaction_label=f"Your reaction · car {i+1}: {int(car['year'])} {car.get('manufacturer') or 'Unspecified'} {car.get('model') or ''}"
                st.pills(reaction_label,list(LABELS),selection_mode='single',format_func=LABELS.get,key=rating_keys[i],persist_state='session',label_visibility='collapsed')
                memory[rating_keys[i]]=st.session_state.get(rating_keys[i])
                with st.expander(f'Description & details · car {i+1}'):
                    st.caption('Short excerpt from the historical advert; source text is not independently verified.')
                    st.text(car.get('description') or 'Description not supplied.')
                    st.write({k:car.get(k) for k in ['condition','drive','cylinders','title_status','paint_color','state']})
    ratings=[st.session_state.get(k) for k in rating_keys]
    complete=all(r in LABELS for r in ratings)
    signature=json.dumps([version,ratings,str(when)])
    different=len(set(ratings))>1 if complete else False
    has_like=any(r in (3,4) for r in ratings)
    if complete and not has_like:st.info('Choose I like it or Love it for at least one car so there is something to recommend.')
    if complete and not different:st.info('Give at least two cars different ratings so the model has an ordering to learn.')
    st.markdown('## 2 · Build your HDBSCAN-based shortlist')
    if not complete:st.caption(f'{profile_count-count} more reaction'+('s' if profile_count-count!=1 else '')+' to go. Your shortlist unlocks once every car has a rating.')
    if st.button('Find my ten cars →',type='primary',disabled=not complete or not different or not has_like,key='taste_train'):
        try:
            with st.spinner('Learning your ordering from vehicle features and prices, then scoring the catalogue…'):
                try:adjusted=adjusted_price_features(arrays['codes'],arrays['numeric'],meta['levels'],arrays['prices'],arrays['source_days'],when,transport(str(root)))
                except (ValueError,OSError,KeyError,TypeError):adjusted=np.full(meta['rows'],np.nan)
                result=train_rank(arrays['codes'],arrays['numeric'],meta['levels'],arrays['clusters'],ratings,[p['position'] for p in meta['profiles']],arrays['validation'],prices=arrays['prices'],adjusted_prices=adjusted,memberships=arrays.get('memberships'),medoid_distances=arrays.get('medoid_cosine',arrays.get('medoid_l2')))
                result['adjusted_price_support']=int(np.isfinite(adjusted).sum())
            st.session_state['taste_result']=(signature,result)
        except (ValueError,OSError,RuntimeError,MemoryError) as exc:
            st.warning('The ranking could not complete. Your ratings are retained; no shortlist is invented.');return
    saved=st.session_state.get('taste_result')
    if saved and saved[0]!=signature:st.info('Your reactions changed. Press Find my ten cars to refresh the shortlist.');saved=None
    if saved:
        announce('Shortlist ready. '+str(len(saved[1]['positions']))+' historical cars found.')
        result=saved[1];st.markdown(f"## Your {len(result['positions'])} · tyres optional, taste mandatory")
        st.write('**HDBSCAN finds your car profiles. Your ratings shape the shortlist.**')
        st.caption('Soft HDBSCAN memberships and exact FAISS cosine distances carry your likes and dislikes across the catalogue. Cars must pass the preference checks; a separate XGBoost ranker orders the eligible matches. These are historical adverts, not available stock. Cars you already rated are excluded.')
        cars=records(root,result['positions'])
        for rank,(position,car) in enumerate(zip(result['positions'],cars),1):
            profile=int(arrays['clusters'][position]);p=meta['profiles'][profile]
            membership=arrays.get('memberships')
            with st.container(border=True,key=f'garage_shortlist_{rank}'):
                car_art(car,compact=True)
                st.markdown(f"### {rank}. {int(car['year'])} {str(car.get('manufacturer') or 'Unspecified').title()} {car.get('model') or 'model not supplied'}")
                car_details(root,car,when)
                if result.get('evidence'):
                    e=result['evidence'][rank-1]
                    st.write(f"Preference evidence: +{e['positive']:.2f} appeal − {e['negative']:.2f} dislike = {e['signed']:+.2f}.")
                    if 'liked_profile' in e:st.caption(f"Nearest liked representative: Profile {e['liked_profile']+1} · distance {e['liked_distance']:.3f}.")
                    if 'disliked_profile' in e:st.caption(f"Nearest disliked representative: Profile {e['disliked_profile']+1} · distance {e['disliked_distance']:.3f}. This car is closer to a liked representative.")
                if membership is not None:
                    weights=membership[position];top=np.argsort(-weights)[:3]
                    st.write('Profile blend: '+ ' · '.join(f'{j+1}: {weights[j]:.0%}' for j in top if weights[j]>.001))
                    st.caption(f'Unassigned membership: {max(0,1-float(weights.sum())):.0%}. Membership strengths are model affinities, not probabilities of liking this car.')
                else:st.write(f'Related profile {profile+1}.')
                if 'medoid_cosine' in arrays or 'medoid_l2' in arrays:
                    distances=arrays.get('medoid_cosine',arrays.get('medoid_l2'))[position];near=np.argsort(distances)[:3]
                    st.caption('Nearest representatives · FAISS distance: '+ ' · '.join(f'Profile {j+1}: {distances[j]:.3f}' for j in near))
                    st.caption(('Cosine distance is 1 minus normalized inner product, from 0 to 2. It compares feature direction; zero vectors have neutral similarity.' if 'medoid_cosine' in arrays else 'This catalogue uses Euclidean distance across scaled features.')+' Lower means closer; this is not a dollar error or probability.')
                if arrays['noise'][position]:st.caption('Exploratory match: HDBSCAN treated this advert as density noise; its soft affinities remain available without forcing a hard assignment.')
                with st.expander(f'Original description & specifications · pick {rank}'):
                    st.caption('Short excerpt from the historical advert; source text is not independently verified.')
                    st.text(car.get('description') or 'Description not supplied.')
                    st.write({k:car.get(k) for k in ['condition','drive','cylinders','title_status','paint_color','state']})
        st.markdown('### Model check · what this score really measures')
        a,b,c=st.columns(3)
        a.metric('Held-out NDCG@10 · inferred preferences',f"{result['ndcg10']:.3f}" if result['ndcg10'] is not None else 'Not defined')
        b.metric('Random-order baseline',f"{result['random_ndcg10']:.3f}" if result['random_ndcg10'] is not None else 'Not defined')
        c.metric('Actual ratings supplied',profile_count)
        st.info('This NDCG@10 measures agreement with soft preferences inferred from cluster memberships. Thousands of adverts still come from only the representative ratings you supplied. It is not measured customer satisfaction or proof the recommendations are good.')
        with st.expander('Optional · rate the shortlist to check it against your real taste'):
            values=[]
            feedback_id=hashlib.sha256(signature.encode()).hexdigest()[:12]
            for i,car in enumerate(cars):values.append(st.radio(f"{i+1}. {int(car['year'])} {car.get('manufacturer')} {car.get('model')}",list(LABELS),index=None,format_func=LABELS.get,horizontal=True,key=f'taste_feedback_{feedback_id}_{i}'))
            if all(v in LABELS for v in values) and len(set(values))>1:
                direct=ndcg10(np.array(values)-1,np.arange(len(cars),0,-1))
                st.metric('NDCG@10 against your additional ratings',f'{direct:.3f}')
                st.caption('Evaluates ordering within the displayed cars only. It does not measure catalogue-wide recall. This feedback is not used to train the displayed list.')
        with st.expander('Technical result'):
            st.json({k:v for k,v in result.items() if k not in ['positions','scores']},expanded=False)
    with st.expander('Business guide · HDBSCAN profiles, FAISS distances & ranking'):
        st.write('HDBSCAN discovers soft density profiles directly in the scaled vehicle features. FAISS measures exact cosine distance to real representative cars in the original scaled feature space. XGBoost orders candidates after the like/dislike safeguards. These are separate jobs, not a guarantee of personal relevance.')
        st.write(meta.get('method_description','Scaled-feature HDBSCAN: smoothed leave-one-out category encoding and standardized numerical features, without UMAP.'))
        st.write(f"Catalogue: {meta['rows']:,} original canonical asking adverts. Native density groups: {meta['density_clusters']}. Native density noise: {meta['noise_fraction']:.1%}. No live inventory or verified condition is implied.")
        st.write('Ratings 1 and 2 subtract preference; ratings 3 and 4 add preference. Unassigned membership contributes zero. Soft density evidence gets 75% weight and medoid affinity 25%. Cars closer to a disliked representative than to a liked one are excluded, and the remaining cars need positive net evidence. Low-membership cars are not forced into a profile. XGBoost then learns an ordering from vehicle attributes and advertised prices. Original log price and a date-adjusted log-price scenario enter XGBoost, with an explicit indicator where date adjustment is unsupported. Cluster IDs, prototype IDs, descriptions and all user features are excluded from its predictors. The trained ranking and your reactions stay in this Streamlit session.')
    with st.expander('Technical guide · rank:ndcg, validation and limits'):
        st.code('XGBRanker(objective="rank:ndcg", eval_metric="ndcg@10",\n          tree_method="hist", enable_categorical=True,\n          max_depth=3, n_jobs=2)')
        st.write('One session is one ranking query group; its all-zero qid is a grouping instruction, never a predictor. Ratings 1–4 map to relevance 0–3. Signed evidence is shifted to nonnegative relevance for LambdaMART and rounded to 0.1 steps for XGBoost’s integer ranking labels. Linear NDCG gains avoid exponential amplification; displayed NDCG uses unrounded soft relevance. A deterministic vehicle-ID split holds out approximately 20% for early stopping; prototypes stay in training. A final model fits every advert once, then scores the full catalogue. Native categorical partitioning avoids one-hot features in XGBoost. Original and date-adjusted log asking prices are predictors; HDBSCAN uses target-free common-category indicators, vehicle year, log mileage and a missing-mileage indicator in the one-hot release; asking-price targets and preference ratings do not enter its geometry. No FM or Cartesian duplication is used.')
        st.write('The category vocabulary and numeric scaling use the full catalogue: this is transductive exploration, not an external benchmark. The held-out targets are still weak labels derived from the same representative judgments. Pooling rare categories can discard distinctions; clustered cars can have different appeal. Real additional ratings are needed to test personal preference quality.')
        st.write('Both tests share the fitted HDBSCAN hierarchy of the installed encoding release. The 10- and 25-profile models and medoids are rebuilt when encoding changes. New listings pass through the frozen category encoder and numeric scaler before HDBSCAN membership prediction. FAISS explanations and dislike safeguards use those original scaled features. The new catalogue uses FAISS IndexFlatIP after L2-normalizing each original feature vector: cosine distance = 1 − inner product. The same distance drives explanations and the nearest-dislike safeguard. HDBSCAN retains its Euclidean density geometry.')
        st.caption('Cluster coverage, size balance and inherited-label NDCG do not establish user satisfaction; please rate the shortlist to give independent feedback.')
        st.write(meta.get('representatives','Legacy representative selection.'))
        st.caption('Price scenarios use existing depreciation curves and the saved CPI/Holt market clock; current-car accuracy is unvalidated. Source: Craigslist Used Cars, Tom / Zenodo 10457828, CC BY 4.0.')


