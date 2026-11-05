"""Ten honest judgments, a vehicle-only ranker, and a readable shortlist."""
import hashlib,json,logging
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
    if (root/'data/merchant-discovery-minilm/manifest.json').exists() and json.loads((root/'data/merchant-discovery-minilm/manifest.json').read_text()).get('ui_check'):
        engine=st.radio('Discovery approach',['Merchant + Meaning','Existing profile experiments'],horizontal=True,key='drive_discovery_engine')
        if engine=='Merchant + Meaning':
            from pricing.merchant_discovery_ui import render as render_merchant_discovery
            render_merchant_discovery(root);return
    page_intro('Your personal car discovery studio','Drive Chemistry','Rate a few cars. Find your type. Skip the awkward small talk.','💚')
    st.caption('Free to explore · No registration · Real historical cars, not live inventory')
    journey_steps([('Pick your pace','A quick taste test or a deeper dive.'),('Trust your gut','Four reactions. No car expertise required.'),('Explore your ten','See the cars, the prices and why they made it.')])
    st.session_state.setdefault('taste_model_choice','HDBSCAN')
    def choose_model(name):
        st.session_state['taste_model_choice']=name
        st.session_state['taste_kmeans_enabled']=name=='KMeans'
        st.session_state['taste_hdbscan_enabled']=name=='HDBSCAN'
    model_columns=st.columns(2)
    with model_columns[0]:
        st.checkbox('KMeans · quick taste groups',value=st.session_state['taste_model_choice']=='KMeans',key='taste_kmeans_enabled',on_change=choose_model,args=('KMeans',))
    with model_columns[1]:
        st.checkbox('HDBSCAN · density taste groups',value=st.session_state['taste_model_choice']=='HDBSCAN',key='taste_hdbscan_enabled',on_change=choose_model,args=('HDBSCAN',))
    st.caption('Choose one model at a time. Both models are retained; each keeps its own ratings.')
    if st.session_state['taste_model_choice']=='KMeans' and not (root/'data/taste-garage-kmeans/release-check.json').exists():
        st.info('KMeans is still being prepared. HDBSCAN remains available through its checkbox.');return
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
    with st.expander('How Drive Chemistry picks your cars'):
        st.markdown('1. **Meet a few real cars.** Each represents a group with shared vehicle characteristics.\n2. **React honestly.** 1 means no thanks; 4 means love it. Rate the whole car, including the price.\n3. **Explore your shortlist.** Your likes and dislikes filter the catalogue; a ranking model selects up to ten cars.\n4. **Check the fit.** Open “Why this car?” or rate the results to tell us how well we did.')
        st.caption('Speed Date uses 10 representative cars. The Long Drive uses 25 finer profiles. Neither promises a perfect match. Your reactions stay in this Streamlit session.')
    folder=active_catalogue_folder(root);path=folder/'manifest.json'
    if not path.exists():
        st.info('The original-data car profiles are being built offline. This panel will unlock when the tested catalogue is installed; no substitute or invented prototypes are shown.');return
    variant=folder/'profiles-25.json'
    revision=(str(folder),path.stat().st_mtime_ns,path.stat().st_size,variant.stat().st_mtime_ns if variant.exists() else 0)
    try:loaded=catalogue(str(root),revision,profile_count)
    except (ValueError,OSError,KeyError):st.warning('The recommendation catalogue could not be verified. Your pricing and comparison tools remain available.');return
    if loaded is None:
        st.info('The 25-car test is being prepared from the verified HDBSCAN hierarchy. Speed Date remains available.');return
    meta,arrays=loaded;version=hashlib.sha256((path.read_bytes()+(variant.read_bytes() if profile_count==25 else b'')+str(profile_count).encode())).hexdigest()[:12]
    mixed=meta.get('encoding_kind')=='lemon-textprice-onehot-umap-v1'
    onehot=mixed or meta.get('encoding_kind')=='common-onehot-v1'
    st.caption(f"Your discovery catalogue · {meta['rows']:,} historical cars · {profile_count} cars to react to")
    if meta.get('density_method')=='minibatch-kmeans-soft':st.caption('Quick Match · temporary soft MiniBatchKMeans profiles. Distance affinities, not density memberships.')
    mode=st.radio('Shortlist price date',['Today','Choose a date'],horizontal=True,key='taste_date_mode')
    when=today() if mode=='Today' else st.date_input('Show price scenarios for',value=today(),key='taste_date')
    st.markdown(f'## 1 · Meet your {profile_count} taste-test cars')
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
                st.caption(f"Taste profile {i+1} · represents {p['cars']:,} historical cars")
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
    signature=json.dumps(['direct-ratings-cosine-descending-v8',st.session_state['taste_model_choice'],version,ratings,str(when)])
    different=len(set(ratings))>1 if complete else False
    has_like=any(r in (3,4) for r in ratings)
    if complete and not has_like:st.info('Choose I like it or Love it for at least one car so there is something to recommend.')
    if complete and not different:st.info('Give at least two cars different ratings so the model has an ordering to learn.')
    st.markdown('## 2 · Find your car chemistry')
    if not complete:st.caption(f'{profile_count-count} more reaction'+('s' if profile_count-count!=1 else '')+' to go. Your shortlist unlocks once every car has a rating.')
    if st.button('Find my ten cars →',type='primary',disabled=not complete or not different or not has_like,key='taste_train'):
        import time as _telemetry_time
        from pricing.feature_telemetry import record_server_run, record_server_error
        started=_telemetry_time.monotonic()
        try:
            with st.spinner('Mixing your likes, dislikes and a little car chemistry…'):
                try:adjusted=adjusted_price_features(arrays['codes'],arrays['numeric'],meta['levels'],arrays['prices'],arrays['source_days'],when,transport(str(root)))
                except (ValueError,OSError,KeyError,TypeError):adjusted=np.full(meta['rows'],np.nan)
                result=train_rank(arrays['codes'],arrays['numeric'],meta['levels'],arrays['clusters'],ratings,[p['position'] for p in meta['profiles']],arrays['validation'],prices=arrays['prices'],adjusted_prices=adjusted,memberships=arrays.get('memberships'),medoid_distances=arrays.get('medoid_rmse',arrays.get('medoid_cosine',arrays.get('medoid_l2'))))
                result['adjusted_price_support']=int(np.isfinite(adjusted).sum())
            st.session_state['taste_result']=(signature,result)
            record_server_run('taste_garage',st.session_state['taste_model_choice'].lower(),
                              'success',profile_count,len(result['positions']),
                              round(1000*(_telemetry_time.monotonic()-started)),
                              event_key=hashlib.sha256(signature.encode()).hexdigest())
        except ValueError as exc:
            record_server_run('taste_garage',st.session_state['taste_model_choice'].lower(),
                              'unavailable',profile_count,0,
                              round(1000*(_telemetry_time.monotonic()-started)))
            record_server_error('taste_garage','ranking_unavailable')
            logging.getLogger(__name__).exception('Drive Chemistry invalid ranking state')
            if str(exc).startswith('No cars pass'):
                st.info('No cars pass this combination of likes and dislikes. Your ratings are saved. Try liking another taste-test car or use The Long Drive for finer choices.')
            else:
                st.error('The shortlist could not be calculated. Your ratings are saved.')
                with st.expander('Error details for support'):st.text(str(exc))
            return
        except (OSError,RuntimeError,MemoryError) as exc:
            record_server_run('taste_garage',st.session_state['taste_model_choice'].lower(),
                              'error',profile_count,0,
                              round(1000*(_telemetry_time.monotonic()-started)))
            record_server_error('taste_garage','ranking_execution_failed')
            logging.getLogger(__name__).exception('Drive Chemistry ranking execution failed')
            st.error('The shortlist ran into a processing problem. Your ratings are saved; please retry.')
            with st.expander('Error details for support'):st.text(type(exc).__name__)
            return
    saved=st.session_state.get('taste_result')
    if saved and saved[0]!=signature:st.info('Your reactions changed. Press Find my ten cars to refresh the shortlist.');saved=None
    if saved:
        announce('Shortlist ready. '+str(len(saved[1]['positions']))+' historical cars found.')
        result=saved[1];st.markdown(f"## Your {len(result['positions'])} · tyres optional, taste mandatory")
        st.write('**Discovery order · more different picks first.** These ten passed your preference checks, then were sorted from most different to closest to a car you liked. This is exploration order, not a best-match ranking.')
        st.caption('Your ratings shaped this shortlist. Cars you rated are left out, so every pick is a new introduction. These are historical examples, not cars currently for sale.')
        cars=records(root,result['positions'])
        for rank,(position,car) in enumerate(zip(result['positions'],cars),1):
            profile=int(arrays['clusters'][position]);p=meta['profiles'][profile]
            membership=arrays.get('memberships')
            with st.container(border=True,key=f'garage_shortlist_{rank}'):
                car_art(car,compact=True)
                st.markdown(f"### {rank}. {int(car['year'])} {str(car.get('manufacturer') or 'Unspecified').title()} {car.get('model') or 'model not supplied'}")
                car_details(root,car,when)
                with st.expander(f'Why this car? · pick {rank}'):
                    if result.get('evidence'):
                        e=result['evidence'][rank-1]
                        st.write(f"Your taste signals: +{e['positive']:.2f} from likes − {e['negative']:.2f} from dislikes = {e['signed']:+.2f}.")
                        if 'liked_profile' in e:st.caption(f"Nearest liked representative: Profile {e['liked_profile']+1} · cosine distance {e['liked_distance']:.3f}.")
                        if 'disliked_profile' in e:st.caption(f"Nearest disliked representative: Profile {e['disliked_profile']+1} · distance {e['disliked_distance']:.3f}. This car is closer to a liked representative.")
                    if membership is not None:
                        weights=membership[position];top=np.argsort(-weights)[:3]
                        st.write('Related taste profiles: '+ ' · '.join(f'{j+1}: {weights[j]:.0%}' for j in top if weights[j]>.001))
                        st.caption(f'Unassigned membership: {max(0,1-float(weights.sum())):.0%}. Membership strengths are model affinities, not probabilities of liking this car.')
                    else:st.write(f'Related profile {profile+1}.')
                    if 'medoid_rmse' in arrays or 'medoid_cosine' in arrays or 'medoid_l2' in arrays:
                        distances=arrays.get('medoid_rmse',arrays.get('medoid_cosine',arrays.get('medoid_l2')))[position];near=np.argsort(distances)[:3]
                        st.caption('Nearest representatives · FAISS distance: '+ ' · '.join(f'Profile {j+1}: {distances[j]:.3f}' for j in near))
                        st.caption(('RMSE is sqrt(mean((x − y)²)) over catalogue-column-standardized features, without row unit normalization. Magnitude matters; this is not a dollar error.' if 'medoid_rmse' in arrays else 'Cosine distance is 1 minus cosine similarity. Lower means closer. No extra column StandardScaler is used.')+' Lower means closer; this is not a dollar error or probability.')
                    if arrays['noise'][position]:st.caption('Exploratory match: HDBSCAN treated this advert as density noise; its soft affinities remain available without forcing a hard assignment.')
                with st.expander(f'Original description & specifications · pick {rank}'):
                    st.caption('Short excerpt from the historical advert; source text is not independently verified.')
                    st.text(car.get('description') or 'Description not supplied.')
                    st.write({k:car.get(k) for k in ['condition','drive','cylinders','title_status','paint_color','state']})
        with st.expander('Model scorecard · what we measured'):
            a,b,c=st.columns(3)
            a.metric('Held-out NDCG@10 · inferred preferences',f"{result['ndcg10']:.3f}" if result['ndcg10'] is not None else 'Not defined')
            b.metric('Random-order baseline',f"{result['random_ndcg10']:.3f}" if result['random_ndcg10'] is not None else 'Not defined')
            c.metric('Actual ratings supplied',profile_count)
            st.info('The displayed list is reordered by highest cosine distance. NDCG below evaluates the underlying ranker before this presentation sort. This NDCG@10 measures agreement with soft preferences inferred from cluster memberships. Thousands of adverts still come from only the representative ratings you supplied. It is not measured customer satisfaction or proof the recommendations are good.')
        with st.expander('Did we find your type? · rate your shortlist'):
            values=[]
            feedback_id=hashlib.sha256(signature.encode()).hexdigest()[:12]
            for i,car in enumerate(cars):values.append(st.radio(f"{i+1}. {int(car['year'])} {car.get('manufacturer')} {car.get('model')}",list(LABELS),index=None,format_func=LABELS.get,horizontal=True,key=f'taste_feedback_{feedback_id}_{i}'))
            if all(v in LABELS for v in values) and len(set(values))>1:
                direct=ndcg10(np.array(values)-1,np.arange(len(cars),0,-1))
                st.metric('NDCG@10 against your additional ratings',f'{direct:.3f}')
                st.caption('Evaluates ordering within the displayed cars only. It does not measure catalogue-wide recall. This feedback is not used to train the displayed list.')
        with st.expander('Technical result'):
            st.json({k:v for k,v in result.items() if k not in ['positions','scores']},expanded=False)
    with st.expander('The business view · what Drive Chemistry does'):
        st.markdown('**A starting point for discovery.** Drive Chemistry turns a handful of reactions into a shortlist of historical asking adverts. It can help you explore styles, price levels and alternatives before a real vehicle search.')
        st.markdown('**Your taste, not a popularity contest.** Similar vehicle profiles share some of your rating signal. Likes add evidence; dislikes subtract it. Cars closer to a disliked representative than a liked one are excluded. A ranking model selects eligible cars; the current presentation then puts larger cosine distances first.')
        st.markdown('**Prices need context.** Displayed amounts are date-adjusted asking-price scenarios at the recorded mileage. They are not completed-sale values, binding offers or live inventory. Changing the date does not rebuild the taste profiles.')
        st.caption(f"Catalogue: {meta['rows']:,} retained historical cars · {meta['density_clusters']} taste profiles · {meta['noise_fraction']:.1%} classified as density noise. Noise cars may still receive soft affinities.")
        st.write('The profile catalogue is learned offline. Your reactions train only the session ranking; no account, demographic profile or other users’ ratings are used.')
    with st.expander('Under the hood · features, clustering and ranking'):
        st.markdown('**Vehicle representation.** Common categorical values become one-hot indicators; rare or unseen values share an Other bucket. Numeric features include year, log mileage, missing-value indicators, log November-2026 asking-price scenarios and a stored cross-fitted MiniLM description-price estimate. Raw text embeddings are not used for this catalogue.')
        if meta.get('density_method')=='minibatch-kmeans-soft':
            st.write('MiniBatchKMeans fits 10 or 25 groups in balanced Euclidean vehicle features. The real assigned car nearest each center is shown. Soft affinities use the three nearest centers: exp(-distance² / (2 × temperature²)), normalized across those centers. Temperature is fitted offline from the median nearest/second-nearest squared-distance gap. These are similarity weights, not probabilities or HDBSCAN density memberships.')
        elif meta.get('density_method')=='direct-weighted-euclidean':
            st.write(meta['method_description'])
        elif meta.get('density_method')=='direct-weighted-manhattan':
            st.markdown('**Soft profiles.** HDBSCAN is fitted directly to these features using weighted Manhattan distance: 50% categorical mismatch rate plus 50% mean absolute numeric difference after frozen robust range scaling. One-hot swaps are counted once per field. There is no dimensionality reduction in this release. The 10- and 25-profile tests use cuts of the fitted density hierarchy.')
        else:
            st.write(meta.get('method_description','See the installed catalogue metadata for its clustering method.'))
        st.markdown('**Personal preference.** Direct ratings 1, 2, 3, 4 become signed evidence −1.5, −0.5, +0.5, +1.5. No inverse-rating transform or session scaler is applied. Soft cluster evidence has 75% weight; representative-distance affinity has 25%. Unassigned membership adds no evidence. Candidates need positive net evidence and must be nearer to a liked than a disliked representative when both exist.')
        st.markdown('**Similarity and display.** FAISS computes exact cosine distance on the existing encoded feature vectors, without an added feature-column StandardScaler. L2 normalization is used only to calculate cosine. Distance is 1 minus cosine similarity: smaller means closer. The selected shortlist is displayed in descending distance to the nearest liked representative, as a discovery order.')
        st.code('XGBRanker(objective="rank:ndcg", eval_metric="ndcg@10",\n          tree_method="hist", enable_categorical=True,\n          max_depth=3, n_jobs=2)')
        st.write('XGBoost learns a vehicle-only ordering from inherited preference evidence, original log asking prices and supported date-adjusted log-price scenarios. User IDs, demographics, cluster IDs and prototype IDs are not predictors. Each advert appears once. A deterministic vehicle-ID split holds out roughly 20% for early stopping; prototypes stay in training, then the final ranker fits the full catalogue. Selected cars are reordered for display afterwards.')
    with st.expander('Evidence & limits · how to judge the results'):
        st.write('NDCG in the scorecard measures agreement with preferences inferred from the same representative ratings. It evaluates the ranker before the display sort; it does not establish customer satisfaction, catalogue-wide recall or commercial accuracy.')
        st.write('The feature vocabulary and numeric ranges use the reference catalogue. This is transductive exploration, not an independent external benchmark. Rare-category pooling and broad clusters can hide distinctions that matter to you. Rate the shortlist for a separate check of its displayed ordering; that feedback does not retrain the list you see.')
        st.caption('Historical data: Craigslist Used Cars, Tom / Zenodo 10457828, CC BY 4.0. Date scenarios use the saved depreciation curves and CPI/Holt market clock. Historical seller claims and vehicle condition are not independently verified.')
