"""Listing decision tools. Heavy inference occurs only on an explicit action."""
import json
from pathlib import Path
import streamlit as st
from pricing.market_review import parse_inventory,duplicate_pairs,export_csv,review_estimate,batch_review


def report(root,side):
    path=Path(root)/'data/listing-insights'/(side+'.json')
    if not path.exists():return None
    return json.loads(path.read_text())


def inventory_upload(key):
    st.caption('CSV columns: manufacturer, model, year, odometer. Optional: id, price (USD), description, state, transmission, title_status, vin and other vehicle fields. Up to 100 rows, 2 MB. Pricing batches are limited to 25.')
    template=b'id,manufacturer,model,year,odometer,price,state,transmission,title_status,description\nexample-assumptions,Toyota,Corolla,2015,95000,8000,ca,automatic,unknown,Full service history. Needs new tires.\n'
    st.download_button('Download CSV template · illustrative inputs',template,'listing-review-template.csv','text/csv',key=key+'_template')
    file=st.file_uploader('Upload your inventory CSV',type=['csv'],key=key,max_upload_size=2)
    if file is None:return None,None
    payload=file.getvalue()
    try:return parse_inventory(payload),__import__('hashlib').sha256(payload).hexdigest()
    except (ValueError,UnicodeError,__import__('csv').Error) as exc:st.error(str(exc));return None,None


def render_duplicates():
    st.subheader('Same car, different story?')
    st.write('Find repeated vehicles and conflicting fields within your uploaded listings.')
    rows,signature=inventory_upload('inspector_duplicates_file')
    if rows is not None:
        if st.button('Check these listings for duplicates',key='inspector_duplicates_run'):
            st.session_state['inspector_duplicates_result']=(signature,duplicate_pairs(rows))
        saved=st.session_state.get('inspector_duplicates_result')
        if saved and saved[0]==signature:
            pairs=saved[1]
            if pairs:
                st.dataframe(pairs,hide_index=True,width='stretch')
                st.download_button('Download duplicate review',export_csv(pairs),'duplicate-review.csv','text/csv')
            else:st.info('No pairs met these conservative rules. This does not verify that every vehicle is unique.')
    st.caption('Matching supplied VIN, or strongly overlapping text with the same make/model/year and mileage within 10,000 miles. Dealer templates alone are insufficient. Different prices or mileage may reflect different dates; no fraud accusation is made. VINs are hashed for matching and excluded from results.')


def render_regions(root):
    st.subheader('Regional Price Lens')
    side=st.radio('Historical market',['Asking adverts','Auction sales'],horizontal=True,key='inspector_region_side')
    r=report(root,'ask' if side=='Asking adverts' else 'bid')
    if r is None:st.info('The original-data regional report is being prepared offline. No sample average is substituted for an adjusted regional effect.');return
    st.caption(r['basis']+' · controlled historical associations, not current regional quotes.')
    regions=r.get('regions',[])
    if not regions:st.info('No states meet the minimum of 100 training vehicles.');return
    options=[x['state'].upper() for x in regions]
    a,b=st.columns(2)
    first=a.selectbox('Region A',options,key='inspector_region_a')
    second=b.selectbox('Region B',options,index=min(1,len(options)-1),key='inspector_region_b')
    for column,name in [(a,first),(b,second)]:
        item=next(x for x in regions if x['state'].upper()==name)
        column.metric('Adjusted association vs reference',f"{item['adjusted_association_pct']:+.1f}%")
        low,high=item['bootstrap_90_pct'];column.caption(f"Exploratory 90% bootstrap interval: {low:+.1f}% to {high:+.1f}% · {item['training_vehicles']:,} training vehicles")
    st.caption('The two intervals are marginal intervals; subtracting their endpoints does not create an interval for the difference. The reference is the training-count-weighted mean of supported states.')
    with st.expander('All supported states and controls'):
        st.dataframe([{'State':x['state'].upper(),'Training vehicles':x['training_vehicles'],'Adjusted association %':round(x['adjusted_association_pct'],2),'90% low %':round(x['bootstrap_90_pct'][0],2),'90% high %':round(x['bootstrap_90_pct'][1],2)} for x in regions],hide_index=True,width='stretch')
        st.write(r['regional_method']);st.write(r['method']);st.json(r['provenance'],expanded=False)
    st.info('This compares historical conditional price patterns. Transport costs, local demand today and profitable arbitrage have not been established.')


def passport(root,saved):
    from pricing.release_contract import release_identity
    if saved.get('model_version')!=release_identity(root):
        st.info('The pricing release changed. Re-estimate the car to refresh this passport.');return
    fields=saved.get('fields',{})
    st.write(f"{fields.get('year','')} {fields.get('manufacturer','')} {fields.get('model','')} · valued for {saved.get('date','')}")
    for column,side,label in zip(st.columns(2),['ask','bid'],['Seller asking','Wholesale auction']):
        with column:
            r=saved['results'].get(side,{'error':'Unavailable'})
            st.markdown('#### '+label)
            if 'error' in r:st.warning(r['error']);continue
            st.metric('Bayesian estimate · USD',f"{r['price']:,.0f}")
            st.write(f"Date-adjusted historical 90% range: USD {r['interval_low']:,.0f}–{r['interval_high']:,.0f}")
            st.caption(f"Make support: {r.get('make_support',0):,} · model support: {r.get('model_support',0):,} · calibration pool: {r.get('interval_calibration_n',0):,}")
            if r.get('unknown_categories'):st.warning('Unseen categories: '+', '.join(r['unknown_categories']))
            if r.get('analogue'):st.warning('Newer-generation auction analogue assumption is active.')
            f=Path(root)/'models/xgboost-hierarchy'/side/'bayesian/bayesian-evaluation.json'
            if f.exists():
                evaluation=json.loads(f.read_text())
                st.caption(f"Historical test MAE: USD {evaluation['point_scores']['mae_usd']:,.0f} · 90% range coverage: {evaluation['test_coverage']:.1%}. These are population assessment results, not this car’s known error.")
    st.caption('Current-date accuracy and transported interval coverage are unvalidated. Broad support and narrow intervals do not verify condition or title. Price differences between markets are not dealer profit.')
    st.caption('Pricing release: '+saved['model_version'][:16])
    st.download_button('Download appraisal passport',json.dumps({'vehicle':fields,'valuation_date':saved['date'],'model_release':saved['model_version'],'estimates':saved['results'],'scope':'Experimental estimates and date-adjusted historical ranges; not verified current offers.'},indent=2,default=str),'appraisal-passport.json','application/json')


def render_deal(root,inspected):
    st.subheader('Bargain—or something missing?')
    st.write('Compare an advertised price with the existing Bayesian asking estimate and historical range.')
    if not inspected:st.info('Inspect your advert above first.');return
    fields,text=inspected
    from marketbridge.time_engine import today
    amount=st.number_input('Advertised asking price · USD',min_value=1.,value=8000.,step=100.,key='inspector_listed_price')
    date=st.date_input('Review valuation date',value=today(),key='inspector_review_date')
    allow=st.checkbox('Allow the newer-generation auction analogue for 2016–2021 cars',value=True,key='inspector_review_allow')
    from pricing.release_contract import release_identity
    signature=json.dumps([fields,text,str(date),allow,release_identity(root)],sort_keys=True)
    if st.button('Evaluate this car & build its passport',key='inspector_evaluate',type='primary'):
        try:
            from pricing.hierarchy_sparse_live import estimate
            with st.spinner('Evaluating this car using the installed pricing models…'):
                results,_,_=estimate(root,{**fields,'text':text},date,allow)
            st.session_state['inspector_passport']={'signature':signature,'fields':fields,'notes':text,'date':str(date),'results':results,'model_version':release_identity(root)}
        except (ValueError,OSError,KeyError,MemoryError,RuntimeError):st.warning('Evaluation is temporarily unavailable. Other Inspector tools remain usable.')
    saved=st.session_state.get('inspector_passport')
    if not saved or saved['signature']!=signature:st.caption('Press Evaluate after changing car details or date. This does not alter your saved landing-page estimate.');return
    decision=review_estimate(saved['results'].get('ask',{'error':'No asking estimate'}),amount)
    with st.container(border=True):
        st.markdown('### '+decision['priority']);st.write(decision['reason'])
        st.caption('This model comparison is not a calibrated probability of finding a bargain. Lower prices may reflect omitted condition, title or specification details.')
    passport(root,saved)


def render_batch(root):
    st.subheader('Dealer review queue')
    st.write('Sort a small inventory by pricing discrepancies and information needing attention. Existing ask and auction models are evaluated sequentially.')
    rows,signature=inventory_upload('inspector_batch_file')
    if rows is None:return
    from marketbridge.time_engine import today
    from pricing.release_contract import release_identity
    date=st.date_input('Batch valuation date',value=today(),key='inspector_batch_date')
    allow=st.checkbox('Allow newer-generation auction analogues',value=True,key='inspector_batch_allow')
    identity=json.dumps([signature,str(date),allow,release_identity(root)])
    if len(rows)>25:st.info('Duplicate review accepts 100 rows; reduce this file to 25 rows to run pricing.');return
    if st.button('Evaluate inventory · up to 25 vehicles',key='inspector_batch_run',type='primary'):
        from pricing.hierarchy_sparse_live import estimate
        with st.spinner('Reviewing vehicles sequentially…'):
            output=batch_review(rows,lambda fields,date,allow:estimate(root,fields,date,allow)[0],date,allow)
        st.session_state['inspector_batch_results']=(identity,output)
    saved=st.session_state.get('inspector_batch_results')
    if saved and saved[0]==identity:
        st.dataframe(saved[1],hide_index=True,width='stretch')
        st.download_button('Download the review queue',export_csv(saved[1]),'dealer-review-queue.csv','text/csv')
        st.caption('Sorted for appraisal review, not expected profit. One invalid vehicle does not stop other rows. Asking discrepancies use asking estimates only. Historical ranges are date-adjusted; current-date coverage is unvalidated.')


def render_information(root):
    st.subheader('Does the advert tell us anything useful?')
    r=report(root,'ask')
    if r is None:st.info('The original-data description experiment is being prepared offline. No invented information premium is shown.');return
    e=r.get('description_experiment')
    if not e or 'with_minilm_mae_usd' not in e:st.info('The MiniLM description experiment is being prepared offline.');return
    a,b,c=st.columns(3)
    a.metric('Facts-only audit MAE · USD',f"{e['base_mae_usd']:,.0f}")
    b.metric('Facts + MiniLM · USD',f"{e['with_minilm_mae_usd']:,.0f}")
    c.metric('Historical MAE saved · USD',f"{e['mae_saved_usd']:+,.0f}")
    low,high=e['paired_vehicle_bootstrap_95_usd'];st.caption(f"Paired vehicle-bootstrap 95% interval for MAE savings: USD {low:+,.0f} to {high:+,.0f}. Positive means MiniLM helped this audit model; negative means it hurt.")
    st.caption(f"{e['embedding_dimensions']} frozen description features · {e['encoder']}")
    st.write(e['interpretation'])
    st.info('The comparison measures predictive information in historical text. It does not show that rewriting an advert changes the vehicle’s value.')
    with st.expander('Research method and original-data provenance'):
        st.write(r['method']);st.write(r['limits']);st.write(r['rows']);st.json(r['provenance'],expanded=False)
    st.divider();st.subheader('Lemon inspection · keep the difficult cars')
    out=r['outlier_review'];st.write(out['method'])
    st.caption(f"{out['flagged_rows']:,} of {out['training_rows']:,} research-training cars selected for residual inspection. {out['limits']}")
    st.dataframe(out['diagnostics'],hide_index=True,width='stretch')
    st.caption('This audit does not delete any car or change Lemon Squeezer, XGBoost or Bayesian production weights.')
