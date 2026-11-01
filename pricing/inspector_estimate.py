"""Own-car pricing for Inspector, using the unchanged landing-page estimator."""
def quote(root,fields,notes,date,allow):
    from pricing.hierarchy_sparse_live import estimate
    from pricing.release_contract import release_identity
    results,_,_=estimate(root,{**fields,'text':notes},date,allow)
    return {'fields':fields,'notes':notes,'date':str(date),'allow':allow,'results':results,'model_version':release_identity(root)}

def render(root,saved,bayesian):
    import streamlit as st
    from pricing.release_contract import release_identity
    from pricing.pricing_comparison import selected_receipt
    if not saved:return
    if saved.get('model_version')!=release_identity(root):st.info('The pricing release changed. Price this car again to refresh its estimates.');return
    fields=saved['fields']
    st.markdown('### Your car · '+saved['date'])
    mileage=f"{fields['odometer']:,} miles" if fields.get('odometer') is not None else 'mileage not supplied'
    st.caption(f"{fields['year']} {fields['manufacturer']} {fields['model']} · {mileage} · "+('Bayesian correction' if bayesian else 'Regular XGBoost'))
    for col,side,label,meaning in zip(st.columns(2),['ask','bid'],['Seller’s asking estimate','Wholesale auction estimate'],['Advertised price before negotiation; not a completed retail sale.','Wholesale market estimate; not a live dealer offer.']):
        with col:
            r=selected_receipt(saved['results'].get(side,{'error':'Unavailable'}),bayesian)
            if 'error' in r:st.warning(label+': '+r['error']);continue
            st.metric(label+' · USD',f"{r['price']:,.0f}");st.caption(meaning)
            if bayesian:st.caption(f"Historical 90% range, date-adjusted: USD {r['interval_low']:,.0f}–{r['interval_high']:,.0f}")
    st.caption('Same production models as the landing page. Current-date prices are model scenarios; historical interval coverage does not validate current-date coverage. The ask–auction difference is not dealer profit. Ask/Bid feature imputation may use this estimate as auxiliary context; neighbour prices are exploratory comparisons, not independent accuracy validation.')

