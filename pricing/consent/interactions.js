// Deliberately finite labels: never send input values, link URLs or arbitrary DOM text.
function installPricingInteractions() {
  if (window.__pricingInteractions) return window.__pricingInteractions;
  const panels = new Map([
    ['🚗 Price my car','calculator'],['↔ Two-market garage','two_market'],
    ['📅 Value over time','value_over_time'],['🧠 The science','science'],
    ['📊 Evaluation','evaluation'],['🧪 Model lab','model_lab'],['🕵️ Deal Detective','deal_detective']
  ]);
  const subpanels = new Map([
    ['📸 Import advert','import_advert'],['🕵️ Price verdict','price_verdict'],
    ['📤 Share & go viral','sharing'],['🚗 Compare cars','compare_cars'],['🔎 Evidence','evidence'],
    ['🎯 Beat the model','price_game'],['🧰 Buyer briefing','buyer_briefing'],['🧾 Sale feedback','sale_feedback'],
    ['Explain it like I buy cars','plain_explanation'],['Under the bonnet · technical','technical_explanation'],
    ['What if?','what_if'],['Why this price?','waterfall'],['Across reference cars','reference_cars'],
    ['Feature effects','feature_effects'],['Compare cars','compare_explanations'],
    ['Year × mileage','year_mileage'],['Technical receipt','technical_receipt']
  ]);
  const actions = new Map([
    ['Estimate asking price',['estimate_requested','asking']],
    ['Let the pricing committee argue',['estimate_requested','model_comparison']],
    ['Price it in both markets',['estimate_requested','two_market']],
    ['Build all explanation graphs',['explanation_requested','all_graphs']],
    ['Build value-over-time graph',['scenario_requested','value_timeline']],
    ['📣 Share my price · image, link & group-chat fuel',['share_opened','price_receipt']],
    ['Extract vehicle details',['import_requested','text']],['Read screenshot',['import_requested','screenshot']],
    ['Use reviewed details → Price my car',['import_applied','reviewed_details']],
    ['Load this car into the calculator',['import_applied','shared_car']],
    ['Park this estimate',['shortlist_action','add']],['Inspect this car',['shortlist_action','inspect']],
    ['Value through time',['shortlist_action','timeline']],['Remove',['shortlist_action','remove']],
    ['Compare all with the selected model',['shortlist_action','compare']],
    ['Lock my guess & reveal',['game_action','reveal']],['Another practice car',['game_action','next']],
    ['Submit privately for review',['feedback_action','submit_requested']],
    ['Delete my feedback and evidence',['feedback_action','delete_requested']],
    ['Open Value over time →',['navigation_action','value_over_time']],
    ['⬇ Download image card · PNG',['download_requested','share_card']],
    ['Valuation facts · JSON',['download_requested','valuation_facts']],
    ['Export shortlist · no seller text',['download_requested','shortlist']],
    ['Take the checklist with me',['download_requested','buyer_checklist']],
    ['Download explanation receipt',['download_requested','explanation']],
    ['Download this value timeline',['download_requested','timeline']],
    ['Download selected ensemble assessment',['download_requested','assessment']],
    ['Download MiniLM assessment',['download_requested','minilm_assessment']],
    ['Download correlations and Ridge coefficients',['download_requested','model_comparison']],
    ['Save private submission/deletion receipt',['download_requested','feedback_receipt']]
  ]);
  const settings = new Map([
    ['Pricing model','pricing_model'],['Asking model','asking_model'],['Bid / auction model','bid_model'],
    ['Auction pricing option','auction_model'],['Browse 400 real car descriptions — type to search','example_description'],
    ['Valuation date','valuation_date'],['Assumed miles per year','annual_miles'],
    ['Member for exact TreeSHAP plots','shap_member'],['Dollar explanation detail','shap_detail'],
    ['Feature to inspect','shap_feature'],['Colour by numeric feature','shap_colour'],
    ['Year','vehicle_year'],['Make','vehicle_make'],['Model / trim','vehicle_model'],
    ['Mileage (miles)','vehicle_mileage'],['US state abbreviation','vehicle_state'],
    ['Body type','body_type'],['Condition','condition'],['Fuel','fuel'],['Transmission','transmission'],
    ['Drive','drive'],['Title status','title_status'],['Cylinders','cylinders'],['Paint color','paint_colour']
  ]);
  const state = {panel:'calculator',subpanel:'none',since:performance.now(),active:!document.hidden,
    allowed:false,seen:new Set(),lastChange:new Map()};
  const allowed = () => window.__pricingGoogleAnalytics?.allowed === true;
  function send(name, params={}) {
    if (!allowed() || document.hidden || !window.gtag) return;
    window.gtag('event',name,{send_to:window.__pricingGoogleAnalytics.id,
      panel_name:state.panel,subpanel_name:state.subpanel,...params});
  }
  function dwell() {
    if (state.active && state.allowed) {
      const seconds=Math.min(1800,Math.floor((performance.now()-state.since)/1000));
      if(seconds>0) send('panel_engagement',{duration_seconds:seconds});
    }
    state.since=performance.now();
  }
  function switchPanel(panel, subpanel='none') {
    if(panel===state.panel && subpanel===state.subpanel) return;
    dwell();state.panel=panel;state.subpanel=subpanel;
    send('panel_view');
  }
  function consentChanged() {
    const current=allowed();
    if(current && !state.allowed) {state.allowed=true;state.since=performance.now();send('panel_view');}
    if(!current) {state.allowed=false;state.seen.clear();state.since=performance.now();}
  }
  const label = node => (node?.getAttribute?.('aria-label') || node?.textContent || '').trim().replace(/\s+/g,' ');
  const visible = node => !!node?.getClientRects?.().length;
  function selectedPanels() {
    for(const tab of document.querySelectorAll('[role="tab"][aria-selected="true"]')) {
      if(visible(tab) && panels.has(label(tab))) {
        const p=panels.get(label(tab));
        if(p!==state.panel) switchPanel(p);
      }
    }
  }
  document.addEventListener('click',event=>{
    const node=event.composedPath().find(n=>n?.matches?.('button,[role="tab"],[role="option"],summary,a'));
    if(!node || !allowed()) return;
    const text=label(node);
    if(node.getAttribute('role')==='option') {
      const family=text.includes('CatBoost')?'catboost':text.includes('LightGBM')?'lightgbm':text.includes('XGBoost')?'xgboost':text.includes('ensemble')?'ensemble':null;
      if(family)send('model_selected',{model_family:family});
      return;
    }
    if(panels.has(text)) {switchPanel(panels.get(text));return;}
    if(subpanels.has(text)) {switchPanel(state.panel,subpanels.get(text));return;}
    if(actions.has(text)) {const [name,action_name]=actions.get(text);send(name,{action_name});return;}
    const networks=['WhatsApp','LinkedIn','Reddit','X / Twitter','X','Facebook','Telegram'];
    const method=networks.find(n=>text===n || text===`Share comparison on ${n}`);
    if(method) {send('share',{method,content_type:'valuation'});return;}
    // Expanders are counted by a bounded section identifier, never their text.
    if(node.matches('summary') || node.getAttribute('data-testid')==='stExpanderToggleIcon')
      send('explanation_opened',{action_name:'expand_section'});
  },true);
  document.addEventListener('change',event=>{
    if(!allowed())return;
    const node=event.composedPath().find(n=>n?.matches?.('input,select,textarea,[role="combobox"],[role="slider"]'));
    if(!node)return;
    let text=node.getAttribute('aria-label') || '';
    if(!settings.has(text)) {
      const holder=node.closest('[data-testid]');
      text=holder?.querySelector('label')?.textContent?.trim() || text;
    }
    const action_name=settings.get(text);if(!action_name)return;
    const now=performance.now();if(now-(state.lastChange.get(action_name)||-Infinity)<500)return;
    state.lastChange.set(action_name,now);send('setting_changed',{action_name});
  },true);
  document.addEventListener('visibilitychange',()=>{
    // Flush while leaving a visible page; gtag uses its normal transport.
    if(document.hidden) {
      if(state.active && state.allowed) {
        const seconds=Math.min(1800,Math.floor((performance.now()-state.since)/1000));
        if(seconds>0 && allowed())window.gtag('event','panel_engagement',{
          send_to:window.__pricingGoogleAnalytics.id,panel_name:state.panel,
          subpanel_name:state.subpanel,duration_seconds:seconds});
      }
      state.active=false;
    } else {state.active=true;state.since=performance.now();selectedPanels();}
  });
  let scheduled=false;
  new MutationObserver(()=>{
    if(scheduled)return;scheduled=true;
    requestAnimationFrame(()=>{scheduled=false;selectedPanels();});
  }).observe(document.body,{subtree:true,childList:true,attributes:true,attributeFilter:['aria-selected']});
  // A bounded heartbeat records visible-panel time even during long sessions.
  setInterval(()=>{if(allowed() && state.active)dwell();},30000);
  window.__pricingInteractions={send,consentChanged,state};
  selectedPanels();return window.__pricingInteractions;
}
