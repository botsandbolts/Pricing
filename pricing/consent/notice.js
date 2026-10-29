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

// Basic consent mode: the Google library is never requested before acceptance.
function pricingGoogleAnalytics(data, allowed, root) {
  const interactions = installPricingInteractions();
  const id = data.google_tag_id;
  if (!/^G-[A-Z0-9]+$/.test(id || '')) return;
  const flag = 'ga-disable-' + id;
  const status = root.querySelector('#google-status');
  const clearCookies = () => {
    for (const pair of document.cookie.split('; ')) {
      const name = pair.split('=')[0];
      if (/^_ga(?:_|$)/.test(name))
        document.cookie = name + '=; Path=/; Max-Age=0; SameSite=Lax; Secure';
    }
  };
  let state = window.__pricingGoogleAnalytics;
  if (!allowed) {
    window[flag] = true;
    if (state?.allowed) {
      state.allowed = false;
      window.gtag?.('consent', 'update', {analytics_storage:'denied',
        ad_storage:'denied', ad_user_data:'denied', ad_personalization:'denied'});
    }
    clearCookies();
    interactions.consentChanged();
    status.textContent = 'Google Analytics: off until you accept.';
    return;
  }
  window[flag] = false;
  if (!state) {
    state = window.__pricingGoogleAnalytics = {id, allowed:false};
    window.dataLayer = window.dataLayer || [];
    window.gtag = window.gtag || function(){window.dataLayer.push(arguments);};
    window.gtag('consent','default',{analytics_storage:'denied',ad_storage:'denied',
      ad_user_data:'denied',ad_personalization:'denied'});
    window.gtag('js',new Date());
    // Never send a garage query, seller text, referrer, visitor ID or receipt key.
    window.gtag('set',{page_location:'https://pricing-5a4k2zenhw2nte9xpshxs3.streamlit.app/',
      page_referrer:'',page_title:'Car Price Studio | US used cars'});
    window.gtag('config',id,{send_page_view:false,allow_google_signals:false,
      allow_ad_personalization_signals:false,cookie_domain:'none',cookie_path:'/',
      cookie_expires:7776000,cookie_update:false,ignore_referrer:true});
    const script = document.createElement('script');
    script.async = true;
    script.src = 'https://www.googletagmanager.com/gtag/js?id=' + id;
    script.referrerPolicy = 'no-referrer';
    script.onload = () => {state.loaded = true;};
    script.onerror = () => {state.failed = true;};
    document.head.appendChild(script);
  }
  if (!state.allowed) {
    state.allowed = true;
    window.gtag('consent','update',{analytics_storage:'granted',ad_storage:'denied',
      ad_user_data:'denied',ad_personalization:'denied'});
    window.gtag('event','page_view',{send_to:id,
      page_location:'https://pricing-5a4k2zenhw2nte9xpshxs3.streamlit.app/',
      page_referrer:'',page_title:'Car Price Studio | US used cars'});
  }
  interactions.consentChanged();
  status.textContent = state.failed ? 'Google Analytics: blocked or unavailable; pricing still works.'
    : state.loaded ? 'Google Analytics: tag loaded after acceptance.'
    : 'Google Analytics: requested after acceptance; delivery is not yet confirmed.';
}

export default function(component) {
  const {parentElement: root, data, setStateValue, setTriggerValue} = component;
  const dialog = root.querySelector('dialog');
  const prefName = 'pricing_consent';
  const visitorName = 'pricing_visitor';
  const read = name => {
    try {return decodeURIComponent(document.cookie.split('; ').find(v=>v.startsWith(name+'='))?.split('=').slice(1).join('=') || '');}
    catch {return '';}
  };
  const write = (name, value, days) => {
    document.cookie = name+'='+encodeURIComponent(value)+'; Path=/; Max-Age='+Math.round(days*86400)+'; SameSite=Lax; Secure';
  };
  const validId = v => /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(v || '');
  const clearIdentity = () => {
    write(visitorName, '', 0);
    try {sessionStorage.removeItem('pricing_session');sessionStorage.removeItem('pricing_geo');} catch {}
  };
  let pref = null;
  try {pref = JSON.parse(read(prefName));} catch {}
  const age = pref ? Date.now() - Date.parse(pref.at) : Infinity;
  if (!pref || pref.version !== data.version || !(age >= 0 && age < 90*86400000) ||
      !['accepted','rejected'].includes(pref.choice)) pref = null;
  if (!pref || pref.choice !== 'accepted' || !data.enabled) clearIdentity();
  pricingGoogleAnalytics(data, pref?.choice === 'accepted' && data.enabled, root);
  root.querySelector('#operator').textContent = 'Operator: '+data.controller+'. Contact: '+data.contact+'.';
  root.querySelector('#status').textContent = data.enabled ? '' : 'Optional analytics are off. No analytics visitor ID is active. You can use every pricing tool normally.';
  root.querySelector('#status').hidden = data.enabled;
  root.querySelector('#delivery').textContent = 'Collection status: '+(data.backend_status || 'pending')+'.';
  root.querySelector('#consent-info').hidden = !data.enabled;
  root.querySelector('.actions').hidden = !data.enabled;
  root.querySelector('#geo-short').textContent = data.geo_mode === 'geojs'
    ? 'After acceptance, GeoJS receives your IP to estimate city/country; we save only the approximate location.'
    : 'Location is estimated locally from an IP supplied by our host.';
  root.querySelector('#accept').disabled = !data.enabled;
  root.querySelector('#geo-notice').textContent = data.geo_mode === 'geojs'
    ? 'After acceptance, your browser contacts GeoJS (using MaxMind GeoLite data) to estimate your country/city. GeoJS and its network provider receive your IP through that connection. We keep only country, city and region; the returned IP, coordinates and other fields are discarded. No lookup is made before consent.'
    : 'After acceptance, a local geolocation database may temporarily process an IP supplied by our hosting provider. No visitor IP is sent to an external geolocation API. Only country, city and region are saved.';
  const lookup = async () => {
    if (data.geo_mode !== 'geojs') return null;
    try {
      const cached = JSON.parse(sessionStorage.getItem('pricing_geo') || 'null');
      if (cached && Date.now()-cached.at < 3600000 && cached.session === sessionStorage.getItem('pricing_session')) return cached.location;
      const response = await fetch('https://get.geojs.io/v1/ip/geo.json', {
        credentials:'omit',referrerPolicy:'no-referrer',signal:AbortSignal.timeout(3000)});
      if (!response.ok) return null;
      const result = await response.json();
      const clean = value => typeof value==='string' ? value.slice(0,100) : null;
      const location = {country:/^[A-Z]{2}$/.test(result.country_code)?result.country_code:null,
        city:clean(result.city),region:clean(result.region),geo_provider:'GeoJS / MaxMind GeoLite'};
      if (pref?.choice !== 'accepted' || !data.enabled || !sessionStorage.getItem('pricing_session')) return null;
      // Raw IP/lat/lon never leave this function or enter storage/component state.
      sessionStorage.setItem('pricing_geo',JSON.stringify({at:Date.now(),location,session:sessionStorage.getItem('pricing_session')}));
      return location;
    } catch {return null;}
  };
  const publish = async () => {
    let payload = {choice: pref?.choice || 'unknown', version: data.version};
    if (pref?.choice === 'accepted' && data.enabled) {
      try {
        let visitor = read(visitorName), session = sessionStorage.getItem('pricing_session');
        if (!validId(visitor)) {visitor=crypto.randomUUID();write(visitorName,visitor,90);}
        if (read(visitorName) !== visitor) throw Error('Cookies unavailable');
        if (!validId(session)) {session=crypto.randomUUID();sessionStorage.setItem('pricing_session',session);}
        const ua=navigator.userAgent;
        const browser=/Edg\//.test(ua)?'Edge':/Firefox\//.test(ua)?'Firefox':/Chrome\//.test(ua)?'Chrome':/Safari\//.test(ua)?'Safari':'Other';
        payload={...payload, visitor_id:visitor, session_id:session, consent_at:pref.at, browser,
          device:/Mobi|Android/i.test(ua)?'mobile':'desktop'};
        const location = await lookup();
        // Ignore a result arriving after the person withdrew or disabled collection.
        if (pref?.choice !== 'accepted' || !data.enabled) return;
        if (location) payload.location = location;
      } catch {payload={choice:'rejected',version:data.version};clearIdentity();}
    }
    root.querySelector('#visitor').textContent=payload.visitor_id ? 'Your analytics visitor ID: '+payload.visitor_id : 'No analytics visitor ID is active.';
    root.querySelector('#check-connection').hidden = !(payload.choice === 'accepted' && data.enabled);
    // A rerender must not create a Python rerun loop.
    const serialized=JSON.stringify(payload);
    if (payload.choice !== 'unknown' && root.__publishedConsent !== serialized && JSON.stringify(data.current) !== serialized) {
      root.__publishedConsent=serialized;
      setStateValue('consent',payload);
    }
  };
  root.querySelector('#privacy').onclick=()=>dialog.showModal();
  root.querySelector('#close').onclick=()=>dialog.close();
  root.querySelector('#check-connection').onclick=()=>{
    if (pref?.choice === 'accepted' && data.enabled) {
      root.querySelector('#delivery').textContent='Checking collection connection…';
      setTriggerValue('check_connection',true);
    }
  };
  for (const choice of ['accepted','rejected']) {
    root.querySelector(choice==='accepted'?'#accept':'#reject').onclick=()=>{
      pref={choice,version:data.version,at:new Date().toISOString()};
      write(prefName,JSON.stringify(pref),90);
      if(choice==='rejected')clearIdentity();
      pricingGoogleAnalytics(data, choice==='accepted' && data.enabled, root);
      publish();dialog.close();
    };
  }
  publish();
  if (!pref && data.enabled && !root.__prompted) {root.__prompted=true;dialog.showModal();}
}
