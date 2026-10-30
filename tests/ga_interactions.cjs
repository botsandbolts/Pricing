const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
let now=0,events=[],handlers={},selected=[];
const document={hidden:false,body:{},querySelectorAll:()=>selected,
 addEventListener:(n,f)=>{(handlers[n]??=[]).push(f);},removeEventListener:()=>{}};
const ctx={window:{gtag:(...args)=>events.push(args)},document,
 performance:{now:()=>now},MutationObserver:class{observe(){}},requestAnimationFrame:f=>f(),setInterval:()=>{},setTimeout:()=>{}};
vm.createContext(ctx);vm.runInContext(fs.readFileSync('pricing/consent/notice.js','utf8').split('function pricingGoogleAnalytics')[0],ctx);
const t=vm.runInContext('installPricingInteractions()',ctx);
const node=(text,role='button')=>({textContent:text,getAttribute:n=>n==='role'?role:null,
 matches:s=>s!=='summary',getClientRects:()=>[1]});
const click=text=>handlers.click[0]({composedPath:()=>[node(text)]});
click('Estimate asking price');assert.equal(events.length,0);
ctx.window.__pricingGoogleAnalytics={allowed:true,featureConsent:true,id:'G-TEST'};t.consentChanged();
assert.equal(events[0][1],'panel_view');
now=5000;click('📊 Evaluation');assert.equal(t.state.panel,'evaluation');
const dwell=events.find(x=>x[1]==='panel_engagement');assert(dwell);assert.equal(dwell[2].duration_seconds,5);
assert.equal(events.at(-1)[1],'page_view');assert.match(events.at(-1)[2].page_location,/panels\/evaluation$/);
click('PRIVATE SELLER TEXT 555-1234');assert.equal(events.length,5);
click('Build all explanation graphs');assert.equal(events.at(-1)[1],'explanation_requested');
click('LinkedIn');assert.equal(events.at(-1)[2].method,'LinkedIn');
const before=events.length;vm.runInContext('installPricingInteractions()',ctx);
assert.equal(handlers.click.length,1);assert.equal(events.length,before);
ctx.window.__pricingGoogleAnalytics.allowed=false;t.consentChanged();click('Estimate asking price');assert.equal(events.length,before);
assert(!JSON.stringify(events).includes('PRIVATE'));assert(!JSON.stringify(events).includes('555-1234'));
// Completion transport ignores passive hidden-panel calculations and deduplicates rerenders.
ctx.window.__pricingGoogleAnalytics.allowed=true;t.consentChanged();
let script=fs.readFileSync('pricing/consent/events.js','utf8').replace('export default function','function deliverComponent');
vm.runInContext(script,ctx);
ctx.component={data:{events:[{id:'1',name:'estimate_completed',params:{market_side:'ask',feature_name:'calculator'}},
{id:'2',name:'estimate_completed',params:{market_side:'bid',feature_name:'model_lab'}}]}};
vm.runInContext('deliverComponent(component)',ctx);assert(!events.some(x=>x[1]==='estimate_completed'));
t.state.panel='calculator';vm.runInContext('deliverComponent(component)',ctx);vm.runInContext('deliverComponent(component)',ctx);
assert.equal(events.filter(x=>x[1]==='estimate_completed').length,1);
console.log('GA interaction consent, navigation, dwell, privacy, hidden panels and deduplication checks passed');

// Every main panel is bounded, including Outlier lab. Repeated clicks are not views.
click('📊 Evaluation');
for(const name of ['🚗 Price my car','↔ Two-market garage','📅 Value over time','🧠 The science','📊 Evaluation','🧪 Model lab','🕵️ Deal Detective','🧹 Outlier lab']) {
 click(name);const count=events.length;click(name);assert.equal(events.length,count);
 assert.equal(events.at(-1)[1],'page_view');
 assert.match(events.at(-1)[2].page_location,/\/panels\/[a-z_]+$/);
}
click('🔎 Reveal my advert’s price drivers');assert.equal(events.at(-1)[2].action_name,'phrase_drivers');
click('Download experiment evidence · JSON');assert.equal(events.at(-1)[2].action_name,'outlier_evidence');
// Keyboard/programmatic tab changes are discovered from selected tabs too.
selected=[node('📅 Value over time','tab')];
handlers.visibilitychange[0]();assert.equal(t.state.panel,'value_over_time');
// Labelled control changes record the category, never the entered value.
const control={getAttribute:n=>n==='aria-label'?'Make':null,matches:()=>true,value:'PRIVATE PHONE 555-1234'};
handlers.change[0]({composedPath:()=>[control]});
assert.equal(events.at(-1)[1],'setting_changed');assert.equal(events.at(-1)[2].action_name,'vehicle_make');
const changedCount=events.length;handlers.change[0]({composedPath:()=>[control]});assert.equal(events.length,changedCount);
// Bounded expander identities distinguish sections and open/close intent.
const section={textContent:'Valuation date & auction coverage',parentElement:{open:false},getAttribute:()=>null,matches:()=>true};
handlers.click[0]({composedPath:()=>[section]});
assert.equal(events.at(-1)[1],'section_toggled');assert.equal(events.at(-1)[2].action_name,'valuation_clock');
assert.equal(events.at(-1)[2].expanded,'true');
assert(!JSON.stringify(events).includes('555-1234'));
console.log('All eight tabs, virtual page views, keyboard navigation, named controls and sections passed');

// Streamlit summaries prepend a material icon to textContent.
const iconSummary={...section,textContent:'keyboard_arrow_downExact survivor counts · no disappearing-car magic',querySelector:()=>({textContent:'Exact survivor counts · no disappearing-car magic'})};
handlers.click[0]({composedPath:()=>[iconSummary]});
assert.equal(events.at(-1)[2].action_name,'survivor_counts');

const radio={getAttribute:()=>null,matches:()=>true,parentElement:{getAttribute:n=>n==='aria-label'?'Choose a market':null}};
now+=1000;handlers.change[0]({composedPath:()=>[radio]});
assert.equal(events.at(-1)[2].action_name,'outlier_market');
