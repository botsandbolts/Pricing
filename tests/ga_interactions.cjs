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
ctx.window.__pricingGoogleAnalytics={allowed:true,id:'G-TEST'};t.consentChanged();
assert.equal(events[0][1],'panel_view');
now=5000;click('📊 Evaluation');assert.equal(t.state.panel,'evaluation');
assert.equal(events[1][1],'panel_engagement');assert.equal(events[1][2].duration_seconds,5);
click('PRIVATE SELLER TEXT 555-1234');assert.equal(events.length,3);
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
