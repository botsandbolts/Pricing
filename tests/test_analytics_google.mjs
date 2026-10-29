import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
const code=(await readFile(new URL('../pricing/consent/notice.js',import.meta.url),'utf8')).split('export default function')[0];
function fixture() {
 const cookies=new Map(),scripts=[],window={},status={textContent:''};
 const document={head:{appendChild:s=>scripts.push(s)},createElement:()=>({})};
 Object.defineProperty(document,'cookie',{get:()=>[...cookies].map(([k,v])=>k+'='+v).join('; '),set:s=>{const pair=s.split(';')[0],i=pair.indexOf('=');if(s.includes('Max-Age=0'))cookies.delete(pair.slice(0,i));else cookies.set(pair.slice(0,i),pair.slice(i+1));}});
 const ctx=vm.createContext({installPricingInteractions:()=>({consentChanged(){}}),window,document});
 vm.runInContext(code,ctx);
 const root={querySelector:()=>status},data={google_tag_id:'G-YT533T36L8'};
 return {window,cookies,scripts,status,run:allowed=>ctx.pricingGoogleAnalytics(data,allowed,root)};
}
test('Google library is blocked before acceptance; rerenders do not duplicate views',()=>{
 const f=fixture();f.run(false);assert.equal(f.scripts.length,0);assert.equal(f.window.dataLayer,undefined);
 f.run(true);f.run(true);assert.equal(f.scripts.length,1);
 assert.equal(f.scripts[0].src,'https://www.googletagmanager.com/gtag/js?id=G-YT533T36L8');
 const q=f.window.dataLayer.map(a=>Array.from(a));
 assert.equal(q.filter(a=>a[0]==='event'&&a[1]==='page_view').length,1);
 const config=q.find(a=>a[0]==='config')[2];
 assert.equal(config.send_page_view,false);assert.equal(config.allow_google_signals,false);
 assert.equal(config.allow_ad_personalization_signals,false);assert.equal(config.cookie_domain,'none');
 const event=q.find(a=>a[0]==='event')[2];
 assert.equal(event.page_location,'https://pricing-5a4k2zenhw2nte9xpshxs3.streamlit.app/');
 assert.equal(event.page_referrer,'');assert.equal(JSON.stringify(q).includes('garage='),false);
});
test('withdrawal stops the tag and removes its cookies; reaccept reuses the script',()=>{
 const f=fixture();f.run(true);f.cookies.set('_ga','secret');f.cookies.set('_ga_YT533T36L8','secret');f.cookies.set('pricing_consent','keep');
 f.run(false);assert.equal(f.window['ga-disable-G-YT533T36L8'],true);
 assert.equal(f.cookies.has('_ga'),false);assert.equal(f.cookies.has('_ga_YT533T36L8'),false);assert.equal(f.cookies.get('pricing_consent'),'keep');
 const q=f.window.dataLayer.map(a=>Array.from(a));assert.equal(q.at(-1)[2].analytics_storage,'denied');
 f.run(true);assert.equal(f.scripts.length,1);assert.equal(f.window['ga-disable-G-YT533T36L8'],false);
});
test('blocked library never breaks pricing or reports confirmed delivery',()=>{
 const f=fixture();f.run(true);f.scripts[0].onerror();f.run(true);assert.match(f.status.textContent,/blocked or unavailable/);
});
