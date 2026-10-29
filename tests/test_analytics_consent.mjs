import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';

const code=(await readFile(new URL('../pricing/consent/notice.js',import.meta.url),'utf8'))
  .replace('export default function(component)', 'function render(component)');

function fixture(enabled=true) {
  const cookies=new Map(), storage=new Map(), nodes=new Map();
  const document={};
  Object.defineProperty(document,'cookie',{
    get:()=>[...cookies].map(([k,v])=>k+'='+v).join('; '),
    set:value=>{const [pair,...attrs]=value.split('; ');const at=pair.indexOf('=');
      if(attrs.includes('Max-Age=0'))cookies.delete(pair.slice(0,at));
      else cookies.set(pair.slice(0,at),pair.slice(at+1));}
  });
  const root={querySelector:key=>{
    if(!nodes.has(key))nodes.set(key,{textContent:'',open:false,showModal(){this.open=true},close(){this.open=false}});
    return nodes.get(key);
  }};
  let fetches=0, outputs=[], triggers=[];
  const context=vm.createContext({document,sessionStorage:{getItem:k=>storage.get(k)||null,
    setItem:(k,v)=>storage.set(k,v),removeItem:k=>storage.delete(k)},
    crypto:globalThis.crypto,navigator:{userAgent:'Chrome/123 Mobi'},AbortSignal,
    fetch:async()=>{fetches++;return {ok:true,json:async()=>({country_code:'DE',city:'Berlin',region:'Berlin',ip:'SECRET_IP',latitude:'SECRET_COORDINATE'})}},
    data:{enabled,version:'v1',geo_mode:'geojs',controller:'Test operator',contact:'Test contact'},root,
    send:(key,value)=>outputs.push(value),trigger:(key,value)=>triggers.push({key,value})});
  vm.runInContext(code+'\nrender({parentElement:root,data,setStateValue:send,setTriggerValue:trigger});',context);
  return {cookies,storage,nodes,get fetches(){return fetches},get outputs(){return outputs},triggers,context};
}

test('no analytics ID, lookup or device data before consent; rejection stays off',async()=>{
  const f=fixture();await new Promise(setImmediate);
  assert.equal(f.fetches,0);assert.equal(f.cookies.has('pricing_visitor'),false);
  assert.equal(f.storage.size,0);assert.equal(f.nodes.get('dialog').open,true);
  assert.equal(f.outputs.length,0);
  f.nodes.get('#reject').onclick();await new Promise(setImmediate);
  assert.equal(f.fetches,0);assert.equal(f.outputs.at(-1).choice,'rejected');
  assert.equal(f.cookies.has('pricing_visitor'),false);
});

test('accept saves only coarse location; withdraw stops analytics and clears identifiers',async()=>{
  const f=fixture();f.nodes.get('#accept').onclick();await new Promise(setImmediate);
  assert.equal(f.fetches,1);assert.equal(f.outputs.at(-1).choice,'accepted');
  assert.equal(f.outputs.at(-1).location.city,'Berlin');
  assert.equal(f.outputs.at(-1).location.country,'DE');
  const saved=JSON.stringify([f.outputs,[...f.cookies],[...f.storage]]);
  assert.equal(saved.includes('SECRET_IP'),false);assert.equal(saved.includes('SECRET_COORDINATE'),false);
  f.nodes.get('#reject').onclick();await new Promise(setImmediate);
  assert.equal(f.outputs.at(-1).choice,'rejected');assert.equal(f.cookies.has('pricing_visitor'),false);
  assert.equal(f.storage.size,0);assert.equal(f.fetches,1);
});

test('disabled integration makes no lookup and does not auto-open a consent solicitation',async()=>{
  const f=fixture(false);await new Promise(setImmediate);
  assert.equal(f.nodes.get('dialog').open,false);assert.equal(f.nodes.get('#accept').disabled,true);
  assert.equal(f.fetches,0);assert.equal(f.storage.size,0);
});

test('connection retry is available only after consent and never contains credentials',async()=>{
  const f=fixture();await new Promise(setImmediate);
  f.nodes.get('#check-connection').onclick();assert.equal(f.triggers.length,0);
  f.nodes.get('#accept').onclick();await new Promise(setImmediate);
  assert.equal(f.nodes.get('#check-connection').hidden,false);
  f.nodes.get('#check-connection').onclick();
  assert.deepEqual(f.triggers,[{key:'check_connection',value:true}]);
  f.nodes.get('#reject').onclick();await new Promise(setImmediate);
  assert.equal(f.nodes.get('#check-connection').hidden,true);
  f.nodes.get('#check-connection').onclick();assert.equal(f.triggers.length,1);
});
