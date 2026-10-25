import {Tokenizer} from './vendor/tokenizers.mjs';
export function clean(text){return String(text).replace(/<[^>]+>/g,' ').replace(/https?:\/\/\S+|\b\S+@\S+\b/g,' ').replace(/(?:[$€£]\s*\d[\d,.]*|\b\d[\d,.]*\s*(?:usd|dollars?|euros?|per month|\/mo|monthly)\b|\b(?:price|asking|payment|down payment)\s*[:=]?\s*\$?\d[\d,.]*)/gi,' price_removed ').replace(/\s+/g,' ').trim();}
export function createTokenizer(data,config){return new Tokenizer(data,config);}
export function tokenIds(text,tokenizer){return tokenizer.encode(text,{add_special_tokens:false}).ids;}
export function normalize(v){const n=Math.hypot(...v)||1;return v.map(x=>x/n);}
export async function encode(text,cfg,tokenizer,session,ort,chunked=true,progress=()=>{}){
 const ids=tokenIds(text,tokenizer),budget=cfg.chunk_text_tokens,prefix=cfg.prefix_ids||[],vector=Array(cfg.dimensions).fill(0);let count=0;
 for(let start=0;start<Math.max(1,ids.length);start+=budget){
  const seq=[cfg.bos,...prefix,...ids.slice(start,start+budget),cfg.eos],n=seq.length;const tensor=x=>new ort.Tensor('int64',BigInt64Array.from(x,BigInt),[1,n]);const feed={input_ids:tensor(seq),attention_mask:tensor(seq.map(()=>1))};if(session.inputNames.includes('token_type_ids'))feed.token_type_ids=tensor(seq.map(()=>0));
  const out=(await session.run(feed))[session.outputNames[0]],dim=cfg.dimensions;const p=Array(dim).fill(0);for(let j=0;j<n;j++)for(let k=0;k<dim;k++)p[k]+=out.data[j*dim+k]/n;const unit=normalize(p);for(let k=0;k<dim;k++)vector[k]+=unit[k];count++;progress(count,Math.ceil(ids.length/budget)||1);if(!chunked)break;
 }return {vector:normalize(vector),tokens:ids.length,chunks:count,characters:text.length};
}
export function predict(embedding,model){
 const x=Float32Array.from(embedding);let log=0;
 for(const tree of model.forest){let node=0;while(tree.left[node]!==-1){node=x[tree.feature[node]]<=tree.threshold[node]?tree.left[node]:tree.right[node];}log+=tree.value[node];}
 log/=model.forest.length;const q=model.meta.band80_log_residual;return{log,price:Math.max(0,Math.expm1(log)),range:[Math.max(0,Math.expm1(log-q)),Math.expm1(log+q)]};
}
