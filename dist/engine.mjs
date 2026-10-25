export function clean(text) {
 return String(text).replace(/<[^>]+>/g,' ').replace(/https?:\/\/\S+|\b\S+@\S+\b/g,' ')
 .replace(/(?:[$€£]\s*\d[\d,.]*|\b\d[\d,.]*\s*(?:usd|dollars?|euros?|per month|\/mo|monthly)\b|\b(?:price|asking|payment|down payment)\s*[:=]?\s*\$?\d[\d,.]*)/gi,' price_removed ').replace(/\s+/g,' ').trim().toLowerCase();
}
export function wordpieces(text,vocab){
 let normalized='';
 for(const c of text){const n=c.codePointAt(0);if(n===0||n===0xfffd)continue;if(/\s/u.test(c)){normalized+=' ';continue;}if(/[\p{Cc}\p{Cf}]/u.test(c))continue;if(/\p{Script=Han}/u.test(c))normalized+=' '+c+' ';else normalized+=c;}
 normalized=normalized.toLowerCase().normalize('NFD').replace(/\p{Mn}/gu,'');
 let separated='';for(const c of normalized){const n=c.codePointAt(0);const punct=(n>=33&&n<=47)||(n>=58&&n<=64)||(n>=91&&n<=96)||(n>=123&&n<=126)||/\p{P}/u.test(c);separated+=punct?' '+c+' ':c;}
 const ids=[];for(const token of separated.trim().split(/\s+/)){if(!token)continue;const chars=Array.from(token);if(chars.length>100){ids.push(100);continue;}let start=0,parts=[],bad=false;while(start<chars.length){let end=chars.length,found;while(start<end){const piece=(start?'##':'')+chars.slice(start,end).join('');if(Object.hasOwn(vocab,piece)){found=vocab[piece];break;}end--;}if(found===undefined){bad=true;break;}parts.push(found);start=end;}ids.push(...(bad?[100]:parts));}return ids;
}
export function sparseText(text,model){
 if(!model.termIndex)model.termIndex=new Map(model.terms.map((t,i)=>[t,i]));
 const tokens=text.match(/[\p{L}\p{N}_]{2,}/gu)||[];const counts=new Map();
 const add=t=>{const i=model.termIndex.get(t);if(i!==undefined)counts.set(i,(counts.get(i)||0)+1);};
 tokens.forEach(add);for(let j=1;j<tokens.length;j++)add(tokens[j-1]+' '+tokens[j]);
 let norm=0;const values=[];for(const[i,c]of counts){const v=(1+Math.log(c))*model.idf[i];values.push([i,v]);norm+=v*v;}norm=Math.sqrt(norm)||1;return values.map(([i,v])=>[i,v/norm]);
}
export function textScore(features,coeff){return features.reduce((s,[i,v])=>s+v*coeff[i],0);}
export function normalise(v){const n=Math.hypot(...v)||1;return v.map(x=>x/n);}
export async function embed(text,model,session,ort,onProgress=()=>{}){
 const ids=wordpieces(text.slice(0,model.meta.max_characters),model.vocab),vector=Array(384).fill(0);let chunks=0;
 for(let start=0;start<Math.max(1,ids.length);start+=254){
  const seq=[101,...ids.slice(start,start+254),102],n=seq.length;const tensor=x=>new ort.Tensor('int64',BigInt64Array.from(x,BigInt),[1,n]);
  const outputs=await session.run({input_ids:tensor(seq),attention_mask:tensor(seq.map(()=>1)),token_type_ids:tensor(seq.map(()=>0))});const out=outputs[session.outputNames[0]];const pooled=Array(384).fill(0);
  for(let j=0;j<n;j++)for(let k=0;k<384;k++)pooled[k]+=out.data[j*384+k]/n;
  const unit=normalise(pooled);for(let k=0;k<384;k++)vector[k]+=unit[k];chunks++;onProgress(chunks,Math.max(1,Math.ceil(ids.length/254)));
 }return {vector:normalise(vector),chunks,tokens:ids.length,characters:text.length};
}
export function predict(text,model,embedding){
 const features=sparseText(text,model),result={};for(const[name,m]of Object.entries(model.models)){
  let score=m.intercept;if(name!=='sentence')score+=textScore(features,m.coef);
  if(name!=='tfidf')score+=embedding.reduce((s,v,i)=>s+v*m.coef[(name==='combined'?model.terms.length:0)+i],0);
  result[name]={log:score,price:Math.max(0,Math.expm1(score))};
 }const q=model.meta.band80_log_residual;result.range=[Math.max(0,Math.expm1(result.combined.log-q)),Math.expm1(result.combined.log+q)];
 result.terms=features.map(([i,v])=>({term:model.terms[i],weight:v*model.models.combined.coef[i]})).sort((a,b)=>Math.abs(b.weight)-Math.abs(a.weight)).slice(0,8);return result;
}
