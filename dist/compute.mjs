import * as ort from './vendor/ort.wasm.min.mjs';
import {clean,createTokenizer,embed,predict} from './engine.mjs';
let model,session,tokenizer;
async function loadEncoder(meta){
 const url=`https://huggingface.co/${meta.encoder_model_id}/resolve/${meta.encoder_revision}/onnx/model_quantized.onnx`;
 self.postMessage({type:'progress',message:'Downloading XLM-RoBERTa (279 MB). This first load may take a few minutes…'});
 const response=await fetch(url,{cache:'force-cache',credentials:'omit'});if(!response.ok)throw Error('XLM-RoBERTa download failed ('+response.status+'). Please check your connection.');
 const reader=response.body.getReader(),parts=[];let received=0,last=0;
 while(true){const{done,value}=await reader.read();if(done)break;parts.push(value);received+=value.length;if(received-last>2000000){self.postMessage({type:'progress',message:`Downloading XLM-RoBERTa: ${Math.round(received/1000000)} / 279 MB…`});last=received;}}
 if(received!==meta.encoder_bytes)throw Error('The XLM-RoBERTa download was incomplete. Try again.');
 const bytes=new Uint8Array(received);let offset=0;for(const p of parts){bytes.set(p,offset);offset+=p.length;}parts.length=0;
 self.postMessage({type:'progress',message:'Starting the XLM-RoBERTa engine…'});
 ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;ort.env.wasm.wasmPaths=new URL('./vendor/',self.location.href).href;
 return ort.InferenceSession.create(bytes,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
}
self.onmessage=async({data})=>{
 try{
  if(!model)model=await(await fetch('/pricing-model.json')).json();
  if(!tokenizer){const[a,b]=await Promise.all([fetch('/encoder/tokenizer.json').then(r=>r.json()),fetch('/encoder/tokenizer_config.json').then(r=>r.json())]);tokenizer=createTokenizer(a,b);}
  if(!session)session=await loadEncoder(model.meta);
  const text=clean(data.text);if(text.length>8000)throw Error('The combined description exceeds the 8,000-character limit. Please shorten your notes.');
  const e=await embed(text,model,session,ort,tokenizer,(n,total)=>self.postMessage({type:'progress',message:`XLM-RoBERTa is reading your whole description: chunk ${n} of ${total}…`}));
  const prediction=predict(text,model,e.vector);self.postMessage({type:'result',prediction,context:{characters:text.length,tokens:e.tokens,chunks:e.chunks},requestId:data.requestId});
 }catch(e){self.postMessage({type:'error',message:e.message||String(e)});}
};
