import * as ort from './vendor/ort.wasm.min.mjs';
import {clean,embed,predict} from './engine.mjs';
let model,session;
self.onmessage=async({data})=>{
 try{
  if(!model)model=await(await fetch('/pricing-model.json')).json();
  if(!session){self.postMessage({type:'progress',message:'Loading the BERT-family encoder (22 MB, once per visit)…'});ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;ort.env.wasm.wasmPaths=new URL('./vendor/',self.location.href).href;session=await ort.InferenceSession.create('/encoder/onnx/model_quantized.onnx',{executionProviders:['wasm'],graphOptimizationLevel:'all'});}
  const text=clean(data.text);if(text.length>8000)throw Error('The combined description exceeds the 8,000-character model limit. Please shorten your notes.');
  const e=await embed(text,model,session,ort,(n,total)=>self.postMessage({type:'progress',message:`Reading the whole story: chunk ${n} of ${total}…`}));
  const prediction=predict(text,model,e.vector);self.postMessage({type:'result',prediction,context:{characters:text.length,tokens:e.tokens,chunks:e.chunks},requestId:data.requestId});
 }catch(e){self.postMessage({type:'error',message:e.message||String(e)});}
};
