import * as ort from './vendor/ort.wasm.min.mjs';
import {clean,createTokenizer,encode,predict} from './engine.mjs';
let model,longTokenizer,longSession;
async function download(cfg,label){
 const url=`https://huggingface.co/${cfg.onnx_model_id}/resolve/${cfg.revision}/onnx/model_quantized.onnx`;self.postMessage({type:'progress',message:`Loading ${label} (${Math.round(cfg.bytes/1000000)} MB on first use)…`});const response=await fetch(url,{cache:'force-cache',credentials:'omit'});if(!response.ok)throw Error(`${label} download failed: HTTP ${response.status}`);const reader=response.body.getReader(),parts=[];let size=0,last=0;while(true){const{done,value}=await reader.read();if(done)break;parts.push(value);size+=value.length;if(size-last>5000000){self.postMessage({type:'progress',message:`Downloading ${label}: ${Math.round(size/1000000)} / ${Math.round(cfg.bytes/1000000)} MB…`});last=size;}}if(size!==cfg.bytes)throw Error(`Incomplete ${label} download. Please retry.`);const bytes=new Uint8Array(size);let offset=0;for(const p of parts){bytes.set(p,offset);offset+=p.length;}parts.length=0;self.postMessage({type:'progress',message:`Starting ${label}…`});return ort.InferenceSession.create(bytes,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
}
async function tokenizer(dir){const[a,b]=await Promise.all([fetch(`/${dir}/tokenizer.json`).then(r=>r.json()),fetch(`/${dir}/tokenizer_config.json`).then(r=>r.json())]);return createTokenizer(a,b);}
self.onmessage=async({data})=>{try{
 if(!model)model=await(await fetch('/pricing-model.json')).json();const meta=model.meta;
 ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;ort.env.wasm.wasmPaths=new URL('./vendor/',self.location.href).href;
 if(!longTokenizer)longTokenizer=await tokenizer('encoder');
 if(!longSession)longSession=await download(meta.description_encoder,'E5-large');
 const text=clean(data.text);if(text.length>meta.max_characters)throw Error('The combined description exceeds 8,000 characters. Please shorten your notes.');
 const long=await encode(text,meta.description_encoder,longTokenizer,longSession,ort,true,(n,total)=>self.postMessage({type:'progress',message:`E5-large is reading description window ${n} of ${total} (up to 512 tokens each)…`}));
 const prediction=predict(long.vector,model);self.postMessage({type:'result',prediction,context:{characters:text.length,tokens:long.tokens,chunks:long.chunks,dimensions:1024}});
}catch(e){self.postMessage({type:'error',message:e.message||String(e)});}};
