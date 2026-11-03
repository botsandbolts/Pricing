"""BART-base-CNN INT8 inference with ONNX Runtime and tokenizers only."""
import json,threading,re
from pathlib import Path
import numpy as np

class OnnxBartEngine:
    def __init__(self,folder):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        import tempfile,hashlib
        folder=Path(folder)
        self.config=json.loads((folder/'config.json').read_text())
        self.tokenizer=Tokenizer.from_file(str(folder/'tokenizer.json'))
        ort.disable_telemetry_events()
        options=ort.SessionOptions()
        options.intra_op_num_threads=2;options.inter_op_num_threads=1
        options.enable_cpu_mem_arena=False;options.enable_mem_pattern=False
        options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        self._files=tempfile.TemporaryDirectory(prefix='bart-cpu-')
        def graph(name):
            source=folder/name
            registry=Path(str(source)+'.parts.json')
            if not registry.exists():return str(source)
            spec=json.loads(registry.read_text());target=Path(self._files.name)/name
            digest=hashlib.sha256();size=0
            with target.open('wb') as file:
                for part in spec['parts']:
                    if Path(part['file']).name!=part['file']:raise ValueError('Invalid BART part name')
                    data=(folder/part['file']).read_bytes()
                    if len(data)!=part['bytes'] or hashlib.sha256(data).hexdigest()!=part['sha256']:
                        raise ValueError('BART part checksum mismatch')
                    file.write(data);digest.update(data);size+=len(data)
            if size!=spec['bytes'] or digest.hexdigest()!=spec['sha256']:raise ValueError('BART graph checksum mismatch')
            return str(target)
        self.encoder=ort.InferenceSession(graph('encoder_model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        self.decoder=ort.InferenceSession(graph('decoder_model_merged.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        self.lock=threading.RLock()
        self.numeric=[i for token,i in self.tokenizer.get_vocab().items() if re.search(r'\d',token)]
    def encode(self,text):
        return self.tokenizer.encode(text,add_special_tokens=False).ids
    def decode(self,ids):
        return self.tokenizer.decode(list(ids),skip_special_tokens=True)
    def generate(self,text,forbid_numbers=False):
        ids=np.array([self.tokenizer.encode(text).ids],dtype=np.int64)
        if ids.shape[1]>1024:raise ValueError('BART input exceeds encoder budget')
        mask=np.ones_like(ids)
        with self.lock:
            encoder_values={'input_ids':ids,'attention_mask':mask}
            enc=self.encoder.run(None,{i.name:encoder_values[i.name] for i in self.encoder.get_inputs()})[0]
            config=self.config
            heads=config['decoder_attention_heads'];width=config['d_model']//heads
            cache={i.name:np.zeros((1,heads,0,width),dtype=np.float32) for i in self.decoder.get_inputs() if i.name.startswith('past_key_values')}
            sequence=[config.get('decoder_start_token_id',config['eos_token_id'])]
            eos=config['eos_token_id'];finished=False
            for step in range(160):
                known={'input_ids':np.array([[sequence[-1]]],dtype=np.int64),
                       'decoder_input_ids':np.array([[sequence[-1]]],dtype=np.int64),
                       'encoder_hidden_states':enc,'encoder_attention_mask':mask,
                       'attention_mask':mask,'use_cache_branch':np.array([step>0],dtype=np.bool_),
                       **cache}
                feed={i.name:known[i.name] for i in self.decoder.get_inputs()}
                output=self.decoder.run(None,feed)
                values=dict(zip([o.name for o in self.decoder.get_outputs()],output))
                scores=values['logits'][0,-1].copy()
                if forbid_numbers:scores[self.numeric]=-np.inf
                if len(sequence)>=2:
                    prefix=sequence[-2:]
                    for i in range(len(sequence)-2):
                        if sequence[i:i+2]==prefix:scores[sequence[i+2]]=-np.inf
                token=int(config['forced_bos_token_id']) if step==0 and config.get('forced_bos_token_id') is not None else int(np.argmax(scores));sequence.append(token)
                if token==eos:finished=True;break
                for name,value in values.items():
                    key=name.replace('present','past_key_values',1)
                    if key in cache and value.size:cache[key]=value
        result=self.decode(sequence).strip()
        if not finished:
            # Keep complete prose instead of rejecting a useful length-capped draft.
            matches=list(re.finditer(r'[.!?](?:\s|$)',result))
            if matches:result=result[:matches[-1].end()].strip()
            else:raise ValueError('BART produced no complete sentence')
        if not result:raise ValueError('BART produced an empty summary')
        return result
