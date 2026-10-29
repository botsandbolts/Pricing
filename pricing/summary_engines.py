"""Pinned local Qwen backend; prompts and orchestration live in vehicle_summary."""
import threading
from pricing.vehicle_summary import (MODEL_ID, MODEL_REVISION, GGUF_ID,
                                     GGUF_REVISION, GGUF_FILE)


class CpuSummaryEngine:
    name = 'llama.cpp/Q4_K_M'

    def __init__(self, model_path=None):
        from llama_cpp import Llama
        from huggingface_hub import hf_hub_download
        path=model_path or hf_hub_download(GGUF_ID,filename=GGUF_FILE,revision=GGUF_REVISION)
        self.model=Llama(model_path=str(path),n_ctx=4096,n_threads=2,n_batch=64,n_ubatch=32,
                         n_gpu_layers=0,use_mmap=True,verbose=False,seed=42)
        self.lock=threading.RLock()

    def tokenize(self,text):
        return self.model.tokenize(text.encode(),add_bos=False)

    def detokenize(self,tokens):
        return self.model.detokenize(tokens).decode('utf-8',errors='replace')

    def complete_batch(self,requests,max_new_tokens):
        outputs=[]
        with self.lock:
            for messages in requests:
                prompt=''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n' for m in messages)+'<|im_start|>assistant\n'
                if len(self.tokenize(prompt))+max_new_tokens>4096:
                    raise ValueError('Summary prompt exceeds the supported serving context')
                out=self.model.create_completion(prompt,max_tokens=max_new_tokens,temperature=0,
                    top_p=1,repeat_penalty=1.,seed=42,stop=['<|im_end|>','<|endoftext|>'])
                if out['choices'][0].get('finish_reason')=='length':
                    raise RuntimeError('Summary exceeded the output budget; incomplete prose withheld')
                outputs.append(out['choices'][0]['text'].strip())
        return outputs

