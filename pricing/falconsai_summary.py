"""Local Falconsai T5-small abstractive summaries; no LexRank or external inference."""
import hashlib
import re
import threading
from pricing.vehicle_narrative import narrative

MODEL_ID='Falconsai/text_summarization'
REVISION='6e505f907968c4a9360773ff57885cdc6dca4bfd'
CHUNK_TOKENS=440
OUTPUT_TOKENS=80


class SummaryEngine:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        torch.set_num_threads(min(2,torch.get_num_threads()))
        self.torch=torch
        self.tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=REVISION,use_fast=True)
        self.model=AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID,revision=REVISION,
                                                       use_safetensors=True).to('cpu').eval()
        self.lock=threading.RLock()

    def encode(self,text):
        return self.tokenizer.encode(text,add_special_tokens=False)

    def decode(self,tokens):
        return self.tokenizer.decode(tokens,skip_special_tokens=True)

    def generate(self,text):
        inputs=self.tokenizer('summarize: '+text,return_tensors='pt',truncation=False)
        if inputs.input_ids.shape[1]>512:raise ValueError('Summary input exceeds token budget')
        with self.lock,self.torch.inference_mode():
            output=self.model.generate(**inputs,max_new_tokens=OUTPUT_TOKENS,num_beams=1,
                                       do_sample=False,use_cache=True,no_repeat_ngram_size=3)
        if output[0,-1].item()!=self.tokenizer.eos_token_id:
            raise ValueError('Summary output did not finish within its token budget')
        return self.tokenizer.decode(output[0],skip_special_tokens=True).strip()


def build_story(fields,description):
    facts,text=narrative(fields,description)
    identity,separator,details=facts.partition(' Supplied details — ')
    # Keep the seller narrative beside the identity, before the specification inventory.
    return identity+(' The seller reports (unverified): '+text if text else '')+(' '+separator.strip()+' '+details if separator else '')


def summarize_story(story,engine):
    """Map/reduce every token; long descriptions are never silently truncated."""
    ids=engine.encode(story)
    if not ids:return {'summary':'','chunks':0,'status':'empty'}
    chunks=[engine.decode(ids[i:i+CHUNK_TOKENS]) for i in range(0,len(ids),CHUNK_TOKENS)]
    summaries=[engine.generate(c) for c in chunks]
    if any(not s for s in summaries):raise ValueError('Model returned an empty summary')
    for _ in range(8):
        if len(summaries)==1:break
        merged=engine.encode(' '.join(summaries))
        groups=[engine.decode(merged[i:i+CHUNK_TOKENS]) for i in range(0,len(merged),CHUNK_TOKENS)]
        summaries=[engine.generate(c) for c in groups]
    else:raise ValueError('Summary reduction did not converge')
    summary=summaries[0]
    numbers=lambda text:set(re.findall(r'\d+(?:\.\d+)?',text.replace(',','')))
    unexpected=numbers(summary)-numbers(story)
    return {'summary':summary if not unexpected else '', 'chunks':len(chunks),
            'status':'review_required' if unexpected else 'ready'}


def render_story(fields,description):
    import streamlit as st
    # Define lazily: unit tests can import orchestration without loading Streamlit/torch.
    @st.cache_resource(show_spinner=False)
    def get_engine(model_id,revision):
        return SummaryEngine()
    story=build_story(fields,description)
    fingerprint=hashlib.sha256(str((REVISION,CHUNK_TOKENS,OUTPUT_TOKENS,story)).encode()).hexdigest()
    st.markdown('#### 📖 The car’s story · AI summary')
    saved=st.session_state.get('_falconsai_summary',{})
    if saved.get('fingerprint')!=fingerprint:
        try:
            with st.spinner('Condensing the car’s story…'):
                result=summarize_story(story,get_engine(MODEL_ID,REVISION))
            saved={'fingerprint':fingerprint,'result':result}
            st.session_state['_falconsai_summary']=saved
        except Exception as error:
            import logging
            logging.getLogger(__name__).warning('Falconsai summary unavailable: %s',type(error).__name__)
            saved={'fingerprint':fingerprint,'result':{'status':'unavailable','summary':''}}
    result=saved['result']
    if result['status']=='ready':st.write(result['summary'])
    elif result['status']=='review_required':st.caption('The draft introduced an unsupported number and was withheld. Read the supplied story below.')
    else:st.caption('The summary is unavailable. Your price estimate is still ready; the supplied story is below.')
    st.caption('Falconsai/text_summarization · AI-generated summary of unverified listing information. It can omit details or misstate facts; check the original. This summary does not affect the price model.')
    with st.expander('Full supplied story · model & limitations'):
        st.text(story)
        st.caption('T5-small runs locally on CPU, using greedy decoding and a shared cached model. Long stories are summarized in token-sized chunks and merged. Results are cached in your session. No LexRank, Qwen, Kaggle job or external AI API. A numeric check catches some invented numbers; it does not establish factual accuracy.')
