"""Kaggle GPU alternative; leaves the live TF-IDF release unchanged."""
from pathlib import Path
import runpy,torch
assert torch.cuda.is_available(),'Enable a Kaggle GPU accelerator before running.'
from transformers import pipeline
from types import SimpleNamespace
import spacy
ner=pipeline('token-classification',model='dslim/bert-base-NER',aggregation_strategy='simple',device=0)
class GPUEntities:
    def __call__(self,text):
        values=ner(text)
        return SimpleNamespace(ents=[SimpleNamespace(text=v['word'],label_=v['entity_group']) for v in values])
spacy.load=lambda *args,**kwargs:GPUEntities()
source=Path('scripts/build_merchant_reco.py').read_text()
start=source.index('    vectorizer=TfidfVectorizer')
end=source.index('    # Shared feature/type',start)
gpu_block="""    from sentence_transformers import SentenceTransformer
    encoder=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cuda')
    encoder.max_seq_length=256
    embeddings=encoder.encode(clean,batch_size=128,normalize_embeddings=True,show_progress_bar=True).astype(np.float32)
    projector=TruncatedSVD(n_components=96,random_state=42)
    semantic=normalize(projector.fit_transform(embeddings)).astype(np.float16)
    import joblib
    joblib.dump(projector,Path('training-cache')/'merchant-minilm-projector.joblib',compress=3)
    del embeddings,encoder,nlp
    torch.cuda.empty_cache()
"""
source=source[:start]+gpu_block+source[end:]
source=source.replace("'semantic_model':'TF-IDF word unigrams/bigrams + up to 96D truncated SVD'","'semantic_model':'GPU all-MiniLM-L6-v2 + 96D projection'")
# No automatic publication: Kaggle writes an independently reviewed artifact.
exec(compile(source,'merchant-gpu-build','exec'),{'__name__':'__main__','torch':torch})
import shutil
shutil.make_archive('/kaggle/working/merchant-minilm-gpu','zip','data/merchant-discovery')
print('GPU alternative saved: /kaggle/working/merchant-minilm-gpu.zip. Not deployed automatically.')
