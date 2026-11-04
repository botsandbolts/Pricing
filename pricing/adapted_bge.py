"""One fixed, locally adapted BGE encoder for offline and GUI inference."""
import json,hashlib,tempfile
from pathlib import Path
import numpy as np
from pricing.advert_text import scrub
from pricing.suite_storage import read_model_file

class AdaptedEncoder:
    def __init__(self,folder):
        import torch
        from transformers import AutoTokenizer,AutoModel
        folder=Path(folder)
        # Rehydrate split Git blobs into a release-specific local cache.
        manifest=json.loads((folder/'bundle.json').read_text())
        dest=Path(tempfile.gettempdir())/'pricing-adapted-bge'/manifest['identity']
        dest.mkdir(parents=True,exist_ok=True)
        for name,digest in manifest['files'].items():
            if Path(name).name!=name:raise ValueError('Invalid checkpoint file')
            path=dest/name
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                data=read_model_file(folder/name)
                if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Checkpoint checksum mismatch')
                path.write_bytes(data)
        self.tokenizer=AutoTokenizer.from_pretrained(dest,local_files_only=True)
        self.model=AutoModel.from_pretrained(dest,local_files_only=True).to('cpu').eval()
        self.max_tokens=256
        torch.set_num_threads(2)
    def encode(self,texts,batch_size=32):
        import torch
        if not texts:return np.empty((0,384),dtype='float32')
        output=[]
        with torch.inference_mode():
            for start in range(0,len(texts),batch_size):
                batch=self.tokenizer([scrub(t) for t in texts[start:start+batch_size]],
                    padding=True,truncation=True,max_length=self.max_tokens,return_tensors='pt')
                hidden=self.model(**batch).last_hidden_state[:,0,:]
                output.append(torch.nn.functional.normalize(hidden,p=2,dim=1).cpu().numpy())
        return np.concatenate(output).astype('float32')

class AdaptedProjector:
    def __init__(self,folder):
        import io,joblib
        from pricing.description_features import MiniLMPriceFeatures
        self.mini=MiniLMPriceFeatures(folder)
        self.encoder=AdaptedEncoder(Path(folder)/'adapted-bge')
        self.reducer=joblib.load(io.BytesIO(read_model_file(Path(folder)/'calibration-refit-bge-umap.joblib')))
        spec=json.loads((Path(folder)/'features.json').read_text())
        if spec['encoder_bundle']['identity']!=json.loads((Path(folder)/'adapted-bge/bundle.json').read_text())['identity']:raise ValueError('Wrong serving encoder')
        if self.reducer.n_components!=spec['umap_dimensions'] or self.reducer.n_epochs!=500 or self.reducer.metric!='cosine':raise ValueError('Wrong serving UMAP')
        self.cache={}
    def transform(self,texts):
        from scipy.sparse import csr_matrix
        cleaned=[scrub(t) for t in texts]
        missing=list(dict.fromkeys(t for t in cleaned if t not in self.cache))
        if missing:
            mini=self.mini.transform(missing).toarray()
            reduced=self.reducer.transform(self.encoder.encode(missing))
            for text,row in zip(missing,np.column_stack([mini,reduced]).astype('float32')):self.cache[text]=row
        values=np.asarray([self.cache[t] for t in cleaned],dtype='float32').reshape(len(cleaned),1+self.reducer.n_components)
        if len(self.cache)>512:self.cache.clear()
        if not np.isfinite(values).all():raise ValueError('Invalid description features')
        n,k=values.shape
        return csr_matrix((values.ravel(),np.tile(np.arange(k),n),np.arange(0,(n+1)*k,k)),shape=(n,k))
    def get_feature_names_out(self):
        return np.array(['MiniLM description price']+[f'bge_umap_{j}' for j in range(self.reducer.n_components)])
