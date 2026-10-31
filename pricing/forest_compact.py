"""Low-memory interpreter for exported XGBoost forest trees (no retraining)."""
import numpy as np


def predict(model,frame):
    x=np.column_stack([frame[c].cat.codes.replace(-1,np.nan).to_numpy(float) if hasattr(frame[c].dtype,'categories') else frame[c].to_numpy(float) for c in frame]).astype(np.float32)
    output=np.full(len(x),model['base'][0],np.float32)
    for tree in range(len(model['starts'])-1):
        start=int(model['starts'][tree]);node=np.full(len(x),start,dtype=int)
        active=np.ones(len(x),bool)
        while active.any():
            rows=np.flatnonzero(active);nodes=node[rows];features=model['feature'][nodes];leaf=features<0
            if leaf.any():
                output[rows[leaf]]+=model['value'][nodes[leaf]];active[rows[leaf]]=False
            rows=rows[~leaf];nodes=nodes[~leaf];features=features[~leaf]
            if not len(rows):break
            values=x[rows,features];missing=np.isnan(values);yes=values<model['value'][nodes]
            categorical=model['cat_length'][nodes]>0
            for i in np.flatnonzero(categorical&~missing):
                code=int(values[i]);offset=int(model['cat_offset'][nodes[i]]);length=int(model['cat_length'][nodes[i]])
                yes[i]=code>=0 and code//8<length and bool(int(model['bits'][offset+code//8])&(1<<(code%8)))
            node[rows]=np.where(missing,model['missing'][nodes],np.where(yes,model['yes'][nodes],model['no'][nodes]))
    return output


class CompactForest:
    def __init__(self,arrays):self.arrays=arrays
    def predict(self,frame):return predict(self.arrays,frame)
