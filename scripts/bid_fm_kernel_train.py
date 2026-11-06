"""Scalable FM, field-aware FM, and random Fourier SVR-like auction challengers.

NO price-derived inputs, invented censoring, seller descriptions, or
validation/test row filtering. All algorithms train exclusively on original
training auctions; internal chronological stopper is training-only.
External VIN-disjoint validation chooses models. Test access reserved for
the independent report of the validation-selected winner.
"""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Subset

from scripts.neural_bid_prepare import CATS, EXPECTED, EXPECTED_LEMON, sha_ids

TRIALS = {
    "fm16_full": ("fm", 16, "full"),
    "fm32_full": ("fm", 32, "full"),
    "fm16_lemon": ("fm", 16, "lemon"),
    "fm32_lemon": ("fm", 32, "lemon"),
    "ffm8_full": ("ffm", 8, "full"),
    "ffm8_lemon": ("ffm", 8, "lemon"),
    "rff512_huber_full": ("rff_huber", 512, "full"),
    "rff512_epsilon_full": ("rff_epsilon", 512, "full"),
    "rff512_huber_lemon": ("rff_huber", 512, "lemon"),
    "rff1024_epsilon_lemon": ("rff_epsilon", 1024, "lemon"),
}
NUMERIC_COUNT=5
FIELD_COUNT=NUMERIC_COUNT+len(CATS)
SEED=20261106
MAX_FM_EPOCHS=22
MAX_FFM_EPOCHS=16
STOPPING_PATIENCE=4


def source(path):
    path=Path(path)
    m=json.loads(path.with_suffix(".json").read_text())
    if (m.get("market"),m.get("no_description"),m.get("no_price_derived_inputs"),
        m.get("no_hierarchy"),m.get("full_train"),m.get("lemon_train"),
        m.get("validation"),m.get("test")) != (
        "bid",True,True,True,384198,288148,82380,83603):
        raise ValueError("Source is not the original frozen full/Lemon auction dataset")
    with np.load(path,allow_pickle=False) as f:
        z={k:f[k] for k in ("num","cat","y","phase","ids","lemon")}
    ids=z["ids"].astype(str)
    phase=z["phase"].astype(str)
    if len(ids)!=550181 or len(np.unique(ids))!=len(ids):
        raise ValueError("Original auction identities missing or duplicated")
    if z["num"].shape!=(550181,5) or z["cat"].shape!=(550181,10):
        raise ValueError("Exactly five original numerical and ten categorical facts required")
    if not np.isfinite(z["num"]).all() or not np.isfinite(z["y"]).all() or (z["y"]<=0).any():
        raise ValueError("Invalid original auction facts or sale prices")
    if not (z["cat"]>=0).all():
        raise ValueError("Negative categorical code in frozen source")
    if np.any((np.max(z["cat"],axis=0)+1) > np.asarray(m["cardinalities"],dtype=int)):
        raise ValueError("Category index outside frozen dictionary")
    for p,count in EXPECTED.items():
        if int(sum(phase==p))!=count:
            raise ValueError("Original auction partition changed: "+p)
    for p in ("validation","test"):
        if sha_ids(ids[phase==p])!=m["holdout_row_sha256"][p]:
            raise ValueError("Original holdout row fingerprints changed")
    if int((z["lemon"] & (phase=="train")).sum()) != EXPECTED_LEMON:
        raise ValueError("Frozen Lemon training membership changed")
    if not z["lemon"][phase!="train"].all():
        raise ValueError("Original holdouts have been trimmed")
    return z,m


def partition(z,cohort):
    phase=z["phase"].astype(str)
    train=np.flatnonzero((phase=="train") & (z["lemon"] if cohort=="lemon" else True))
    val=np.flatnonzero(phase=="validation")
    test=np.flatnonzero(phase=="test")
    expected=EXPECTED_LEMON if cohort=="lemon" else EXPECTED["train"]
    if len(train)!=expected or len(val)!=82380 or len(test)!=83603:
        raise ValueError("Original auction train/holdout membership differs")
    stopper=max(512,int(len(train)*.1))
    fit,inner=train[:-stopper],train[-stopper:]
    if min(len(fit),len(inner))<=0 or np.intersect1d(fit,inner).size:
        raise ValueError("Missing or overlapping training/early stopping rows")
    return fit,inner,val,test


def _spline(x):
    """Hinge spline MAIN effects: explicitly model depreciation nonlinearity."""
    knots=torch.tensor([-3.,-2.,-1.,0.,1.,2.],device=x.device,dtype=x.dtype)
    return torch.relu(x[:,:,None]-knots[None,None,:])


class AuctionFactorization(nn.Module):
    """Second-order FM or true field-aware FM, with piecewise numeric main effects.

    For FFM a categorical feature's embedding depends on which target FIELD
    it interacts with. 15 fields are small enough for the exact 105 pairs.
    """
    def __init__(self,sizes,rank,field_aware=False,initial_log_price=9.):
        super().__init__()
        self.field_aware=bool(field_aware)
        self.rank=rank
        self.base=nn.Parameter(torch.tensor(float(initial_log_price)))
        self.num_linear=nn.Parameter(torch.zeros(NUMERIC_COUNT))
        self.num_spline=nn.Parameter(torch.zeros(NUMERIC_COUNT,6))
        self.cat_linear=nn.ModuleList([nn.Embedding(int(size),1,padding_idx=0)
                                       for size in sizes])
        latent_dim=rank*(FIELD_COUNT if field_aware else 1)
        self.cat_latent=nn.ModuleList([nn.Embedding(int(size),latent_dim,padding_idx=0)
                                       for size in sizes])
        self.num_latent=nn.Parameter(torch.randn(NUMERIC_COUNT,latent_dim)*.01)
        for table in self.cat_linear:
            nn.init.zeros_(table.weight)
        for table in self.cat_latent:
            nn.init.normal_(table.weight,mean=0,std=.01)
            with torch.no_grad():
                table.weight[0].zero_()

    def forward(self,num,cat):
        if num.ndim!=2 or cat.ndim!=2 or num.shape[1]!=5 or cat.shape[1]!=10:
            raise ValueError("Wrong five-numeric ten-categorical auction feature geometry")
        bias=self.base + num@self.num_linear
        bias=bias+(_spline(num)*self.num_spline).sum((1,2))
        for j,table in enumerate(self.cat_linear):
            bias=bias+table(cat[:,j]).squeeze(-1)
        if self.field_aware:
            # (B, feature_field, target_field, rank)
            n=(num.unsqueeze(-1)*self.num_latent[None,:,:]).reshape(
                len(num),NUMERIC_COUNT,FIELD_COUNT,self.rank)
            c=torch.stack([
                table(cat[:,j]).reshape(len(num),FIELD_COUNT,self.rank)
                for j,table in enumerate(self.cat_latent)],dim=1)
            features=torch.cat([n,c],dim=1)
            pairs=[]
            for i in range(FIELD_COUNT):
                for j in range(i+1,FIELD_COUNT):
                    pairs.append((features[:,i,j,:]*features[:,j,i,:]).sum(1))
            interaction=torch.stack(pairs,1).sum(1)
        else:
            n=num[:,:,None]*self.num_latent[None,:,:]
            c=torch.stack([table(cat[:,j]) for j,table in enumerate(self.cat_latent)],dim=1)
            latent=torch.cat((n,c),dim=1)
            interaction=.5*((latent.sum(1)**2).sum(1)-(latent**2).sum((1,2)))
        return bias+interaction


def y_price(mu):
    return np.exp(np.clip(np.asarray(mu,dtype=np.float64),np.log(100.),np.log(1000000.)))


@torch.no_grad()
def fm_prediction(model,nums,cats,rows,batch=8192):
    model.eval()
    output=[]
    for begin in range(0,len(rows),batch):
        at=rows[begin:begin+batch]
        p=model(torch.from_numpy(nums[at].astype(np.float32)),
                torch.from_numpy(cats[at].astype(np.int64)))
        output.append(p.detach().cpu().numpy())
    return y_price(np.concatenate(output))


def _scores(actual,pred):
    actual=np.asarray(actual,float);pred=np.asarray(pred,float)
    if not np.isfinite(pred).all() or (pred<=0).any():
        raise ValueError("Invalid price predictions")
    error=pred-actual
    return dict(n=int(len(actual)),mae_usd=float(np.abs(error).mean()),
                rmse_usd=float(np.sqrt(np.square(error).mean())),
                bias_usd=float(error.mean()),
                median_ape_percent=float(np.median(np.abs(error)/actual)*100))


def fit_fm(name,z,manifest,fit,inner,val,out,seconds):
    family,rank,cohort=TRIALS[name]
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    nums=z["num"].astype(np.float32)
    cats=z["cat"].astype(np.int64)
    y=z["y"].astype(np.float64)
    init=float(np.median(np.log(y[fit])))
    model=AuctionFactorization(manifest["cardinalities"],rank,
                               field_aware=(family=="ffm"),initial_log_price=init)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003 if family=="fm" else .0015,
                                weight_decay=.003)
    trainset=TensorDataset(torch.from_numpy(nums[fit]),
                           torch.from_numpy(cats[fit]),
                           torch.from_numpy(np.log(y[fit]).astype(np.float32)))
    generator=torch.Generator().manual_seed(SEED)
    loader=DataLoader(trainset,batch_size=2048 if family=="fm" else 1024,
                      shuffle=True,generator=generator,num_workers=0)
    best=1e15;best_model=None;best_epoch=0;stale=0
    run_start=time.monotonic()
    epochs=MAX_FM_EPOCHS if family=="fm" else MAX_FFM_EPOCHS
    for epoch in range(1,epochs+1):
        model.train()
        for num,cat,target in loader:
            optimizer.zero_grad(set_to_none=True)
            pred=model(num,cat)
            loss=nn.functional.smooth_l1_loss(pred,target,beta=.25)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite factorization loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
        inner_pred=fm_prediction(model,nums,cats,inner)
        score=_scores(y[inner],inner_pred)["mae_usd"]
        print(json.dumps({"trial":name,"epoch":epoch,"stopper_mae":score,
                          "elapsed_seconds":round(time.monotonic()-run_start,1)}),flush=True)
        if score<best-.5:
            best=score;best_model=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
        else:
            stale+=1
        if stale>=STOPPING_PATIENCE or time.monotonic()-run_start>seconds:
            break
    if best_model is None:
        raise ValueError("No successful factorization checkpoint")
    model.load_state_dict(best_model)
    valid=fm_prediction(model,nums,cats,val)
    torch.save(dict(state_dict=best_model,trial=name,rank=rank,
                    field_aware=family=="ffm",cards=manifest["cardinalities"],
                    initial_log_price=init),
               out/"weights.pt")
    return valid,dict(best_epoch=best_epoch,inner_mae_usd=best,
                      architecture="field-aware FM" if family=="ffm" else "2nd-order FM",
                      training_time_seconds=time.monotonic()-run_start)


def rff_features(num,cat,encoder,scaler,fourier,indices):
    import scipy.sparse as sp
    # Category ID 1 is explicit unknown; original dictionaries were train-fit.
    scaled=np.clip(scaler.transform(num[indices]),-5,5).astype(np.float32)/np.sqrt(5)
    hot=encoder.transform(cat[indices])
    hot=hot.astype(np.float32)/np.sqrt(10)
    combined=sp.hstack([sp.csr_matrix(scaled),hot],format="csr",dtype=np.float32)
    transformed=fourier.transform(combined)
    return np.asarray(transformed,dtype=np.float32)


def fit_rff(name,z,manifest,fit,inner,val,out,seconds):
    from sklearn.preprocessing import OneHotEncoder,StandardScaler
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.linear_model import SGDRegressor
    family,n_components,cohort=TRIALS[name]
    nums=z["num"].astype(np.float32)
    cat=z["cat"].astype(np.int64)
    y=z["y"].astype(np.float64)
    scaler=StandardScaler().fit(nums[fit])
    ohe=OneHotEncoder(sparse_output=True,handle_unknown="ignore",
                      dtype=np.float32).fit(cat[fit])
    import scipy.sparse as sp
    example=sp.hstack([
        sp.csr_matrix(np.clip(scaler.transform(nums[fit[:2048]]),-5,5).astype(np.float32)/np.sqrt(5)),
        ohe.transform(cat[fit[:2048]]).astype(np.float32)/np.sqrt(10)],format="csr")
    transform=RBFSampler(gamma=.5,n_components=n_components,random_state=SEED)
    transform.fit(example)
    print(json.dumps({"trial":name,"rff_features":n_components,
                      "one_hot_features":example.shape[1]}),flush=True)
    t0=time.monotonic()
    Xfit=rff_features(nums,cat,ohe,scaler,transform,fit)
    Xinner=rff_features(nums,cat,ohe,scaler,transform,inner)
    Xval=rff_features(nums,cat,ohe,scaler,transform,val)
    center=float(np.median(np.log(y[fit])))
    estimator=SGDRegressor(loss="huber" if family=="rff_huber" else "epsilon_insensitive",
                           epsilon=.15 if family=="rff_huber" else .10,
                           alpha=.0001,eta0=.01,learning_rate="invscaling",
                           average=True,random_state=SEED,
                           fit_intercept=True)
    target=(np.log(y[fit])-center).astype(np.float64)
    best=1e15;best_est=None;best_epoch=0;stale=0
    rng=np.random.default_rng(SEED)
    for epoch in range(1,41):
        order=rng.permutation(len(fit))
        for at in np.array_split(order,max(1,int(np.ceil(len(fit)/8192)))):
            estimator.partial_fit(Xfit[at],target[at])
        inner_predictions=y_price(estimator.predict(Xinner)+center)
        score=_scores(y[inner],inner_predictions)["mae_usd"]
        print(json.dumps({"trial":name,"epoch":epoch,"stopper_mae":score,
                          "elapsed_seconds":round(time.monotonic()-t0,1)}),flush=True)
        if score<best-.5:
            best=score;best_est=copy.deepcopy(estimator);best_epoch=epoch;stale=0
        else:
            stale+=1
        if stale>=6 or time.monotonic()-t0>seconds:break
    if best_est is None:
        raise ValueError("No completed RFF stochastic SVR-like fit")
    pred=y_price(best_est.predict(Xval)+center)
    joblib.dump(dict(trial=name,scaler=scaler,onehot=ohe,rbf=transform,
                     estimator=best_est,target_log_center=center,
                     model="RFF + epsilon-SVR" if family=="rff_epsilon" else "RFF + Huber"),
                out/"model.joblib",compress=3)
    return pred,dict(best_epoch=best_epoch,inner_mae_usd=best,
                     architecture="RFF approximate SVR" if family=="rff_epsilon" else "RFF robust regression",
                     training_time_seconds=time.monotonic()-t0,
                     random_fourier_features=n_components)


def run(name,src,out,max_seconds=4600):
    if name not in TRIALS:
        raise ValueError("Unregistered factorization/kernel trial")
    family,rank,cohort=TRIALS[name]
    z,m=source(src)
    fit,inner,val,_test=partition(z,cohort)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    if family in ("fm","ffm"):
        predictions,metadata=fit_fm(name,z,m,fit,inner,val,out,max_seconds)
    else:
        predictions,metadata=fit_rff(name,z,m,fit,inner,val,out,max_seconds)
    original=z["y"][val].astype(float)
    scores=_scores(original,predictions)
    pd.DataFrame(dict(row_id=z["ids"][val].astype(str),
                      split="validation",observed_price=original,
                      predicted_price=predictions)).to_csv(out/"validation.csv.gz",index=False)
    report={
        "status":"completed","market":"bid","trial":name,
        "model_family":family,"rank_or_components":rank,"cohort":cohort,
        "split":"original VIN-disjoint chronological validation",
        "original_train_rows":384198,"lemon_train_rows":288148,
        "fitting_rows":len(fit),"stopper_rows":len(inner),
        "validation_rows":len(val),"test_rows":83603,
        "test_evaluated":False,
        "text_features":False,"price_derived_inputs":False,
        "no_hierarchy":True,
        "target":"Observed wholesale auction sales in historical 2015-normalized USD",
        "loss":"Huber on log USD" if family in ("fm","ffm","rff_huber")
               else "epsilon-insensitive on log USD",
        "holdout_row_sha256":m["holdout_row_sha256"],
        "canonical_cohort_sha256":m["cohort_sha256"],
        "original_feature_cardinalities":m["cardinalities"],
        "validation":scores,
        "wall_seconds":time.monotonic()-started,
        "selection":"Validation MAE only; test not predicted by training",
        **metadata}
    (out/"result.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    print("BID_CHALLENGER_VALIDATION",name,scores["mae_usd"],flush=True)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trial",required=True,choices=tuple(TRIALS))
    p.add_argument("--features",default="training-cache/neural-bid/features.npz")
    p.add_argument("--output",required=True)
    p.add_argument("--max-seconds",type=float,default=4600)
    a=p.parse_args()
    run(a.trial,a.features,a.output,a.max_seconds)
