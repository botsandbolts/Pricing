"""29 prespecified FT-Transformer Huber tuning runs on frozen Lemon bid cohort.

Individual jobs compute ONLY original validation predictions. Test is evaluated
once by the separate report after all candidates are ranked on validation.
"""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from torch.utils.data import DataLoader,Subset,TensorDataset
from scripts.neural_ask_train import FTTransformer,predict,metrics,seed_everything

BASE=dict(d=48,layers=2,heads=4,ff=4,drop=.18,head_drop=.20,head_width=96,
          prenorm=True,token_drop=0.,jitter=0.,lr=.00065,wd=.025,
          batch=768,epochs=35,patience=7,huber=.25)
TRIALS={
 "control":{},"width32":{"d":32},"width64":{"d":64},"width96":{"d":96},
 "depth1":{"layers":1},"depth3":{"layers":3},"depth4":{"layers":4},
 "heads2":{"heads":2},"heads8":{"heads":8},"ff2":{"ff":2},"ff6":{"ff":6},
 "drop08":{"drop":.08,"head_drop":.12},
 "drop30":{"drop":.30,"head_drop":.32},
 "wd005":{"wd":.005},"wd075":{"wd":.075},
 "lr035":{"lr":.00035},"lr100":{"lr":.001},
 "batch384":{"batch":384},"batch1536":{"batch":1536},
 "token05":{"token_drop":.05},"token12":{"token_drop":.12},
 "numericnoise":{"jitter":.012},"postnorm":{"prenorm":False},
 "headwide":{"head_width":160},"huber015":{"huber":.15},
 "huber040":{"huber":.40},
 "wide64depth3":{"d":64,"layers":3,"drop":.22},
 "depth3token":{"layers":3,"token_drop":.05},
 "wide64wd075":{"d":64,"wd":.075},
}
assert len(TRIALS)==29

def settings(name):
    if name not in TRIALS:raise ValueError("Unknown registered tuning arm")
    c=dict(BASE);c.update(TRIALS[name])
    if c["d"]%c["heads"] or not 0<=c["token_drop"]<1 or c["huber"]<=0:
        raise ValueError("Invalid transformer dimensions / hyperparameters")
    return c

class TunedFT(nn.Module):
    def __init__(self,cards,n_num,n_text,c):
        super().__init__();d=c["d"]
        self.embeddings=nn.ModuleList([nn.Embedding(int(s),d,padding_idx=0) for s in cards])
        self.num_weight=nn.Parameter(torch.empty(n_num,d))
        self.num_bias=nn.Parameter(torch.zeros(n_num,d))
        nn.init.normal_(self.num_weight,std=.05)
        self.text=nn.Sequential(nn.Linear(n_text,d),nn.LayerNorm(d))
        self.cls=nn.Parameter(torch.zeros(1,1,d))
        layer=nn.TransformerEncoderLayer(d_model=d,nhead=c["heads"],
             dim_feedforward=c["ff"]*d,dropout=c["drop"],
             activation="gelu",batch_first=True,norm_first=c["prenorm"])
        self.encoder=nn.TransformerEncoder(layer,num_layers=c["layers"],enable_nested_tensor=False)
        self.norm=nn.LayerNorm(d)
        self.out=nn.Sequential(nn.Linear(d,c["head_width"]),nn.GELU(),
               nn.Dropout(c["head_drop"]),nn.Linear(c["head_width"],2))
        self.token_drop=c["token_drop"]

    def forward(self,n,c,t):
        tokens=[e(c[:,j]).unsqueeze(1) for j,e in enumerate(self.embeddings)]
        tokens.extend((n.unsqueeze(-1)*self.num_weight+self.num_bias,
                       self.text(t).unsqueeze(1)))
        x=torch.cat(tokens,dim=1)
        if self.training and self.token_drop:
            mask=(torch.rand(x.shape[:2],device=x.device)>=self.token_drop)
            x=x*mask.unsqueeze(-1).to(x.dtype)/(1-self.token_drop)
        x=torch.cat((self.cls.expand(n.shape[0],-1,-1),x),dim=1)
        return self.out(self.norm(self.encoder(x)[:,0,:]))

def model_for(name,cards,n_num,n_text):
    return FTTransformer(cards,n_num,n_text) if name=="control" else TunedFT(
        cards,n_num,n_text,settings(name))

def ids_sha(ids):
    return hashlib.sha256("\n".join(sorted(map(str,ids))).encode()).hexdigest()

def frozen(source):
    source=Path(source);m=json.loads(source.with_suffix(".json").read_text())
    expected=("bid",384198,288148,82380,83603,True)
    actual=tuple(m.get(k) for k in ("market","full_train","lemon_train","validation","test","no_description"))
    if actual!=expected or not m.get("cohort_sha256") or m.get("no_price_derived_inputs") is not True:
        raise ValueError("Frozen auction data provenance mismatch")
    with np.load(source,allow_pickle=False) as f:
        z={key:f[key] for key in ("num","cat","txt","y","phase","ids","lemon")}
    phase=z["phase"].astype(str);ids=z["ids"].astype(str)
    if len(ids)!=550181 or len(set(ids))!=len(ids) or not np.isfinite(z["y"]).all():
        raise ValueError("Original auction row identity/target validation failed")
    for split in ("validation","test"):
        if ids_sha(ids[phase==split])!=m["holdout_row_sha256"][split]:
            raise ValueError("Changed original holdout identities")
    train=np.flatnonzero((phase=="train")&z["lemon"].astype(bool))
    val=np.flatnonzero(phase=="validation");test=np.flatnonzero(phase=="test")
    if (len(train),len(val),len(test))!=(288148,82380,83603):
        raise ValueError("Unexpected original Lemon/validation/test sizes")
    stopper=max(512,int(len(train)*.1))
    fit,inner=train[:-stopper],train[-stopper:]
    if (len(fit),len(inner))!=(259334,28814):
        raise ValueError("Wrong train-only early-stopper split")
    return z,m,fit,inner,val,test

def tensors(z):
    return TensorDataset(
        torch.from_numpy(z["num"].astype(np.float32)),
        torch.from_numpy(z["cat"].astype(np.int64)),
        torch.from_numpy(z["txt"].astype(np.float32)),
        torch.from_numpy(np.log(z["y"].astype(float)).astype(np.float32)))

def run(name,source,output):
    c=settings(name);seed_everything()
    z,m,fit,inner,val,_test=frozen(source)
    dataset=tensors(z)
    net=model_for(name,m["cardinalities"],z["num"].shape[1],z["txt"].shape[1])
    optim=torch.optim.AdamW(net.parameters(),lr=c["lr"],weight_decay=c["wd"])
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,T_max=c["epochs"],eta_min=.00006)
    loader=DataLoader(Subset(dataset,fit.tolist()),shuffle=True,batch_size=c["batch"])
    best=float("inf");state=None;best_epoch=0;stale=0
    for epoch in range(1,c["epochs"]+1):
        net.train()
        for num,cat,txt,target in loader:
            optim.zero_grad(set_to_none=True)
            if c["jitter"]:num=num+torch.randn_like(num)*c["jitter"]
            pred=net(num,cat,txt)
            loss=nn.functional.smooth_l1_loss(pred[:,0],target,beta=c["huber"])
            if not torch.isfinite(loss):raise ValueError("Invalid loss")
            loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);optim.step()
        schedule.step()
        p=predict(net,dataset,inner,c["batch"])
        mae=metrics(z["y"][inner].astype(float),p)["mae_usd"]
        print(json.dumps({"name":name,"epoch":epoch,"inner_mae":mae}),flush=True)
        if mae<best-.1:
            best,best_epoch,state,stale=mae,epoch,copy.deepcopy(net.state_dict()),0
        else:
            stale+=1
            if stale>=c["patience"]:break
    if state is None:raise ValueError("No fitted checkpoint")
    net.load_state_dict(state)
    pv=predict(net,dataset,val,c["batch"])
    result={"status":"completed","trial":name,"config":c,"market":"bid",
            "architecture":"fttransformer","loss":"huber","cohort":"lemon",
            "train_fit":len(fit),"inner_rows":len(inner),"best_epoch":best_epoch,
            "inner_mae_usd":best,"validation":metrics(z["y"][val].astype(float),pv),
            "test_evaluated":False,"canonical_cohort_sha256":m["cohort_sha256"],
            "holdout_row_sha256":m["holdout_row_sha256"],"no_hierarchy":True}
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({"row_id":z["ids"][val].astype(str),"split":"validation",
                  "observed_price":z["y"][val].astype(float),"predicted_price":pv}).to_csv(
        out/"validation.csv.gz",index=False)
    torch.save({"state":state,"trial":name,"config":c,
                "cards":m["cardinalities"],"n_num":z["num"].shape[1],
                "n_text":z["txt"].shape[1]},out/"weights.pt")
    (out/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    print("VALIDATION_RESULT",name,result["validation"]["mae_usd"],flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--trial",choices=tuple(TRIALS),required=True)
    p.add_argument("--data",default="training-cache/neural-bid/features.npz")
    p.add_argument("--output",required=True)
    a=p.parse_args();run(a.trial,a.data,a.output)
