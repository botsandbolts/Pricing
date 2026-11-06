"""Price-free encoder embedding -> learnable projector -> DeepCross -> AFT.

The transformer is frozen, and its ONLY output used here is a sentence
embedding. NO text-to-price heads, text-derived price scalars, TF-IDF or SVD
enter the neural network. The entire price function is trained jointly by
DeepCross on vehicle specifications + raw pretrained embedding coordinates.

The original 211127 listing identities and 147777/31668/31682 chronological
split are immutable. Trials return validation ONLY. A separate report selects
the encoder by validation and evaluates the test only once.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from scripts.deepcross_aft_tune import TunedDeepCross, config
from scripts.neural_ask_train import metrics, objective, predict, seed_everything
from scripts.raw_embedding_ask_encode import SPECS, id_hash

PROJECTION_DIM = 96


class EmbeddingOnlyDeepCross(nn.Module):
    def __init__(self, category_sizes, numeric_dim, embedding_dim):
        super().__init__()
        self.projector = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, PROJECTION_DIM),
            nn.GELU(),
            nn.LayerNorm(PROJECTION_DIM),
            nn.Dropout(.08))
        self.price_network = TunedDeepCross(
            category_sizes, numeric_dim, PROJECTION_DIM, config("dropout12"))

    def forward(self, numeric, category, embedding):
        representation = self.projector(embedding)
        return self.price_network(numeric, category, representation)


def frozen_source(original, enc_dir, candidate):
    original = Path(original)
    enc_dir = Path(enc_dir)
    if candidate not in SPECS:
        raise ValueError("Unregistered frozen embedding encoder")
    source = json.loads(original.with_suffix(".json").read_text())
    emeta = json.loads((enc_dir / "manifest.json").read_text())
    if ((source.get("full_train"), source.get("validation"), source.get("test"))
        != (147777, 31668, 31682)):
        raise ValueError("Original asking-price cohort has changed")
    if (emeta.get("status"), emeta.get("variant")) != ("complete", candidate):
        raise ValueError("Unverified pretrained embedding candidate")
    if (emeta.get("model_id"), emeta.get("max_tokens"), emeta.get("dimensionality")) != SPECS[candidate]:
        raise ValueError("Wrong frozen embedding model or context length")
    if (emeta.get("price_derived_features") is not False or
        emeta.get("text_price_heads") != 0 or
        emeta.get("supervised_encoder_finetuning") is not False):
        raise ValueError("NO PRICE PREDICTION FEATURE OR HEAD IS PERMITTED")
    if (emeta["canonical_cohort_sha256"] != source["cohort_sha256"] or
        emeta["holdout_row_sha256"] != source["holdout_row_sha256"]):
        raise ValueError("Pretrained embeddings were not built on original immutable cohort")
    parity = emeta["batch_consistency"]
    if parity["max_absolute_difference"] > .005 or parity["minimum_cosine_similarity"] < .999:
        raise ValueError("Unreliable batch-one serving representation")
    with open(enc_dir / "embeddings.npy", "rb") as file:
        h = hashlib.file_digest(file, "sha256").hexdigest()
    if h != emeta["embedding_file_sha256"]:
        raise ValueError("Encoder embeddings artifact checksum mismatch")
    with np.load(original, allow_pickle=False) as f:
        z = {k: f[k] for k in ("num", "cat", "y", "phase", "ids")}
    embeddings = np.load(enc_dir / "embeddings.npy", mmap_mode="r")
    if embeddings.shape != (211127, SPECS[candidate][2]) or embeddings.dtype != np.float16:
        raise ValueError("Wrong frozen embedding matrix shape or precision")
    if (z["num"].shape, z["cat"].shape) != ((211127, 7), (211127, 11)):
        raise ValueError("Expected original seven numeric / eleven categorical features only")
    if len(z["ids"]) != 211127 or len(np.unique(z["ids"])) != 211127:
        raise ValueError("Missing canonical vehicle ID")
    if (not np.isfinite(z["y"]).all() or (z["y"] <= 0).any() or
        not np.isfinite(embeddings).all()):
        raise ValueError("Invalid original prices or pretrained embeddings")
    phases = z["phase"].astype(str)
    for split, length in (("train",147777),("validation",31668),("test",31682)):
        if int(sum(phases == split)) != length:
            raise ValueError("Wrong chronological " + split + " split")
    for split in ("validation","test"):
        if id_hash(z["ids"][phases == split].astype(str)) != source["holdout_row_sha256"][split]:
            raise ValueError("Original holdout identity fingerprint differs")
    return z, np.asarray(embeddings,dtype=np.float32), source, emeta


def train(candidate, original, embedding_dir, output, seed=20261106,
          epochs=35, patience=7, batch_size=768):
    z, embeddings, provenance, emeta = frozen_source(original,embedding_dir,candidate)
    seed_everything(seed)
    phases = z["phase"].astype(str)
    fit = np.flatnonzero(phases == "train")[:-14777]
    early = np.flatnonzero(phases == "train")[-14777:]
    val = np.flatnonzero(phases == "validation")
    if (len(fit),len(early),len(val))!=(133000,14777,31668):
        raise ValueError("Immutable training stopper/validation split changed")
    dataset = TensorDataset(
        torch.from_numpy(z["num"].astype(np.float32)),
        torch.from_numpy(z["cat"].astype(np.int64)),
        torch.from_numpy(embeddings),
        torch.from_numpy(np.log(z["y"].astype(float)).astype(np.float32)))
    net = EmbeddingOnlyDeepCross(provenance["cardinalities"],7,embeddings.shape[1])
    optim = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.025)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,T_max=epochs,eta_min=.00006)
    batches = DataLoader(Subset(dataset,fit.tolist()),batch_size=batch_size,
                         shuffle=True,num_workers=0)
    best=float("inf"); best_state=None;best_epoch=0;stale=0
    for epoch in range(1,epochs+1):
        net.train()
        total=0.;rows=0
        for numeric,categories,text,target in batches:
            optim.zero_grad(set_to_none=True)
            numeric = numeric + torch.randn_like(numeric)*.008
            value = net(numeric,categories,text)
            loss = objective(value,target,"aft")
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite exact-observation AFT loss")
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.)
            optim.step()
            total+=float(loss.detach())*len(target);rows+=len(target)
        schedule.step()
        early_pred=predict(net,dataset,early,batch_size)
        mae=metrics(z["y"][early].astype(float),early_pred)["mae_usd"]
        print(json.dumps({"candidate":candidate,"seed":seed,"epoch":epoch,
                          "training_loss":total/rows,"internal_mae_usd":mae}),flush=True)
        if mae<best-.1:
            best, best_state, best_epoch, stale = mae,copy.deepcopy(net.state_dict()),epoch,0
        else:
            stale+=1
            if stale>=patience:break
    if best_state is None:
        raise ValueError("Training never produced a valid neural checkpoint")
    net.load_state_dict(best_state)
    pv=predict(net,dataset,val,batch_size)
    scores=metrics(z["y"][val].astype(float),pv)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({"row_id":z["ids"][val].astype(str),"split":"validation",
                  "observed_price":z["y"][val].astype(float),
                  "predicted_price":pv}).to_csv(output/"validation.csv.gz",index=False)
    torch.save({"state_dict":best_state,"candidate":candidate,
                "seed":seed,"model_id":emeta["model_id"],
                "model_revision":emeta["model_revision"],
                "max_tokens":emeta["max_tokens"],
                "cardinalities":provenance["cardinalities"],
                "n_num":7,"embedding_dim":embeddings.shape[1],
                "projection_dim":PROJECTION_DIM,
                "cross_configuration":config("dropout12"),
                "price_head_location":"DeepCross only",
                "pretrained_text_price_heads":0},output/"weights.pt")
    record={"status":"complete","market":"ask",
            "candidate":candidate,
            "architecture":"DeepCross + learnable raw-embedding projection",
            "objective":"exact observation log-normal AFT",
            "encoder":emeta["model_id"],
            "encoder_revision":emeta["model_revision"],
            "max_tokens":emeta["max_tokens"],
            "embedding_dimension":embeddings.shape[1],
            "embedding_normalization":"pretrained L2 + learned LayerNorm",
            "text_features":"raw pretrained embeddings ONLY; no Ridge/text-price/SVD",
            "text_price_heads":0,"price_derived_inputs":False,
            "price_predictions_generated_by":"complete DeepCross AFT neural model",
            "training_seed":seed,
            "full_train_rows":147777,"fitting_rows":len(fit),
            "internal_early_stopper":len(early),
            "validation_rows":len(val),
            "test_predictions_created":False,
            "validation":scores,"best_epoch":best_epoch,
            "internal_stopper_mae_usd":best,
            "source_fingerprint":provenance["cohort_sha256"],
            "holdout_row_sha256":provenance["holdout_row_sha256"],
            "embedding_fingerprint":emeta["embedding_file_sha256"],
            "batch1_parity":emeta["batch_consistency"],
            "selection_policy":"external validation MAE; test accessed only by selected winner in report"}
    (output/"result.json").write_text(json.dumps(record,indent=2,allow_nan=False))
    print("VALIDATION_ONLY",candidate,seed,scores["mae_usd"],flush=True)
    return record


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate",required=True,choices=tuple(SPECS))
    p.add_argument("--original",default="training-cache/neural-ask/features.npz")
    p.add_argument("--embeddings",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--seed",type=int,default=20261106)
    p.add_argument("--epochs",type=int,default=35)
    p.add_argument("--patience",type=int,default=7)
    p.add_argument("--batch-size",type=int,default=768)
    a=p.parse_args()
    train(a.candidate,a.original,a.embeddings,a.output,a.seed,a.epochs,a.patience,a.batch_size)
