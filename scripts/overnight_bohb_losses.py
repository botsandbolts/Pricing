"""Experimental TRAIN-only regression objectives for overnight BOHB.

All predictions are trained on log(price). CVaR95 at evaluation remains exact
worst-five-percent absolute *dollar* error, independent of the train loss.
Never read original validation or test to compute the training loss.
"""
import math
import torch
from torch.nn import functional as F


LOSS_NAMES=(
    "aft", "huber_log", "charbonnier_log", "logcosh_log", "mae_log",
    "mse_log", "expectile_log", "huber_dollar", "hybrid_aft_dollar",
    "cvar_dollar", "soft_cvar_dollar", "focal_dollar",
    "asymmetric_dollar", "hybrid_log_dollar",
)


def _log_mu(pred,market):
    if market=="ask":
        if pred.ndim != 2 or pred.shape[1] != 2:
            raise ValueError("Ask network must emit log-location and log-scale heads")
        return pred[:,0]
    if market=="bid":
        if pred.ndim != 1:
            raise ValueError("Bid network must emit one log-price per row")
        return pred
    raise ValueError("Unknown market")


def regression_loss(pred,log_target,hp,market):
    """Dimensionless, bounded-gradient tail objectives; strict finite check."""
    kind=hp["loss"]
    if kind not in LOSS_NAMES:
        raise ValueError("Unknown BOHB training loss "+str(kind))
    mu=_log_mu(pred,market)
    if mu.shape!=log_target.shape:
        raise ValueError("Mismatched BOHB target/prediction shape")
    logdiff=mu-log_target
    abslog=logdiff.abs()
    beta=float(hp["beta"])
    if not (0 < beta <= 2):
        raise ValueError("Bad log-Huber radius")
    tau=float(hp["tau"])
    tail=float(hp["tail_fraction"])
    strength=float(hp["tail_strength"])
    scale=float(hp["usd_scale"])
    temp=float(hp["tail_temperature"])
    if not (0<tau<1 and .01<=tail<=.30 and 0<=strength<=2
            and 1000<=scale<=50000 and .005<=temp<=2):
        raise ValueError("Invalid risk-loss hyperparameters")
    huber=F.smooth_l1_loss(mu,log_target,beta=beta)
    if kind=="aft":
        if market=="ask":
            sigma=(F.softplus(pred[:,1])+float(hp["sigma_floor"])).clamp(max=3.)
            basic=(.5*(logdiff/sigma).square()+sigma.log()).mean()
        else:
            basic=huber
    elif kind=="huber_log":
        basic=huber
    elif kind=="charbonnier_log":
        basic=(torch.sqrt(logdiff.square()+beta*beta)-beta).mean()
    elif kind=="logcosh_log":
        basic=(abslog+F.softplus(-2*abslog)-math.log(2.)).mean()
    elif kind=="mae_log":
        basic=abslog.mean()
    elif kind=="mse_log":
        basic=logdiff.square().mean()
    elif kind=="expectile_log":
        basic=torch.where(logdiff>=0,tau,1-tau).mul(logdiff.square()).mean()
    else:
        basic=huber

    # Guard exponential before differentiating; prices in USD, dollar error
    # scaled to roughly 1, preserving safe gradients for early unstable trials.
    dollar_pred=mu.clamp(min=math.log(100),max=math.log(1000000)).exp()
    dollar_true=log_target.detach().exp()
    residual=(dollar_pred-dollar_true)/scale
    absd=residual.abs()
    d_huber=F.smooth_l1_loss(residual,torch.zeros_like(residual),beta=.5)
    top_n=max(1,math.ceil(tail*len(residual)))
    worst=absd.topk(top_n).values.mean()
    threshold=torch.quantile(absd.detach(),1-tail)
    soft_tail=(threshold+F.softplus((absd-threshold)/temp).mean()*temp/tail)

    if kind=="huber_dollar":
        basic=d_huber
    elif kind=="hybrid_aft_dollar":
        sigma=(F.softplus(pred[:,1])+float(hp["sigma_floor"])).clamp(max=3.) if market=="ask" else None
        likelihood=(.5*(logdiff/sigma).square()+sigma.log()).mean() if market=="ask" else huber
        basic=(1-strength/3)*likelihood+(strength/3)*d_huber
    elif kind=="cvar_dollar":
        basic=d_huber+strength*worst
    elif kind=="soft_cvar_dollar":
        basic=d_huber+strength*soft_tail
    elif kind=="focal_dollar":
        gamma=float(hp["focal_gamma"])
        basic=(absd.pow(1+gamma)/(1+gamma)).mean()+.15*huber
    elif kind=="asymmetric_dollar":
        basic=(torch.where(residual>=0,tau,1-tau)*absd).mean()+.2*huber
    elif kind=="hybrid_log_dollar":
        basic=(1-strength/3)*huber+(strength/3)*d_huber

    # The strength knob also permits tail-aware versions of conventional
    # log-price training losses without inspecting held-out outcomes.
    if kind in ("aft","huber_log","charbonnier_log","logcosh_log",
                "mae_log","mse_log","expectile_log"):
        basic=basic+strength*worst
    if not bool(torch.isfinite(basic).item()):
        raise ValueError("Nonfinite BOHB objective")
    return basic


def make_optimizer(model,hp):
    name=hp["optimizer"]
    opts={"lr":float(hp["lr"]),"weight_decay":float(hp["wd"])}
    betas=(float(hp["beta1"]),float(hp["beta2"]))
    if name=="adamw":
        return torch.optim.AdamW(model.parameters(),betas=betas,**opts)
    if name=="adam":
        return torch.optim.Adam(model.parameters(),betas=betas,**opts)
    if name=="radam":
        return torch.optim.RAdam(model.parameters(),betas=betas,**opts)
    if name=="nadam":
        return torch.optim.NAdam(model.parameters(),betas=betas,**opts)
    if name=="amsgrad":
        return torch.optim.AdamW(model.parameters(),betas=betas,amsgrad=True,**opts)
    raise ValueError("Unknown BOHB optimizer")


def make_scheduler(optimizer,hp):
    """Promotion-consistent step counts: fixed 216-epoch horizon at every rung."""
    mode=hp["schedule"]
    if mode=="constant":return None
    if mode=="cosine216":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,T_max=216,eta_min=min(.00002,float(hp["lr"])*.03))
    if mode=="warm_restart":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,T_0=18,T_mult=2,eta_min=min(.00002,float(hp["lr"])*.03))
    if mode=="step_decay":
        return torch.optim.lr_scheduler.StepLR(optimizer,step_size=36,gamma=.5)
    raise ValueError("Unknown BOHB scheduler")
