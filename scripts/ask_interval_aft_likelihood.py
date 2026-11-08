"""Censor-aware log-normal AFT price likelihood.

Prices refer to the SAME defined target. The original Craigslist Ask source
has EXACT observed asking prices (lower == upper) and ZERO measured censored
labels. Never infer sale-price intervals from an advertised asking price.

This module provides an optional likelihood for future verified observations:
 exact       L == U  > 0
 left        L == 0, U > 0 finite
 right       L > 0, U == +inf
 interval    0 < L < U finite
Input bounds are in DOLLARS, while network location is log dollars.
"""
import math
import torch
from torch.nn import functional as F


def _logdiffexp(log_big,log_small):
    """Stable log(exp(big)-exp(small)), with big > small."""
    # log1p avoids cancellation for nearby probabilities in ordinary ranges.
    # Clamp only at representable precision, not to invented price bounds.
    delta=torch.clamp(log_small-log_big,max=-torch.finfo(log_big.dtype).eps)
    return log_big+torch.log(-torch.expm1(delta))


def lognormal_aft_nll(raw,lower,upper,sigma_floor=.035):
    """Mean negative log-likelihood, supporting genuine censoring on price.

    raw: (N,2) log-price mu and *unconstrained* log-scale head.
    lower/upper: measured positive dollar bounds, with 0/+inf special cases.
    The exact event PDF includes log(y), constant w.r.t. predictions.
    """
    if raw.ndim!=2 or raw.shape[1]!=2:
        raise ValueError("AFT network requires (n,2) location/scale outputs")
    mu=raw[:,0]
    if lower.shape!=mu.shape or upper.shape!=mu.shape:
        raise ValueError("AFT bounds do not match network batch size")
    if not 0<sigma_floor<=.5:
        raise ValueError("AFT sigma_floor must be positive and <= .5")
    lower=lower.to(device=raw.device,dtype=raw.dtype)
    upper=upper.to(device=raw.device,dtype=raw.dtype)
    if (not bool(torch.isfinite(lower).all().item())
        or bool(torch.isnan(upper).any().item())
        or bool((lower<0).any().item())
        or bool((upper<=0).any().item())
        or bool((upper<lower).any().item())
        or bool(torch.isneginf(upper).any().item())
        or bool((lower==0).logical_and(torch.isinf(upper)).any().item())):
        raise ValueError("Missing, inconsistent or uninformative AFT price bounds")
    sigma=(F.softplus(raw[:,1])+sigma_floor).clamp(max=3.)
    exact=(lower==upper)
    left=(lower==0)&torch.isfinite(upper)
    right=(lower>0)&torch.isposinf(upper)
    interval=(lower>0)&torch.isfinite(upper)&(upper>lower)
    if not bool((exact|left|right|interval).all().item()):
        raise ValueError("Unrecognized AFT censoring pattern")
    neglog=torch.zeros_like(mu)
    if exact.any():
        y=lower[exact]
        z=(y.log()-mu[exact])/sigma[exact]
        # density of log-normal in dollars, not a time-to-sale model
        neglog[exact]=.5*z.square()+sigma[exact].log()+y.log()+.5*math.log(2*math.pi)
    if left.any():
        z=(upper[left].log()-mu[left])/sigma[left]
        neglog[left]=-torch.special.log_ndtr(z)
    if right.any():
        z=(lower[right].log()-mu[right])/sigma[right]
        neglog[right]=-torch.special.log_ndtr(-z)
    if interval.any():
        zl=(lower[interval].log()-mu[interval])/sigma[interval]
        zu=(upper[interval].log()-mu[interval])/sigma[interval]
        # For positive z, subtract survival functions to avoid 1-1.
        loglo=torch.special.log_ndtr(zl)
        loghi=torch.special.log_ndtr(zu)
        logsurv_lo=torch.special.log_ndtr(-zl)
        logsurv_hi=torch.special.log_ndtr(-zu)
        prob_log=torch.where(
            zl>0,
            _logdiffexp(logsurv_lo,logsurv_hi),
            _logdiffexp(loghi,loglo),
        )
        neglog[interval]=-prob_log
    result=neglog.mean()
    if not bool(torch.isfinite(result).item()):
        raise ValueError("Nonfinite censor-aware lognormal AFT loss")
    return result


def exact_ask_aft_nll(raw,asking_price,sigma_floor=.035):
    """Correct adapter for the current dataset: *no censoring whatsoever*."""
    if not bool((torch.isfinite(asking_price)&(asking_price>0)).all().item()):
        raise ValueError("Exact Ask label must be positive finite dollars")
    return lognormal_aft_nll(raw,asking_price,asking_price,sigma_floor=sigma_floor)
