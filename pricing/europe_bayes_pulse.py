"""Fast conjugate hierarchical shrinkage for *descriptive* make/model contrasts.

The observation model is d_j | theta_j ~ Normal(theta_j, se_j**2);
theta_j | mu_make ~ Normal(mu_make, tau_model**2);
mu_make ~ Normal(0, tau_make**2).

Brand context is leave-one-model-out and an approximate empirical-Bayes
posterior; hyperparameters are fixed, and between-cohort sampling/covariance
and selection uncertainty are NOT fully captured. This is NOT Bayesian MCMC
nor a validated model-specific time-series price index.
"""
from __future__ import annotations

import math

DEFAULT_TAU_MAKE=0.04
DEFAULT_TAU_MODEL=0.06
Z_95=1.959963984540054


def shrink_model_differences(records, *, tau_make=DEFAULT_TAU_MAKE,
                             tau_model=DEFAULT_TAU_MODEL):
    if not (0 < tau_make < 1 and 0 < tau_model < 1):
        raise ValueError("Invalid prior scale")
    valid=[]
    for r in records:
        try:
            difference=float(r["raw"])
            error=float(r["se"])
            if (not math.isfinite(difference) or not math.isfinite(error)
                    or error<=0 or abs(difference)>0.40):
                continue
            valid.append({**r, "raw":difference, "se":error})
        except (KeyError, ValueError, TypeError):
            continue
    grouped={}
    for r in valid:
        grouped.setdefault(str(r["make"]), []).append(r)

    outputs=[]
    for row in valid:
        siblings=[v for v in grouped[str(row["make"])] if v is not row]
        make_precision=1.0/tau_make**2
        weighted_sum=0.0
        for peer in siblings:
            variance=peer["se"]**2+tau_model**2
            make_precision+=1.0/variance
            weighted_sum+=peer["raw"]/variance
        make_mean=weighted_sum/make_precision
        make_var=1.0/make_precision
        prior_variance=tau_model**2+make_var
        likelihood_variance=row["se"]**2
        likelihood_weight=prior_variance/(likelihood_variance+prior_variance)
        mean=likelihood_weight*row["raw"]+(1-likelihood_weight)*make_mean
        sd=math.sqrt(likelihood_variance*prior_variance/
                     (likelihood_variance+prior_variance))
        probability=0.5*math.erfc(-mean/(sd*math.sqrt(2)))
        lower=mean-Z_95*sd
        upper=mean+Z_95*sd
        if not all(map(math.isfinite,(mean,sd,probability,lower,upper))):
            continue
        outputs.append({**row, "posterior_log_change":mean,
                        "posterior_sd_log":sd,
                        "posterior_probability_positive":probability,
                        "interval_low_log":lower,
                        "interval_high_log":upper,
                        "brand_prior_log":make_mean,
                        "brand_prior_sd_log":math.sqrt(make_var),
                        "brand_peer_count":len(siblings),
                        "shrink_weight":likelihood_weight})
    return outputs
