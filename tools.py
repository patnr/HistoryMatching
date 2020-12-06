"""Common tools."""

import numpy as np

def center(E):
    return E - E.mean(axis=0)

def norm(xx):
    # return nla.norm(xx/xx.size)
    return np.sqrt(np.mean(xx*xx))

def RMS(truth, ensemble):
    """RMS error & dev."""
    mean = ensemble.mean(axis=0)
    err  = truth - mean
    dev  = ensemble - mean
    return "%6.4f (rmse),  %6.4f (std)" % (norm(err), norm(dev))

def RMS_all(series, vs):
    """RMS for each item in series."""
    for k in series:
        if k != vs:
            print(f"{k:12}:", RMS(series[vs], series[k]))
