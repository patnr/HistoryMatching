"""Common tools."""

import numpy as np


def norm(xx):
    # return nla.norm(xx/xx.size)
    return np.sqrt(np.sum(xx@xx)/xx.size)


def center(E):
    return E - E.mean(axis=0)


def truncate_01(E, warn=""):
    """Saturations should be between 0 and 1."""
    # assert E.max() <= 1 + 1e-10
    # assert E.min() >= 0 - 1e-10
    if (E.max() - 1e-10 >= 1) or (E.min() + 1e-10 <= 0):
        if warn:
            print(f"Warning -- {warn}: needed to truncate ensemble.")
        E = E.clip(0, 1)
    return E


def sigmoid(x):
    """The logistic function."""
    return 1/(1+np.exp(-x))


class Stats:
    """RMSE & STDDEV"""

    def __init__(self, truth, ensemble):
        self.x = truth
        self.E = ensemble

    def __str__(self):
        err = self.x - self.E.mean(axis=0)
        err = norm(err)
        std = np.sqrt((center(self.E)**2).mean())
        return "%6.4f (rmse),  %6.4f (std)" % (err, std)
