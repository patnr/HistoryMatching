"""Common tools."""

import numpy as np


def norm(xx):
    # return nla.norm(xx/xx.size)
    return np.sqrt(np.sum(xx@xx)/xx.size)


def center(E):
    return E - E.mean(axis=0)


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
