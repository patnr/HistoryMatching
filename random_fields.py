"""Generate initial reservoir realisations.
"""

import numpy as np
import scipy.linalg as sla
from matplotlib import pyplot as plt
from mpl_tools.misc import fig_colorbar, freshfig
from numpy.random import randn

from model.grid import Grid2D


def variogram_gauss(xx, r, n=0, a=1/3):
    """Compute the Gaussian variogram for the  points xx.

    Params:
    radius r, nugget n, and a

    Ref:
    https://en.wikipedia.org/wiki/Variogram#Variogram_models

    Example:
    >>> xx = np.array([0, 1, 2])
    >>> variogram_gauss(xx, 1, n=0.1, a=1)
    array([0.        , 0.6689085 , 0.98351593])
    """
    # Gauss
    gamma = 1 - np.exp(-xx**2/r**2/a)
    # Sill (=1)
    gamma *= (1-n)
    # Nugget
    gamma[xx != 0] += n
    return gamma


def comp_dists(XX, YY):
    """Compute distances."""
    GG = np.vstack((XX.ravel(), YY.ravel())).T  # shape (M,2)
    diff = GG[:, None, :] - GG                  # shape (M,M,2)
    d2 = np.sum(diff**2, axis=-1)               # shape (M,M)
    dd = np.sqrt(d2)
    return dd


def gen_surfs(n, Cov):
    M = len(Cov)
    SS = sla.sqrtm(Cov).real @ randn(M, n)

    # Normalize
    # TODO Don't know why the random fields get such high amplitudes.
    # In any case, gotta keep em within [0,1]
    normfactor = SS.max() - SS.min()
    SS /= normfactor  # make max spread 1
    SS -= SS.min(axis=0)
    return SS, normfactor


def gen_cov(grid, radius=0.5):
    XX, YY = grid.mesh_coords()
    dd = comp_dists(XX, YY)
    return 1 - variogram_gauss(dd, radius)


def gen_ens_01(grid, N):
    Cov = gen_cov(grid)
    SS, normfactor = gen_surfs(N, Cov)
    return SS, Cov/normfactor**2


def gen_ens(grid, N, sill):
    # Return ensemble of fields, and its true covariance
    SS, Cov = gen_ens_01(grid, N)
    SS *= sill  # set amplitude/sill
    return SS.T, Cov*sill**2


if __name__ == "__main__":
    np.random.seed(9)

    ## 2D
    N = 100
    sill = 0.7

    grid = Grid2D(Lx=4, Ly=10, Nx=2, Ny=5)
    SS, Cov = gen_ens(grid, N, sill)

    fig, axs = freshfig(21, nrows=3, ncols=int(12/3),
                        sharex=True, sharey=True)
    CC = []
    for i, (ax, S) in enumerate(zip(axs.ravel(), SS)):
        ax.set_title(i)
        CC.append(ax.contourf(
            1 - S.reshape(grid.gridshape).T,
            levels=21, vmin=1-sill, vmax=1))
    fig_colorbar(fig, CC[0])

    ## 1D
    M  = 201
    xx = np.linspace(0, 1, M)
    vv = variogram_gauss(xx, .2, n=0.1)

    # Distances
    dd = comp_dists(xx, np.zeros_like(xx))
    C  = 1 - variogram_gauss(dd, .2)
    SS = C @ randn(M, 10)

    fig, ax = freshfig(1)
    ax.plot(xx, SS)
    plt.pause(.01)
