"""Generate initial reservoir realisations with geostatistical methods."""

import numpy as np
import scipy.linalg as sla
from matplotlib import pyplot as plt
from mpl_tools.misc import nRowCol
from numpy.random import randn


def variogram_gauss(xx, r, n=0, a=1/3):
    """Compute the Gaussian variogram for the 1D points xx.

    Params:
    range r, nugget n, and a

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


def vectorize(*XYZ):
    """Reshape coordinate points.

    Input: `nDim` arrays with equal `shape`.
    Let `nPt = np.prod(shape)`
    Output: array of shape `(nPt, nDim)`.
    """
    return np.stack(XYZ).reshape((len(XYZ), -1)).T


def dist_euclid(X):
    """Compute distances, like `squareform(pdist(X))`."""
    diff = X[:, None, :] - X
    d2 = np.sum(diff**2, axis=-1)
    return np.sqrt(d2)


def funm_psd(C, fun, rk=None, rtol=1e-8, sym_square=True, **kwargs):
    """Matrix function evaluation for pos-sem-def mat.

    Adapted from `sla.funm` doc.

    Note: small `rk` and `driver="evx"` should be faster,
    but in my (simple but hopefully relevant) trials
    sticking with the default "evr" and `rk=None` is usually faster.

    Example
    -------
    >>> def sqrtm(C):
    ...     return funm_psd(C, sqrt)
    """
    # EVD -- possibly truncated (for speed)
    idx = [max(0, len(C)-rk), len(C)-1] if rk else None
    ews, V = sla.eigh(C, subset_by_index=idx, **kwargs)

    # Truncate (for stability) -- NB: ordering low-->high!
    nNull = sum(ews <= rtol*ews.max())
    ews = ews[nNull:]
    V = V[:, nNull:]

    # Apply
    ews = fun(ews)

    # Reconstruct
    funC = V * ews
    if sym_square:
        # Optional, since not necessary e.g. for cholesky factors (for sampling)
        funC = funC @ V.T
    return funC


def gaussian_fields(pts, N=1, r=0.2):
    """Random field generation.

    Uses:
    - Gaussian variogram.
    - Gaussian distributions.
    """
    dists  = dist_euclid(vectorize(*pts))
    Cov    = 1 - variogram_gauss(dists, r)
    # C12    = sla.sqrtm(Cov).real  # unstable for len(Cov) >≈ 20
    C12    = funm_psd(Cov, np.sqrt, sym_square=False)
    fields = randn(N, len(C12.T)) @ C12.T
    return fields


if __name__ == "__main__":
    from TPFA_ResSim.grid import Grid2D

    # np.random.seed(3000)
    plt.ion()
    N = 12  # ensemble size

    ## 1D
    xx = np.linspace(0, 1, 201)
    fields = gaussian_fields((xx,), N)

    fignum = "1D-fields"
    plt.figure(num=fignum).clear()
    fig, ax = plt.subplots(num=fignum)
    ax.plot(xx, fields.T, lw=2)

    ## 2D
    fig, axs = plt.subplots(
        num="2D-fields", **nRowCol(min(12, N)), sharex=True, sharey=True)
    grid = Grid2D(Lx=1, Ly=1, Nx=20, Ny=20)
    fields = gaussian_fields(grid.mesh, N)
    fields = 0.5 + .2*fields
    for field, ax in zip(fields, axs.ravel()):
        cc = ax.contourf(field.reshape(grid.shape).T, levels=17)
        ax.contour(field.reshape(grid.shape).T, levels=17)
