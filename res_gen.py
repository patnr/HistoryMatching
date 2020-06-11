"""Generate initial reservoirs."""
## Imports
import numpy as np
import scipy as sp
import scipy.stats as ss

from pylib.all import *
from mpl_tools.misc import *

## Functions

rand = ss.uniform(0,1).rvs
randn = ss.norm(0,1).rvs

def variogram_gauss(xx,r,n=0,a=1/3):
    # Gauss
    gamma = 1 - exp(-xx**2/r**2/a)
    # Sill
    gamma *= (1-n)
    # Nugget
    gamma[xx!=0] += n
    return gamma

def gen_grid(Nx,Ny,Dx,Dy):
    xx = linspace(0,Dx,Nx)
    yy = linspace(0,Dy,Ny)
    YY,XX = np.meshgrid(xx,yy)
    return XX,YY

def comp_dists(XX,YY):
    GG = np.vstack((XX.ravel(), YY.ravel())).T # shape (M,2)
    diff = GG[:,None,:] - GG                   # shape (M,M,2)
    d2   = np.sum(diff**2,axis=-1)             # shape (M,M)
    dd   = sqrt(d2)
    return dd

def gen_surfs(n,Cov):
    M = len(Cov)
    SS = Cov @ randn((M,n))

    # Normalize
    # Don't know why the surfaces get such high amplitudes.
    # In any case, gotta keep em within [0,1]
    SS /= (SS.max(axis=0) - SS.min(axis=0)) # make max spread 1
    SS -= SS.min(axis=0)
    return SS

def gen_ens_01(N, grid):
    XX,YY = gen_grid(*grid)
    dd = comp_dists(XX,YY)
    Cov = 1 - variogram_gauss(dd,.2)
    SS = gen_surfs(N,Cov)
    return SS

def gen_ens(N,grid,sill):
    SS = gen_ens_01(N, grid)
    SS *= sill # set amplitude/sill
    return SS


## Plot
if __name__ == "__main__":
    from simulation import grid
    gridshape = grid[:2]

    np.random.seed(9)

    N = 12
    sill = 0.7
    SS = gen_ens(N,grid,sill)

    fig, axs = freshfig(1,nrows=3,ncols=int(N/3),sharex=True,sharey=True)
    CC = []
    for i, (ax, S) in enumerate(zip(axs.ravel(),SS.T)):
        ax.set_title(i)
        collections = ax.contourf(1 - S.reshape(gridshape).T,
                    levels=21,vmin=1-sill,vmax=1)
        CC.append(collections)

    fig.subplots_adjust(right=0.8)
    cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(CC[0], cax)


## 1D
# M  = 201
# xx = linspace(0,1,M)
# vv = variogram_gauss(xx, .2, n=0.1)
#
# # Distances
# xx   = xx[:,None]              # shape (M,1)
# diff = xx[:,None,:] - xx       # shape (M,M,1)
# d2   = np.sum(diff**2,axis=-1) # shape (M,M)
# dd   = sqrt(d2)
# C    = 1 - variogram_gauss(dd,.2)
# xx   = xx.squeeze()
# SS   = C @ randn((M,10))
#
# fig, ax = freshfig(1)
# ax.plot(xx, SS)
