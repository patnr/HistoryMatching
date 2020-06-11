"""Generate initial reservoirs."""
## Imports
import numpy as np
import scipy as sp
import scipy.stats as ss

from pylib.all import *
from mpl_tools.misc import *

np.random.seed(9)
rand = ss.uniform(0,1).rvs
randn = ss.norm(0,1).rvs

##

Dx, Dy = 1,1    # Domain lengths
Nx, Ny = 64, 64 # Domain points
gridshape = (Nx,Ny)
sub2ind = lambda ix,iy: np.ravel_multi_index((ix,iy), gridshape)
xy2sub  = lambda x,y: ( int(round(x/Dx*Nx)), int(round(y/Dy*Ny)) )
xy2i    = lambda x,y: sub2ind(*xy2sub(x,y))
M = np.prod(gridshape)

# Resolution
hx, hy = Dx/Nx, Dy/Ny
h2 = hx*hy # Cell volumes (could be array?)



##
# Gaussian variogram
def vg_gauss(xx,r,n=0,a=1/3):
    # Gauss
    gamma = 1 - exp(-xx**2/r**2/a)
    # Sill
    gamma *= (1-n)
    # Nugget
    gamma[xx!=0] += n
    return gamma


##
xx = linspace(0,Dx,Nx)
yy = linspace(0,Dy,Ny)
YY,XX = np.meshgrid(xx,yy)

# Distances
GG = np.vstack((XX.ravel(), YY.ravel())).T # shape (M,2)
diff = GG[:,None,:] - GG                   # shape (M,M,2)
d2   = np.sum(diff**2,axis=-1)             # shape (M,M)
dd   = sqrt(d2)
# Covariance
Cov  = 1 - vg_gauss(dd,.2)
## Generate surfaces
nS = 12
SS = Cov @ randn((M,nS))
## Normalize
# Don't know why the surfaces get such high amplitudes.
# In any case, gotta keep em within [0,1]
sill = 0.7
SS /= (SS.max(axis=0) - SS.min(axis=0)) # make max spread 1
SS -= SS.min(axis=0) # offset to [0,1]
SS *= sill # set amplitude/sill

##
fig, axs = freshfig(1,nrows=3,ncols=int(nS/3),sharex=True,sharey=True)
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
# vv = vg_gauss(xx, .2, n=0.1)

# # Distances
# xx   = xx[:,None]              # shape (M,1)
# diff = xx[:,None,:] - xx       # shape (M,M,1)
# d2   = np.sum(diff**2,axis=-1) # shape (M,M)
# dd   = sqrt(d2)
# C    = 1 - vg_gauss(dd,.2)
# xx = xx.squeeze()
# ##
# SS = C @ randn((M,10))

# fig, ax = freshfig(1)
# ax.plot(xx, SS)
