## Imports
import warnings
import builtins

import numpy as np
import scipy as sp
import scipy.stats as ss
from scipy import sparse
# from scipy.special import errstate
# from scipy.linalg import solve_banded
from scipy.sparse.linalg import spsolve

from pylib.all import *
from mpl_tools.misc import *

# Profiling
try:
    profile = builtins.profile     # will exists if launched via kernprof
except AttributeError:
    def profile(func): return func # provide a pass-through version.


# Ignore warnings
from contextlib import contextmanager
@contextmanager
def ignore_inefficiency():
    warnings.simplefilter("ignore",sparse.SparseEfficiencyWarning)
    yield
    warnings.simplefilter("default",sparse.SparseEfficiencyWarning)

rand = ss.uniform(0,1).rvs
randn = ss.norm(0,1).rvs


## Grid
# Note: x is 1st coord, y is 2nd.
Dx, Dy    = 1,1    # lengths
Nx, Ny    = 32, 32 # num. of pts.
gridshape = Nx,Ny
grid      = Nx, Ny, Dx, Dy
M         = np.prod(gridshape)
sub2ind = lambda ix,iy: np.ravel_multi_index((ix,iy), gridshape)
xy2sub  = lambda x,y: (int(round( x/Dx*(Nx-1) )),
                       int(round( y/Dy*(Ny-1) )))
xy2i    = lambda x,y: sub2ind(*xy2sub(x,y))

# Resolution
hx, hy = Dx/Nx, Dy/Ny
h2 = hx*hy # Cell volumes (could be array?)

Gridded = Bunch(
    K  =np.ones((2,*gridshape)), # permeability
    por=np.ones(gridshape),   # porosity
)

Fluid = Bunch(
     vw=1.0,  vo=1.0,  # Viscosities
    swc=0.0, sor=0.0 # Irreducible saturations
)


def normalize_wellset(ww):
    ww = array(ww,float).T
    ww[0] *= Dx
    ww[1] *= Dy
    ww[2] /= ww[2].sum()
    return ww.T

def plot_field(ax, field, *args, **kwargs):
    # Needed to transpose coz contour() uses
    # the same orientation as array printing.
    field = field.reshape(gridshape).T
    # Center nodes (coz finite-volume)
    xx = linspace(0,Dx-hx,Nx)+hx/2
    yy = linspace(0,Dy-hy,Ny)+hy/2
    collections = ax.contourf(xx, yy, field,
        *args, levels=21,vmax=1,**kwargs)
    ax.set(xlim=(0,1),ylim=(0,1))
    # ax.set(xlim=(hx/2,1-hx/2),ylim=(hy/2,1-hy/2)) # tight
    if ax.is_first_col(): ax.set_ylabel("y")
    if ax.is_last_row (): ax.set_xlabel("x")
    return collections


def fig_colorbar(fig,collections):
    fig.subplots_adjust(right=0.8)
    cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(collections, cax)
