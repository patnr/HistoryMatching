"""Based on Matlab codes from NTNU/Sintef:

http://folk.ntnu.no/andreas/papers/ResSimMatlab.pdf

Translated to python by Patrick N. Raanes.
"""

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


# TODO: 1d case

np.random.seed(9)
rand = ss.uniform(0,1).rvs


## Grid
# Note: x is 1st coord, y is 2nd.
Dx, Dy = 1,1    # Domain lengths
Nx, Ny = 64, 64 # Domain points
gridshape = (Nx,Ny)
sub2ind = lambda ix,iy: np.ravel_multi_index((ix,iy), gridshape)
xy2sub  = lambda x,y: ( int(round(x/Dx*Nx)), int(round(y/Dy*Ny)) )
xy2i    = lambda x,y: sub2ind(*xy2sub(x,y))
N = np.prod(gridshape)

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

## Wells
Q = np.zeros(N)
# Injectors
injectors = rand((3,7))
injectors[0] *= Dx
injectors[1] *= Dy
injectors[2] /= injectors[2].sum()
for x,y,q in zip(*injectors):
    Q[xy2i(x,y)] = q
# Producers
producers = rand((3,5))
producers[0] *= Dx
producers[1] *= Dy
producers[2] /= -producers[2].sum()
for x,y,q in zip(*producers):
    Q[xy2i(x,y)] = q


## Functions
@profile
def RelPerm(s,Fluid,nargout_is_4=False):
    """Rel. permeabilities of oil and water."""
    S = (s-Fluid.swc)/(1-Fluid.swc-Fluid.sor) # Rescale saturations
    Mw = S**2/Fluid.vw # Water mobility
    Mo =(1-S)**2/Fluid.vo # Oil mobility
    # Derivatives:
    # dMw = 2*S/Fluid.vw/(1-Fluid.swc-Fluid.sor)
    # dMo = -2*(1-S)/Fluid.vo/(1-Fluid.swc-Fluid.sor)
    return Mw,Mo

@profile
def upwind_diff(Gridded,V,q):
    """Upwind finite-volume scheme."""
    fp =   q.clip(max=0) # production
    x1 = V.x.clip(max=0)[:-1,:].ravel() # separate flux into
    y1 = V.y.clip(max=0)[:,:-1].ravel() # - flow in positive coordinate
    x2 = V.x.clip(min=0)[1:,:] .ravel() # - flow in negative coordinate
    y2 = V.y.clip(min=0)[:,1:] .ravel() #   direction (XN,YN)
    DiagVecs=[    x2, y2, fp+y1-y2+x1-x2, -y1, -x1] # diagonal vectors
    DiagIndx=[  -Ny , -1,        0      ,  1 ,  Ny] # diagonal index
    A=sparse.spdiags(DiagVecs,DiagIndx,N,N) # matrix with upwind FV stencil
    return A

@profile
def TPFA(Gridded,K,q):
    """Two-point flux-approximation (TPFA) of Darcy:

    diffusion w/ nonlinear coefficient K."""
    # Compute transmissibilities by harmonic averaging.
    L = K**(-1)
    TX = np.zeros((Nx+1,Ny))
    TY = np.zeros((Nx,Ny+1))

    TX[1:-1,:] = 2*hy/hx/(L[0,:-1,:] + L[0,1:,:])
    TY[:,1:-1] = 2*hx/hy/(L[1,:,:-1] + L[1,:,1:])

    # Assemble TPFA discretization matrix.
    x1 = TX[:-1,:].ravel(); x2 = TX[1:,:].ravel()
    y1 = TY[:,:-1].ravel(); y2 = TY[:,1:].ravel()

    DiagVecs = [-x2, -y2, y1+y2+x1+x2, -y1, -x1]
    DiagIndx = [-Ny,  -1,      0     ,   1,  Ny]
    # Coerce system to be SPD (ref article, page 13).
    DiagVecs[2][0] += np.sum(Gridded.K[:,0,0])

    # Solve linear system and extract interface fluxes.

    # Note on the matrix inversion:
    # We would like to use solve_banded (not solveh_banded),
    # despite it being somewhat convoluted (https://github.com/scipy/scipy/issues/2285)
    # which according to stackexchange (see below) uses the Thomas algorithm,
    # as recommended by Aziz and Settari ("Petro. Res. simulation").
    # (TODO: How can I specify offset from diagonal?)
    # ab = array([x for (x,n) in zip(DiagVecs,DiagIndx)])
    # u = solve_banded((3,3), ab, q, check_finite=False)

    # However, according to https://scicomp.stackexchange.com/a/30074/1740
    # solve_banded does not work well for when the band offsets large,
    # i.e. higher-dimensional problems.
    # Therefore we use sp.sparse.linalg.spsolve, even though it
    # converts DIAgonal formats to CSC (and throws inefficiency warning).
    A = sparse.spdiags(DiagVecs, DiagIndx, N, N)
    with ignore_inefficiency():
        u = spsolve(A,q)
    # The above is still much more efficient than going to full matrices,
    # indeed I get comparable speed to Matlab.
    # A = A.toarray()
    # u = np.linalg.solve(A, q)

    # Other options to consider: scipy.sparse.linalg.lsqr, etc.

    P = u.reshape(gridshape)

    V = Bunch(
        x = np.zeros((Nx+1,Ny)),
        y = np.zeros((Nx,Ny+1)),
    )
    V.x[1:-1,:] = (P[:-1,:] - P[1:,:]) * TX[1:-1,:]
    V.y[:,1:-1] = (P[:,:-1] - P[:,1:]) * TY[:,1:-1]
    return P,V

@profile
def pressure_step(Gridded,S,Fluid,q):
    """TPFA finite-volume of Darcy: -nabla(K lambda(s) nabla(u)) = q."""
    # Compute K*lambda(S)
    Mw,Mo = RelPerm(S,Fluid)
    Mt = Mw+Mo
    Mt = Mt.reshape(gridshape)
    KM = Mt*Gridded.K
    # Compute pressure and extract fluxes
    [P,V]=TPFA(Gridded,KM,q)
    return P, V

@profile
def saturation_step(Gridded,S,Fluid,q,V,T):
    """Explicit upwind finite-volume discretisation of CoM."""
    pv = h2*Gridded['por'].ravel() # pore volume=cell volume*porosity

    fi = q.clip(min=0)# inflow from wells

    XP=V.x.clip(min=0); XN=V.x.clip(max=0) # influx and outflux, x-faces
    YP=V.y.clip(min=0); YN=V.y.clip(max=0) # influx and outflux, y-faces

    Vi = XP[:-1]-XN[1:]+YP[:,:-1]-YN[:,1:] # each gridblock

    # Compute dt
    from numpy import errstate
    with errstate(divide="ignore"):
        pm  = min(pv/(Vi.ravel()+fi)) # estimate of influx
    cfl = ((1-Fluid.swc-Fluid.sor)/3)*pm # CFL restriction # NB: 3-->2 since no z ?
    Nts = int(np.ceil(T/cfl)) # number of local time steps
    dtx = (T/Nts)/pv # local time steps

    # Discretized transport operator
    A=upwind_diff(Gridded,V,q)     # system matrix
    A=sparse.spdiags(dtx,0,N,N)@A # A * dt/|Omega i|

    for iT in range(1,Nts+1):
        mw,mo=RelPerm(S,Fluid)  # compute mobilities
        fw = mw/(mw+mo)         # compute fractional flow
        S = S + (A@fw + fi*dtx) # update saturation

    return S

def liveplot(S,t):

    if not liveplot.init:
        for c in liveplot.CC.collections:
            ax.collections.remove(c)

    liveplot.CC = ax.contourf(
        linspace(0,Dx-hx,Nx)+hx/2,
        linspace(0,Dy-hy,Ny)+hy/2,
        S.reshape(gridshape).T,
        # Needed to transpose coz contour() uses
        # the same orientation as array printing.
        levels=linspace(0,1,11), vmin=0,vmax=1)

    ax.set_title("Water saturation, t = %.1f"%(t))
    if liveplot.init:
        fig.colorbar(liveplot.CC)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.plot(*injectors[:2], "v", ms=16)
        ax.plot(*producers[:2], "^", ms=16)
    # ax.set_aspect("equal")

    plt.pause(.01)
    liveplot.init = False


@profile
def simulate(nSteps,dt_animation=.025,plotting=True):
    S=np.zeros(N) # Initial saturation

    for iT in 1+arange(nSteps):
        [P,V] =   pressure_step(Gridded,S,Fluid,Q)
        S     = saturation_step(Gridded,S,Fluid,Q,V,dt_animation)

        t = iT*dt_animation
        if plotting:
            if iT==1:
                liveplot.init = True
            liveplot(S,t)

    return P,V,S


## Main
if __name__ == "__main__":

    plt.ion()
    fig, ax = freshfig(1)

    P,V,S = simulate(28)
    # P,V,S = simulate(1)
