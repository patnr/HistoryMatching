"""Based on Matlab codes from NTNU/Sintef:

http://folk.ntnu.no/andreas/papers/ResSimMatlab.pdf

Translated to python by Patrick N. Raanes.
"""
import numpy as np
import scipy as sp
from scipy import sparse
# from scipy.special import errstate
import warnings
# from scipy.linalg import solve_banded
from scipy.sparse.linalg import spsolve
from pylib.all import *
from mpl_tools.misc import *
import builtins

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



## Grid
# Note: x is 1st coord, y is 2nd.
Dx, Dy = 1,1    # Domain lengths
Nx, Ny = 64, 64 # Domain points
gridshape = (Nx,Ny)
sub2ind = lambda ix,iy: np.ravel_multi_index((ix,iy), gridshape)
xy2sub  = lambda x,y: [ int(round(x/Dx*Nx)), int(round(y/Dy*Ny)) ]
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

# Production/injection
Q = np.zeros(N)
# injectors = rand(())
Q[xy2i(.7,.2)]  = .5
Q[xy2i(.2,.7)] = .5
Q[-1] = -1
# Q[0]  = +1

##
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
def GenA(Gridded,V,q):
    """Upwind finite-volume scheme."""
    fp=q.clip(max=0) # production
    XN=V.x.clip(max=0); x1=XN[:-1,:].ravel() # separate flux into
    YN=V.y.clip(max=0); y1=YN[:,:-1].ravel() # - flow in positive coordinate
    XP=V.x.clip(min=0); x2=XP[1:,:] .ravel() # - flow in negative coordinate
    YP=V.y.clip(min=0); y2=YP[:,1:] .ravel() #   direction (XN,YN)
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
    tx = 2*hy/hx; TX = np.zeros((Nx+1,Ny))
    ty = 2*hx/hy; TY = np.zeros((Nx,Ny+1))

    TX[1:-1,:] = tx/(L[0,:-1,:] + L[0,1:,:])
    TY[:,1:-1] = ty/(L[1,:,:-1] + L[1,:,1:])

    # Assemble TPFA discretization matrix.
    x1 = TX[:-1,:].ravel(); x2 = TX[1:,:].ravel()
    y1 = TY[:,:-1].ravel(); y2 = TY[:,1:].ravel()

    DiagVecs = [-x2, -y2, y1+y2+x1+x2, -y1, -x1]
    DiagIndx = [-Ny,  -1,         0  ,   1,  Ny]
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
def Pres(Gridded,S,Fluid,q):
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
def Upstream(Gridded,S,Fluid,V,q,T):
    """Explicit upwind finite-volume discretisation of CoM."""
    pv = h2*Gridded['por'].ravel() # pore volume=cell volume*porosity

    fi = q.clip(min=0)# inflow from wells

    XP=V.x.clip(min=0); XN=V.x.clip(max=0) # influx and outflux, x-faces
    YP=V.y.clip(min=0); YN=V.y.clip(max=0) # influx and outflux, y-faces

    Vi = XP[:-1,:]+YP[:,:-1]-\
         XN[ 1:,:]-YN[:, 1:] # each gridblock

    # Comppute dt
    from numpy import errstate
    with errstate(divide="ignore"):
        pm  = min(pv/(Vi.ravel()+fi)) # estimate of influx
    cfl = ((1-Fluid.swc-Fluid.sor)/3)*pm # CFL restriction # NB: 3-->2 since no z ?
    Nts = int(np.ceil(T/cfl)) # number of local time steps
    dtx = (T/Nts)/pv # local time steps

    # Discretized transport operator
    A=GenA(Gridded,V,q)           # system matrix
    A=sparse.spdiags(dtx,0,N,N)@A # A * dt/|Omega i|

    fi = q.clip(min=0)*dtx # injection

    for iT in range(1,Nts+1):
        mw,mo=RelPerm(S,Fluid)      # compute mobilities
        fw = mw/(mw+mo)             # compute fractional flow
        S = S + (A@fw + fi) # update saturation

    return S
##

plt.ion()
fig, ax = freshfig(1)

# Animation (i.e. external) time step
dt=0.025
##

@profile
def simulate(nSteps=28,plotting=True):
    S=np.zeros(N) # Initial saturation

    for iT in range(1,nSteps+1):
        [P,V]=Pres(Gridded,S,Fluid,Q) # pressure solver
        S=Upstream(Gridded,S,Fluid,V,Q,dt) # saturation solver

        # Plotting
        if plotting:
            if iT>1:
                for c in CC.collections:
                    ax.collections.remove(c)
            CC = ax.contourf(
                linspace(0,Dx-hx,Nx)+hx/2,
                linspace(0,Dy-hy,Ny)+hy/2,
                # Need to transpose coz contour() uses
                # the same orientation as array printing.
                S.reshape(gridshape).T,
                levels=linspace(0,1,11),
                vmin=0,vmax=1,
            )
            ax.set_title("Water saturation, t = %.1f"%(iT*dt))
            if iT==1:
                fig.colorbar(CC)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
            # ax.set_aspect("equal")
            plt.pause(.01)

    return P,V,S

if __name__ == "__main__":
    P,V,S = simulate(28)
    # P,V,S = simulate(1)
