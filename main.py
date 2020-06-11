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



# Profiling
import builtins
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
# TODO: orientation, ravel, etc

## Grid
Dx, Dy, Dz = 1,1,1     # Domain lengths
Nx, Ny, Nz = 64, 64, 1 # Domain points
gridshape = (Nx,Ny,Nz)
xy2i = lambda x,y: np.ravel_multi_index((x,y,0), gridshape, order='C')
N = np.prod(gridshape)

# Resolution
hx, hy, hz = Dx/Nx, Dy/Ny,  Dz/Nz
h3 = hx*hy*hz # Cell volumes (could be array?)

Gridded = Bunch(
    K  =np.ones((3,*gridshape)), # permeability
    por=np.ones(gridshape),   # porosity
)

Fluid = Bunch(
     vw=1.0,  vo=1.0,  # Viscosities
    swc=0.0, sor=0.0 # Irreducible saturations
)

# Production/injection
Q = np.zeros(N)
# injectors = rand(())
# Q[xy2i(0,40)]  = .5
Q[-1] = -1
Q[0]  = +1
# Q[xy2i(50,20)] = .5
# Q[xy2i(50,20)] = .5

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
    XN=V.x.clip(max=0); x1=XN[:-1,:,:].ravel(order="F") # separate flux into
    YN=V.y.clip(max=0); y1=YN[:,:-1,:].ravel(order="F") # - flow in positive coordinate
    ZN=V.z.clip(max=0); z1=ZN[:,:,:-1].ravel(order="F") #   direction (XP,YP,ZP)
    XP=V.x.clip(min=0); x2=XP[1:,:,:] .ravel(order="F") # - flow in negative coordinate
    YP=V.y.clip(min=0); y2=YP[:,1:,:] .ravel(order="F") #   direction (XN,YN,ZN)
    ZP=V.z.clip(min=0); z2=ZP[:,:,1:] .ravel(order="F") #
    DiagVecs=[    z2,  y2, x2, fp+x1-x2+y1-y2+z1-z2, -x1, -y1, -z1]   # diagonal vectors
    DiagIndx=[-Nx*Ny, -Nx, -1,           0         ,  1 ,  Nx, Nx*Ny] # diagonal index
    A=sparse.spdiags(DiagVecs,DiagIndx,N,N) # matrix with upwind FV stencil
    return A

@profile
def TPFA(Gridded,K,q):
    """Two-point flux-approximation (TPFA) of Darcy:

    diffusion w/ nonlinear coefficient K."""
    # Compute transmissibilities by harmonic averaging.
    L = K**(-1)
    tx = 2*hy*hz/hx; TX = np.zeros((Nx+1,Ny,Nz))
    ty = 2*hx*hz/hy; TY = np.zeros((Nx,Ny+1,Nz))
    tz = 2*hx*hy/hz; TZ = np.zeros((Nx,Ny,Nz+1))

    TX[1:-1,:,:] = tx/(L[0,:-1,:,:] + L[0,1:,:,:])
    TY[:,1:-1,:] = ty/(L[1,:,:-1,:] + L[1,:,1:,:])
    TZ[:,:,1:-1] = tz/(L[2,:,:,:-1] + L[2,:,:,1:])

    # Assemble TPFA discretization matrix.
    x1 = TX[:-1,:,:].ravel(order="F"); x2 = TX[1:,:,:].ravel(order="F")
    y1 = TY[:,:-1,:].ravel(order="F"); y2 = TY[:,1:,:].ravel(order="F")
    z1 = TZ[:,:,:-1].ravel(order="F"); z2 = TZ[:,:,1:].ravel(order="F")

    DiagVecs = [   -z2, -y2, -x2, x1+x2+y1+y2+z1+z2, -x1, -y1,  -z1]
    DiagIndx = [-Nx*Ny, -Nx,  -1,         0        ,   1,  Nx, Nx*Ny]
    DiagVecs[3][0] += np.sum(Gridded.K[:,0,0,0])

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

    P = u.reshape(gridshape,order="F")

    V = Bunch(
        x = np.zeros((Nx+1,Ny,Nz)),
        y = np.zeros((Nx,Ny+1,Nz)),
        z = np.zeros((Nx,Ny,Nz+1)),
    )
    V.x[1:-1,:,:] = (P[:-1,:,:] - P[1:,:,:]) * TX[1:-1,:,:]
    V.y[:,1:-1,:] = (P[:,:-1,:] - P[:,1:,:]) * TY[:,1:-1,:]
    V.z[:,:,1:-1] = (P[:,:,:-1] - P[:,:,1:]) * TZ[:,:,1:-1]
    return P,V

@profile
def Pres(Gridded,S,Fluid,q):
    """TPFA finite-volume of Darcy: -nabla(K lambda(s) nabla(u)) = q."""
    # Compute K*lambda(S)
    Mw,Mo = RelPerm(S,Fluid)
    Mt = Mw+Mo
    Mt = Mt.reshape(gridshape,order="F")
    KM = Mt*Gridded.K
    # Compute pressure and extract fluxes
    [P,V]=TPFA(Gridded,KM,q)
    return P, V

@profile
def Upstream(Gridded,S,Fluid,V,q,T):
    """Explicit upwind finite-volume discretisation of CoM."""
    pv = h3*Gridded['por'].ravel(order="F") # pore volume=cell volume*porosity

    fi = q.clip(min=0)# inflow from wells

    XP=V.x.clip(min=0); XN=V.x.clip(max=0) # influx and outflux, x-faces
    YP=V.y.clip(min=0); YN=V.y.clip(max=0) # influx and outflux, y-faces
    ZP=V.z.clip(min=0); ZN=V.z.clip(max=0) # influx and outflux, z-faces

    Vi = XP[:-1,:,:]+YP[:,:-1,:]+ZP[:,:,:-1]- \
         XN[ 1:,:,:]-YN[:, 1:,:]-ZN[:,:, 1:] # each gridblock

    # Comppute dt
    pm = min(pv/(Vi.ravel(order="F")+fi)) # estimate of influx
    cfl = ((1-Fluid.swc-Fluid.sor)/3)*pm # CFL restriction
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

##

@profile
def main(nSteps=28,plotting=True):
    S=np.zeros(N) # Initial saturation

    dt=0.7/nSteps

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
                S.reshape(gridshape,order="F")[:,:,0],
                levels=linspace(0,1,11),
                vmin=0,vmax=1,
            )
            ax.set_title("Water saturation, t = %.1f"%(iT*dt))
            if iT==1:
                fig.colorbar(CC)
            # ax.set_aspect("equal")
            plt.pause(.01)

    return P,V,S

if __name__ == "__main__":
    P,V,S = main()
