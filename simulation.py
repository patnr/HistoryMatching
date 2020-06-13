"""Based on Matlab codes from NTNU/Sintef:

http://folk.ntnu.no/andreas/papers/ResSimMatlab.pdf

Translated to python by Patrick N. Raanes.
"""

# TODO: 1d case

from common import *
from res_gen import gen_ens

## Functions
def RelPerm(s,Fluid,nargout_is_4=False):
    """Rel. permeabilities of oil and water."""
    S = (s-Fluid.swc)/(1-Fluid.swc-Fluid.sor) # Rescale saturations
    Mw = S**2/Fluid.vw # Water mobility
    Mo =(1-S)**2/Fluid.vo # Oil mobility
    # Derivatives:
    # dMw = 2*S/Fluid.vw/(1-Fluid.swc-Fluid.sor)
    # dMo = -2*(1-S)/Fluid.vo/(1-Fluid.swc-Fluid.sor)
    return Mw,Mo

def upwind_diff(Gridded,V,q):
    """Upwind finite-volume scheme."""
    fp =   q.clip(max=0) # production
    x1 = V.x.clip(max=0)[:-1,:].ravel() # separate flux into
    y1 = V.y.clip(max=0)[:,:-1].ravel() # - flow in positive coordinate
    x2 = V.x.clip(min=0)[1:,:] .ravel() # - flow in negative coordinate
    y2 = V.y.clip(min=0)[:,1:] .ravel() #   direction (XN,YN)
    DiagVecs=[    x2, y2, fp+y1-y2+x1-x2, -y1, -x1] # diagonal vectors
    DiagIndx=[  -Ny , -1,        0      ,  1 ,  Ny] # diagonal index
    A=sparse.spdiags(DiagVecs,DiagIndx,M,M) # matrix with upwind FV stencil
    return A

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
    # ab = array([x for (x,M) in zip(DiagVecs,DiagIndx)])
    # u = solve_banded((3,3), ab, q, check_finite=False)

    # However, according to https://scicomp.stackexchange.com/a/30074/1740
    # solve_banded does not work well for when the band offsets large,
    # i.e. higher-dimensional problems.
    # Therefore we use sp.sparse.linalg.spsolve, even though it
    # converts DIAgonal formats to CSC (and throws inefficiency warning).
    A = sparse.spdiags(DiagVecs, DiagIndx, M, M)
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
    A=sparse.spdiags(dtx,0,M,M)@A # A * dt/|Omega i|

    for iT in range(1,Nts+1):
        mw,mo=RelPerm(S,Fluid)  # compute mobilities
        fw = mw/(mw+mo)         # compute fractional flow
        S = S + (A@fw + fi*dtx) # update saturation

    return S

@profile
def step(S,dt):
    [P,V] =   pressure_step(Gridded,S,Fluid,Q)
    S     = saturation_step(Gridded,S,Fluid,Q,V,dt)
    return S

def obs(S):
    return [S[xy2i(x,y)] for (x,y,_) in producers]

def simulate(nSteps,S,dt_ext=.025,dt_plot=0.01):
    saturation = []
    production = []

    for iT in 1+arange(nSteps):
        S = step(S,dt_ext)

        saturation += [S]
        production += [obs(S)]

        liveplot(S,iT*dt_ext,dt_plot)

    if dt_plot: del liveplot.ax
    return array(saturation), array(production)


def liveplot(S,t,dt_pause):

    # Exit if dt==0 or None
    if not dt_pause: return

    if not hasattr(liveplot,'ax'):
        # Init fig
        plt.ion()
        fig, ax = freshfig(1)
    else:
        # Clear plot
        ax = liveplot.ax
        for c in liveplot.CC.collections:
            ax.collections.remove(c)

    # Plot
    liveplot.CC = plot_field(ax, 1-S, vmin=0)

    # Adjust plot
    ax.set_title("Oil saturation, t = %.1f"%(t))
    if not hasattr(liveplot,'ax'):
        liveplot.ax = ax

        fig.colorbar(liveplot.CC)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.plot(*injectors.T[:2], "v", ms=16)
        ax.plot(*producers.T[:2], "^", ms=16)
        for i,w in enumerate(injectors):
            ax.text(w[0]-.01, w[1]-.01, 1+i, color="w")
        for i,w in enumerate(producers):
            ax.text(w[0]-.01, w[1]-.02, 1+i)
        # ax.set_aspect("equal")
    plt.pause(dt_pause)

def prod_plot(production, dt, nT, obs=None):
    fig, ax = freshfig(2)
    tt = dt*(1+arange(nT))
    hh = []
    for i,p in enumerate(1-production.T):
        hh += ax.plot(tt,p,"-",label=1+i)

    if obs is not None:
        for i,y in enumerate(1-obs.T):
            ax.plot(tt,y,"*",c=hh[i].get_color())

    ax.legend(title="(Production)\nwell num.")
    ax.set_ylabel("Oil saturation (rel. production)")
    ax.set_xlabel("Time")


def setup_wells(wells):
    wells = normalize_wellset(wells)
    # Add to source field
    Q = np.zeros(M)
    for x,y,q in wells: Q[xy2i(x,y)] = +q
    return wells, Q


## Global params. Available also when this module is imported.
np.random.seed(9)
injectors = rand((3,7)).T
producers = rand((3,5)).T

# injectors = [[0,0,1]]
# producers = [[1,1,-1]]

injectors, Qi = setup_wells(injectors)
producers, Qp = setup_wells(producers)
Q = Qi - Qp

if __name__ == "__main__":

    S0 = gen_ens(1,grid,0.7).squeeze()
    # S0 = np.zeros(M)

    dt = 0.025
    nT = 28
    saturation,production = simulate(nT,S0,dt,dt_plot=.01)

    prod_plot(production,dt,nT)
