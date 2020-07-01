"""Based on Matlab codes from NTNU/Sintef:

http://folk.ntnu.no/andreas/papers/ResSimMatlab.pdf

Translated to python by Patrick N. Raanes.

Programming choices:

- Classes vs modules:
  https://stackoverflow.com/a/600201/38281
"""

from common import *

## __init__
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
ind2sub = lambda ind: np.unravel_index(ind, gridshape)
def ind2xy(ind):
    i,j = ind2sub(ind)
    x   = i/(Nx-1)*Dx
    y   = j/(Ny-1)*Dy
    return x,y

# Resolution
hx, hy = Dx/Nx, Dy/Ny
h2 = hx*hy # Cell volumes (could be array?)

Gridded = DotDict(
      K=np.ones((2,*gridshape)), # permeability in x&y dirs.
    por=np.ones(gridshape),      # porosity
)

Fluid = DotDict(
     vw=1.0,  vo=1.0, # Viscosities
    swc=0.0, sor=0.0  # Irreducible saturations
)

def init_Q(inj,prod):
    # Globals (in python) are actually local to the module,
    # making them less dangerous.
    global injectors, producers, Q

    def normalize_wellset(ww):
        ww = array(ww,float).T
        ww[0] *= Dx
        ww[1] *= Dy
        ww[2] /= ww[2].sum()
        return ww.T

    injectors = normalize_wellset(inj)
    producers = normalize_wellset(prod)

    # Scale production so as to equal injection.
    # Otherwise, model will silently input deficit from SW corner.
    # producers[:,2] *= injectors[:,2].sum() / producers[:,2].sum()

    # Insert in source FIELD
    Q = np.zeros(M)
    for x,y,q in injectors: Q[xy2i(x,y)] += q
    for x,y,q in producers: Q[xy2i(x,y)] -= q

    assert np.isclose(Q.sum(),0)
    return injectors, producers, Q


# np.random.seed(1)
# injectors = [[0,0,1]]
# producers = [[1,1,-1]]
injectors = [ [0.1, 0.0, 1.0], [0.9, 0.0, 1.0] ]
producers = [ [0.1, 0.7, 1.0], [0.9, 1.0, 1.0] , [.5,.2,1]]
# injectors = rand((5,3))
# producers = rand((10,3))
init_Q(injectors, producers)





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
    # Attempt:
    # ab = array([x for (x,M) in zip(DiagVecs,DiagIndx)])
    # u = solve_banded((3,3), ab, q, check_finite=False)
    # However, according to https://scicomp.stackexchange.com/a/30074/1740
    # solve_banded does not work well for when the band offsets large,
    # i.e. higher-dimensional problems.
    # Therefore we use sp.sparse.linalg.spsolve, even though it
    # converts DIAgonal formats to CSC (and throws inefficiency warning).
    A = sparse.spdiags(DiagVecs, DiagIndx, M, M)
    with suppress_w(sparse.SparseEfficiencyWarning):
        u = spsolve(A,q)
    # The above is still much more efficient than going to full matrices,
    # indeed I get comparable speed to Matlab.
    # A = A.toarray()
    # u = np.linalg.solve(A, q)

    # Other options to consider: scipy.sparse.linalg.lsqr, etc.

    P = u.reshape(gridshape)

    V = DotDict(
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
    cfl = ((1-Fluid.swc-Fluid.sor)/3)*pm # CFL restriction # NB: 3-->2 since no z-dim ?
    Nts = int(np.ceil(T/cfl)) # number of local time steps
    dtx = (T/Nts)/pv # local time steps

    # Discretized transport operator
    A=upwind_diff(Gridded,V,q)     # system matrix
    A=sparse.spdiags(dtx,0,M,M)@A # A * dt/|Omega i|

    for iT in range(Nts):
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

def simulate(nSteps,S,dt_ext=.025):
    saturation = np.zeros((nSteps,)+S.shape)
    production = np.zeros((nSteps,len(producers)))

    for iT in arange(nSteps):
        S = step(S,dt_ext)

        saturation[iT] = S
        production[iT] = obs(S)

    return saturation, production




if __name__ == "__main__":
    # ICs

    # Gen. random field
    # np.random.seed(1)
    from random_fields import gen_cov
    Cov = 0.3**2 * gen_cov(grid, radius=0.5)
    C12 = sla.sqrtm(Cov).real.T
    surf  = 0.5 + randn(M) @ C12
    surf  = truncate_01(surf)

    # IC saturation
    # Varying 
    # S0 = surf
    # Constant
    S0 = np.zeros(M)

    # Varying permeability
    surf  = 0.5 + 2*randn(M) @ C12
    surf = surf.clip(.01,1)
    surf = surf.reshape(gridshape)
    Gridded.K = np.stack([surf,surf])

    fig, (ax1,ax2) = freshfig(47,figsize=(8,4), ncols=2)
    cc = ax1.contourf(surf)
    fig.colorbar(cc)
    ax2.hist(surf.ravel())


    dt = 0.025
    nTime = 28
    saturation,production = simulate(nTime,S0,dt)

    from plots import animate1
    ani = animate1(saturation,production)
    plt.show(block=False)
