"""Reservoir simulator: 2D, two-phase, immiscible, incompressible, using TPFA.

Based on Matlab codes from NTNU/Sintef:
http://folk.ntnu.no/andreas/papers/ResSimMatlab.pdf
Translated to python by Patrick N. Raanes.

Originally this simulator/model was a set of functions,
related through shared module variables, which is perfectly pythonic,
and worked well when estimating only the simulator state (saturation field),
which is THE input/output (by definition of "state").
However, if we want to estimate other parameters,
e.g. the (fixed) permeability field,
we need to make copies (members) of the entire model.
=> Use OOP.

Note: Index ordering/labels: `x` is 1st coord., `y` is 2nd.
  This is hardcoded in the model code, in what takes place
  **between** `np.ravel` and `np.reshape` (using standard "C" ordering).
  It also means that the letters `x` and `y` tend to occur in alphabetic order.

Example:
>>> (Nx, Ny), (Dx, Dy) = (2, 5), (4, 10)
>>> X, Y = mesh_coords();
>>> X
array([[1., 1., 1., 1., 1.],
       [3., 3., 3., 3., 3.]])
>>> Y
array([[1., 3., 5., 7., 9.],
       [1., 3., 5., 7., 9.]])

>>> XY = np.stack((X, Y), axis=-1)
>>> xy2sub(*XY[1,3])
(1, 3)

>>> sub2xy(1,3) == XY[1,3]
array([ True,  True])
"""

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import scipy.linalg as sla
from matplotlib import pyplot as plt
from pylib.std import DotDict, suppress_w
from mpl_tools.misc import freshfig
from tqdm.auto import tqdm as progbar
from numpy import errstate

# __init__
Dx, Dy    = 1, 1  # lengths
Nx, Ny    = 32, 32   # num. of pts.
gridshape = Nx, Ny
grid      = Nx, Ny, Dx, Dy
M         = np.prod(gridshape)

# Resolution
hx, hy = Dx/Nx, Dy/Ny
h2 = hx*hy  # Cell volumes (could be array?)


def mesh_coords(centered=True):
    """Generate 2D coordinate grids."""

    xx = np.linspace(0, Dx, Nx, endpoint=False)
    yy = np.linspace(0, Dy, Ny, endpoint=False)

    if centered:
        xx += hx/2
        yy += hy/2

    return np.meshgrid(xx, yy, indexing="ij")


def sub2ind(ix, iy):
    """Convert index `(ix, iy)` to index in flattened array."""
    idx = np.ravel_multi_index((ix, iy), gridshape)
    return idx


def ind2sub(ind):
    """Inv. of `sub2ind`."""
    ix, iy = np.unravel_index(ind, gridshape)
    return ix, iy


def xy2sub(x, y):
    """Convert physical coordinate tuple to tuple `(ix, iy)`."""
    # ix = int(round(x/Dx*(Nx-1)))
    # iy = int(round(y/Dy*(Ny-1)))
    ix = (np.array(x) / Dx*(Nx-1)).round().astype(int)
    iy = (np.array(y) / Dy*(Ny-1)).round().astype(int)
    return ix, iy


def sub2xy(ix, iy):
    """Approximate inverse of `xy2sub`.

    Approx. because `xy2sub` aint injective, so we map to cell centres.
    """
    x = Dx * (ix + .5)/Nx
    y = Dy * (iy + .5)/Ny
    return x, y


def xy2ind(x, y):
    """Convert physical coordinates to flattened array index."""
    return sub2ind(*xy2sub(x, y))


def ind2xy(ind):
    """Inv. of `xy2ind`."""
    i, j = ind2sub(ind)
    x    = i/(Nx-1)*Dx
    y    = j/(Ny-1)*Dy
    return x, y


def truncate_01(E, warn=""):
    """Saturations should be between 0 and 1."""
    # assert E.max() <= 1 + 1e-10
    # assert E.min() >= 0 - 1e-10
    if (E.max() - 1e-10 >= 1) or (E.min() + 1e-10 <= 0):
        if warn:
            print(f"Warning -- {warn}: needed to truncate ensemble.")
        E = E.clip(0, 1)
    return E


Gridded = DotDict(
    K  =np.ones((2, *gridshape)),  # permeability in x&y dirs.
    por=np.ones(gridshape),        # porosity
)

Fluid = DotDict(
    vw=1.0,  vo=1.0,  # Viscosities
    swc=0.0, sor=0.0  # Irreducible saturations
)

injectors = producers = Q = None


def init_Q(inj, prod):
    # Globals (in python) are actually local to the module,
    # making them less dangerous.
    global injectors, producers, Q

    def normalize_wellset(ww):
        ww = np.array(ww, float).T
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
    for x, y, q in injectors:
        Q[xy2ind(x, y)] += q
    for x, y, q in producers:
        Q[xy2ind(x, y)] -= q

    assert np.isclose(Q.sum(), 0)
    return injectors, producers, Q


# np.random.seed(1)
# injectors = [[0,0,1]]
# producers = [[1,1,-1]]
injectors = [[0.1, 0.0, 1.0], [0.9, 0.0, 1.0]]
producers = [[0.1, 0.7, 1.0], [0.9, 1.0, 1.0], [.5, .2, 1]]
# injectors = rand(5,3)
# producers = rand(10,3)
init_Q(injectors, producers)


# Functions
def RelPerm(s, Fluid, nargout_is_4=False):
    """Rel. permeabilities of oil and water."""
    S = (s-Fluid.swc)/(1-Fluid.swc-Fluid.sor)  # Rescale saturations
    Mw = S**2/Fluid.vw  # Water mobility
    Mo = (1-S)**2/Fluid.vo  # Oil mobility
    # Derivatives:
    # dMw = 2*S/Fluid.vw/(1-Fluid.swc-Fluid.sor)
    # dMo = -2*(1-S)/Fluid.vo/(1-Fluid.swc-Fluid.sor)
    return Mw, Mo


def upwind_diff(Gridded, V, q):
    """Upwind finite-volume scheme."""
    fp = q.clip(max=0)  # production
    # Flow fluxes, separated into direction (x-y) and sign
    x1 = V.x.clip(max=0)[:-1, :].ravel()
    x2 = V.x.clip(min=0)[1:, :] .ravel()
    y1 = V.y.clip(max=0)[:, :-1].ravel()
    y2 = V.y.clip(min=0)[:, 1:] .ravel()
    # Compose flow matrix
    DiagVecs = [x2, y2, fp+y1-y2+x1-x2, -y1, -x1]  # diagonal vectors
    DiagIndx = [-Ny, -1,        0,  1,  Ny]        # diagonal index
    A = sparse.spdiags(DiagVecs, DiagIndx, M, M)   # matrix with upwind FV stencil
    return A


def TPFA(Gridded, K, q):
    """Two-point flux-approximation (TPFA) of Darcy:

    diffusion w/ nonlinear coefficient K."""
    # Compute transmissibilities by harmonic averaging.
    L = K**(-1)
    TX = np.zeros((Nx+1, Ny))
    TY = np.zeros((Nx, Ny+1))

    TX[1:-1, :] = 2*hy/hx/(L[0, :-1, :] + L[0, 1:, :])
    TY[:, 1:-1] = 2*hx/hy/(L[1, :, :-1] + L[1, :, 1:])

    # Assemble TPFA discretization matrix.
    x1 = TX[:-1, :].ravel()
    x2 = TX[1:, :] .ravel()
    y1 = TY[:, :-1].ravel()
    y2 = TY[:, 1:] .ravel()

    DiagVecs = [-x2, -y2, y1+y2+x1+x2, -y1, -x1]
    DiagIndx = [-Ny,  -1,      0,   1,  Ny]
    # Coerce system to be SPD (ref article, page 13).
    DiagVecs[2][0] += np.sum(Gridded.K[:, 0, 0])

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
    # Therefore we use spsolve, even though it
    # converts DIAgonal formats to CSC (and throws inefficiency warning).
    A = sparse.spdiags(DiagVecs, DiagIndx, M, M)
    with suppress_w(sparse.SparseEfficiencyWarning):
        u = spsolve(A, q)
    # The above is still much more efficient than going to full matrices,
    # indeed I get comparable speed to Matlab.
    # A = A.toarray()
    # u = np.linalg.solve(A, q)

    # Other options to consider: scipy.sparse.linalg.lsqr, etc.

    P = u.reshape(gridshape)

    V = DotDict(
        x = np.zeros((Nx+1, Ny)),
        y = np.zeros((Nx, Ny+1)),
    )
    V.x[1:-1, :] = (P[:-1, :] - P[1:, :]) * TX[1:-1, :]
    V.y[:, 1:-1] = (P[:, :-1] - P[:, 1:]) * TY[:, 1:-1]
    return P, V


def pressure_step(Gridded, S, Fluid, q):
    """TPFA finite-volume of Darcy: -nabla(K lambda(s) nabla(u)) = q."""
    # Compute K*lambda(S)
    Mw, Mo = RelPerm(S, Fluid)
    Mt = Mw+Mo
    Mt = Mt.reshape(gridshape)
    KM = Mt*Gridded.K
    # Compute pressure and extract fluxes
    [P, V] = TPFA(Gridded, KM, q)
    return P, V


def saturation_step(Gridded, S, Fluid, q, V, T):
    """Explicit upwind finite-volume discretisation of CoM."""
    pv = h2*Gridded['por'].ravel()  # pore volume=cell volume*porosity

    fi = q.clip(min=0)    # inflow from wells

    XP = V.x.clip(min=0)
    XN = V.x.clip(max=0)  # influx and outflux, x-faces
    YP = V.y.clip(min=0)
    YN = V.y.clip(max=0)  # influx and outflux, y-faces

    Vi = XP[:-1]-XN[1:]+YP[:, :-1]-YN[:, 1:]  # each gridblock

    # Compute dt
    with errstate(divide="ignore"):
        pm = min(pv/(Vi.ravel()+fi))      # estimate of influx
    cfl = ((1-Fluid.swc-Fluid.sor)/3)*pm  # CFL restriction NB: 3-->2 since no z-dim ?
    Nts = int(np.ceil(T/cfl))             # number of local time steps
    dtx = (T/Nts)/pv                      # local time steps

    # Discretized transport operator
    A = upwind_diff(Gridded, V, q)      # system matrix
    A = sparse.spdiags(dtx, 0, M, M)@A  # A * dt/|Omega i|

    for iT in range(Nts):
        mw, mo = RelPerm(S, Fluid)  # compute mobilities
        fw = mw/(mw+mo)             # compute fractional flow
        S = S + (A@fw + fi*dtx)     # update saturation

    return S


def step(S, dt):
    [P, V] =   pressure_step(Gridded, S, Fluid, Q)
    S      = saturation_step(Gridded, S, Fluid, Q, V, dt)
    return S


def obs(S):
    return [S[xy2ind(x, y)] for (x, y, _) in producers]


def simulate(nSteps, S, dt_ext=.025, pbar=True):
    saturation = np.zeros((nSteps,)+S.shape)
    production = np.zeros((nSteps, len(producers)))

    rge = np.arange(nSteps)
    if pbar:
        rge = progbar(rge)

    for iT in rge:
        S = step(S, dt_ext)

        saturation[iT] = S
        production[iT] = obs(S)

    return saturation, production


if __name__ == "__main__":
    from random_fields import gen_cov
    from numpy.random import randn
    import plots

    np.random.seed(3000)

    # Random field cov
    Cov = 0.3**2 * gen_cov(grid, radius=0.5)
    C12 = sla.sqrtm(Cov).real.T

    # IC (saturation)
    # surf = 0.5 + randn(M) @ C12
    # surf = truncate_01(surf)
    # S0 = surf
    # Constant
    S0 = np.zeros(M)

    # Varying grid params
    surf = 0.5 + 2*randn(M) @ C12
    surf = surf.clip(.01, 1)
    surf = surf.reshape(gridshape)

    # Rectangles
    i1, j1 = xy2sub(0, .4)
    i2, j2 = xy2sub(0.6, .45)
    surf[i1:i2, j1:j2] = 1e-3

    Gridded.K = np.stack([surf, surf])

    # Plot
    fig, (ax1, ax2) = freshfig(47, figsize=(8, 4), ncols=2)
    cc = plots.field(ax1, surf)
    # fig.colorbar(cc)
    ax2.hist(surf.ravel())

    # Simulate
    dt = 0.025
    nTime = 28
    saturation, production = simulate(nTime, S0, dt)
    # Plot
    ani = plots.animate1(saturation, production)
    plt.pause(.1)
