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
# import matplotlib as mpl
import scipy.sparse as sparse
from numpy import errstate
from pylib.dict_tools import DotDict, NicePrint
from pylib.std import suppress_w
from scipy.sparse.linalg import spsolve
from tqdm.auto import tqdm as progbar

# TODO
# - Monkey patch geometry functions
# - Rename Dx Lx
# - Protect Nx, Ny
# - Nx/Ny = 2
# - Can the cell volumnes (h2) be arrays?

class ResSim(NicePrint):
    """Reservoir simulator."""

    def __init__(self, Dx=1, Dy=1, Nx=32, Ny=32):
        self.Dx = Dx
        self.Dy = Dy
        self.Nx = Nx
        self.Ny = Ny

        self.gridshape = Nx, Ny
        self.grid      = Nx, Ny, Dx, Dy
        self.M         = np.prod(self.gridshape)

        # Resolution
        self.hx, self.hy = Dx/Nx, Dy/Ny
        self.h2 = self.hx*self.hy

        self.Gridded = DotDict(
            K  =np.ones((2, *self.gridshape)),  # permeability in x&y dirs.
            por=np.ones(self.gridshape),        # porosity
        )

        self.Fluid = DotDict(
            vw=1.0,  vo=1.0,  # Viscosities
            swc=0.0, sor=0.0  # Irreducible saturations
        )

    def init_Q(self, inj, prod):

        # Scale production so as to equal injection.
        # Otherwise, model will silently input deficit from SW corner.
        # producers[:,2] *= injectors[:,2].sum() / producers[:,2].sum()

        def normalize_wellset(ww):
            ww = np.array(ww, float).T
            ww[0] *= self.Dx
            ww[1] *= self.Dy
            ww[2] /= ww[2].sum()
            return ww.T

        injectors = normalize_wellset(inj)
        producers = normalize_wellset(prod)

        # Insert in source FIELD
        Q = np.zeros(self.M)
        for x, y, q in injectors:
            Q[self.xy2ind(x, y)] += q
        for x, y, q in producers:
            Q[self.xy2ind(x, y)] -= q
        assert np.isclose(Q.sum(), 0)

        self.Q = Q
        return injectors, producers

    def mesh_coords(self, centered=True):
        """Generate 2D coordinate grids."""

        xx = np.linspace(0, self.Dx, self.Nx, endpoint=False)
        yy = np.linspace(0, self.Dy, self.Ny, endpoint=False)

        if centered:
            xx += self.hx/2
            yy += self.hy/2

        return np.meshgrid(xx, yy, indexing="ij")

    def sub2ind(self, ix, iy):
        """Convert index `(ix, iy)` to index in flattened array."""
        idx = np.ravel_multi_index((ix, iy), self.gridshape)
        return idx

    def ind2sub(self, ind):
        """Inv. of `self.sub2ind`."""
        ix, iy = np.unravel_index(ind, self.gridshape)
        return ix, iy

    def xy2sub(self, x, y):
        """Convert physical coordinate tuple to tuple `(ix, iy)`."""
        # ix = int(round(x/self.Dx*(self.Nx-1)))
        # iy = int(round(y/self.Dy*(self.Ny-1)))
        ix = (np.array(x) / self.Dx*(self.Nx-1)).round().astype(int)
        iy = (np.array(y) / self.Dy*(self.Ny-1)).round().astype(int)
        return ix, iy

    def sub2xy(self, ix, iy):
        """Approximate inverse of `self.xy2sub`.

        Approx. because `self.xy2sub` aint injective, so we map to cell centres.
        """
        x = self.Dx * (ix + .5)/self.Nx
        y = self.Dy * (iy + .5)/self.Ny
        return x, y

    def xy2ind(self, x, y):
        """Convert physical coordinates to flattened array index."""
        return self.sub2ind(*self.xy2sub(x, y))

    def ind2xy(self, ind):
        """Inv. of `self.xy2ind`."""
        i, j = self.ind2sub(ind)
        x    = i/(self.Nx-1)*self.Dx
        y    = j/(self.Ny-1)*self.Dy
        return x, y

    def spdiags(self, data, diags, format=None):
        return sparse.spdiags(data, diags, self.M, self.M, format)

    def RelPerm(self, s, nargout_is_4=False):
        """Rel. permeabilities of oil and water."""
        Fluid = self.Fluid
        S = (s-Fluid.swc)/(1-Fluid.swc-Fluid.sor)  # Rescale saturations
        Mw = S**2/Fluid.vw  # Water mobility
        Mo = (1-S)**2/Fluid.vo  # Oil mobility
        # Derivatives:
        # dMw = 2*S/Fluid.vw/(1-Fluid.swc-Fluid.sor)
        # dMo = -2*(1-S)/Fluid.vo/(1-Fluid.swc-Fluid.sor)
        return Mw, Mo

    def upwind_diff(self, V, q):
        """Upwind finite-volume scheme."""
        fp = q.clip(max=0)  # production
        # Flow fluxes, separated into direction (x-y) and sign
        x1 = V.x.clip(max=0)[:-1, :].ravel()
        x2 = V.x.clip(min=0)[1:, :] .ravel()
        y1 = V.y.clip(max=0)[:, :-1].ravel()
        y2 = V.y.clip(min=0)[:, 1:] .ravel()
        # Compose flow matrix
        DiagVecs = [x2, y2, fp+y1-y2+x1-x2, -y1, -x1]      # diagonal vectors
        DiagIndx = [-self.Ny, -1,        0,  1,  self.Ny]  # diagonal index
        # Matrix with upwind FV stencil
        A = self.spdiags(DiagVecs, DiagIndx)
        return A

    def TPFA(self, K, q):
        """Two-point flux-approximation (TPFA) of Darcy:

        diffusion w/ nonlinear coefficient K."""
        # Compute transmissibilities by harmonic averaging.
        L = K**(-1)
        TX = np.zeros((self.Nx+1, self.Ny))
        TY = np.zeros((self.Nx,   self.Ny+1))

        TX[1:-1, :] = 2*self.hy/self.hx/(L[0, :-1, :] + L[0, 1:, :])
        TY[:, 1:-1] = 2*self.hx/self.hy/(L[1, :, :-1] + L[1, :, 1:])

        # Assemble TPFA discretization matrix.
        x1 = TX[:-1, :].ravel()
        x2 = TX[1:, :] .ravel()
        y1 = TY[:, :-1].ravel()
        y2 = TY[:, 1:] .ravel()

        DiagVecs = [-x2,      -y2, y1+y2+x1+x2, -y1,     -x1]
        DiagIndx = [-self.Ny,  -1,      0,   1,      self.Ny]
        # Coerce system to be SPD (ref article, page 13).
        DiagVecs[2][0] += np.sum(self.Gridded.K[:, 0, 0])

        # Solve linear system and extract interface fluxes.

        # Note on the matrix inversion:
        # We would like to use solve_banded (not solveh_banded),
        # despite it being somewhat convoluted
        # https://github.com/scipy/scipy/issues/2285
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
        A = self.spdiags(DiagVecs, DiagIndx)
        with suppress_w(sparse.SparseEfficiencyWarning):
            u = spsolve(A, q)
        # The above is still much more efficient than going to full matrices,
        # indeed I get comparable speed to Matlab.
        # A = A.toarray()
        # u = np.linalg.solve(A, q)

        # Other options to consider: scipy.sparse.linalg.lsqr, etc.

        P = u.reshape(self.gridshape)

        V = DotDict(
            x = np.zeros((self.Nx+1, self.Ny)),
            y = np.zeros((self.Nx,   self.Ny+1)),
        )
        V.x[1:-1, :] = (P[:-1, :] - P[1:, :]) * TX[1:-1, :]
        V.y[:, 1:-1] = (P[:, :-1] - P[:, 1:]) * TY[:, 1:-1]
        return P, V

    def pressure_step(self, S, q):
        """TPFA finite-volume of Darcy: -nabla(K lambda(s) nabla(u)) = q."""
        # Compute K*lambda(S)
        Mw, Mo = self.RelPerm(S)
        Mt = Mw+Mo
        Mt = Mt.reshape(self.gridshape)
        KM = Mt*self.Gridded.K
        # Compute pressure and extract fluxes
        [P, V] = self.TPFA(KM, q)
        return P, V

    def saturation_step(self, S, q, V, T):
        """Explicit upwind finite-volume discretisation of CoM."""
        pv = self.h2*self.Gridded['por'].ravel()  # pore volume=cell volume*porosity

        fi = q.clip(min=0)    # inflow from wells

        XP = V.x.clip(min=0)
        XN = V.x.clip(max=0)  # influx and outflux, x-faces
        YP = V.y.clip(min=0)
        YN = V.y.clip(max=0)  # influx and outflux, y-faces

        Vi = XP[:-1]-XN[1:]+YP[:, :-1]-YN[:, 1:]  # each gridblock

        # Compute dt
        with errstate(divide="ignore"):
            pm = min(pv/(Vi.ravel()+fi))      # estimate of influx
        sat = self.Fluid.swc + self.Fluid.sor
        # CFL restriction NB: 3-->2 since no z-dim ?
        cfl = ((1-sat)/3)*pm
        Nts = int(np.ceil(T/cfl))             # number of local time steps
        dtx = (T/Nts)/pv                      # local time steps

        # Discretized transport operator
        A = self.upwind_diff(V, q)           # system matrix
        A = self.spdiags(dtx, 0)@A           # A * dt/|Omega i|

        for iT in range(Nts):
            mw, mo = self.RelPerm(S)         # compute mobilities
            fw = mw/(mw+mo)                  # compute fractional flow
            S = S + (A@fw + fi*dtx)          # update saturation

        return S

    def step(self, S, dt):
        [P, V] = self.  pressure_step(S, self.Q)
        S      = self.saturation_step(S, self.Q, V, dt)
        return S


def truncate_01(self, E, warn=""):
    """Saturations should be between 0 and 1."""
    # assert E.max() <= 1 + 1e-10
    # assert E.min() >= 0 - 1e-10
    if (E.max() - 1e-10 >= 1) or (E.min() + 1e-10 <= 0):
        if warn:
            print(f"Warning -- {warn}: needed to truncate ensemble.")
        E = E.clip(0, 1)
    return E


def simulate(model_step, obs, nSteps, x0, dt=.025, pbar=True):

    # Range with or w/o progbar
    rge = np.arange(nSteps)
    if pbar:
        rge = progbar(rge)

    # Init
    xx = np.zeros((nSteps+1,)+x0.shape)
    yy = np.zeros((nSteps,)+(obs.length,))
    xx[0] = x0

    # Loop
    for iT in rge:
        xx[iT+1] = model_step(xx[iT], dt)
        yy[iT] = obs(xx[iT+1])

    return xx, yy

model = ResSim(Dx=1, Dy=1, Nx=32, Ny=32)

if __name__ == "__main__":
    # from matplotlib import pyplot as plt
    import scipy.linalg as sla

    # from mpl_tools.misc import fig_placement_load, freshfig, is_notebook_or_qt
    # import plots
    from random_fields import gen_cov
    from tools import sigmoid

    # injectors = [[0,0,1]]
    # producers = [[1,1,-1]]
    injectors = [[0.1, 0.0, 1.0], [0.9, 0.0, 1.0]]
    producers = [[0.1, 0.7, 1.0], [0.9, 1.0, 1.0], [.5, .2, 1]]
    # np.random.seed(1)
    # injectors = rand(5,3)
    # producers = rand(10,3)

    injectors, producers = model.init_Q(injectors, producers)

    # Random field cov
    np.random.seed(3000)
    Cov = 0.3**2 * gen_cov(model.grid, radius=0.5)
    C12 = sla.sqrtm(Cov).real.T
    S0 = np.zeros(model.M)

    # Varying grid params
    surf = 0.5 + 2*np.random.randn(model.M) @ C12
    # surf = surf.clip(.01, 1)
    surf = sigmoid(surf)
    surf = surf.reshape(model.gridshape)

    # Rectangles
    surf[:20, 10] = 0.01

    model.Gridded.K = np.stack([surf, surf])

    obs_inds = [model.xy2ind(x, y) for (x, y, _) in producers]

    def obs(saturation):
        return [saturation[i] for i in obs_inds]
    obs.length = len(producers)

    nTime = 28
    saturation, production = simulate(model.step, obs, nTime, S0, 0.025)
