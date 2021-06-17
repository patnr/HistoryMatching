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
See grid.py for more info.
"""

from functools import wraps

import numpy as np
# import matplotlib as mpl
import scipy.sparse as sparse
from numpy import errstate
from scipy.sparse.linalg import spsolve
# from scipy.sparse.linalg import cg
from struct_tools import DotDict, NicePrint
from tqdm.auto import tqdm as progbar

from .grid import Grid2D

# TODO
# - Protect Nx, Ny, shape, etc?
# - Can the cell volumnes (h2) be arrays?


class ResSim(NicePrint, Grid2D):
    """Reservoir simulator.

    Example:
    >>> model = ResSim(Lx=1, Ly=1, Nx=32, Ny=32)
    >>> model.config_wells([[0, 0, 1]], [[1, 1, -1]])
    >>> S0 = np.zeros(model.M)
    >>> saturation = simulate(model.step, 3, S0, 0.025)
    >>> saturation[-1, :3]
    array([0.9884098 , 0.97347222, 0.95294563])
    """

    @wraps(Grid2D.__init__)
    def __init__(self, *args, **kwargs):

        # Init grid
        super().__init__(*args, **kwargs)

        # Gridded properties
        self.Gridded = DotDict(
            K  =np.ones((2, *self.shape)),  # permeability in x&y dirs.
            por=np.ones(self.shape),        # porosity
        )

        self.Fluid = DotDict(
            vw=1.0, vo=1.0,  # Viscosities
            swc=0.0, sor=0.0,  # Irreducible saturations
        )

    def config_wells(self, inj, prod):

        # Scale production so as to equal injection.
        # Otherwise, model will silently input deficit from SW corner.
        # producers[:,2] *= injectors[:,2].sum() / producers[:,2].sum()

        def normalize_wellset(ww):
            ww = np.array(ww, float).T
            ww[0] *= self.Lx
            ww[1] *= self.Ly
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
        self.injectors = injectors
        self.producers = producers

    def spdiags(self, data, diags):
        return sparse.spdiags(data, diags, self.M, self.M)

    def RelPerm(self, s):
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
        DiagVecs = [x2, y2, fp+y1-y2+x1-x2, -y1, -x1]      # noqa diagonal vectors
        DiagIndx = [-self.Ny, -1,        0,  1,  self.Ny]  # noqa diagonal index
        # Matrix with upwind FV stencil
        A = self.spdiags(DiagVecs, DiagIndx)
        return A

    def TPFA(self, K, q):
        """Two-point flux-approximation (TPFA) of Darcy:

        diffusion w/ nonlinear coefficient K.
        """
        # Compute transmissibilities by harmonic averaging.
        L = K**(-1)
        TX = np.zeros((self.Nx+1, self.Ny))
        TY = np.zeros((self.Nx,   self.Ny+1))  # noqa

        TX[1:-1, :] = 2*self.hy/self.hx/(L[0, :-1, :] + L[0, 1:, :])
        TY[:, 1:-1] = 2*self.hx/self.hy/(L[1, :, :-1] + L[1, :, 1:])

        # Assemble TPFA discretization matrix.
        x1 = TX[:-1, :].ravel()
        x2 = TX[1:, :] .ravel()
        y1 = TY[:, :-1].ravel()
        y2 = TY[:, 1:] .ravel()

        # Setup linear system
        DiagVecs = [-x2,      -y2, y1+y2+x1+x2, -y1,     -x1]  # noqa
        DiagIndx = [-self.Ny,  -1,      0,   1,      self.Ny]  # noqa
        # Coerce system to be SPD (ref article, page 13).
        DiagVecs[2][0] += np.sum(self.Gridded.K[:, 0, 0])
        A = self.spdiags(DiagVecs, DiagIndx)

        # Solve
        # u = np.linalg.solve(A.A, q)  # direct dense solver
        u = spsolve(A.tocsr(), q)  # direct sparse solver
        # u, _info = cg(A, q)  # conjugate gradient
        # Could also try scipy.linalg.solveh_banded which, according to
        # https://scicomp.stackexchange.com/a/30074 uses the Thomas algorithm,
        # as recommended by Aziz and Settari ("Petro. Res. simulation").
        # NB: stackexchange also mentions that solve_banded does not work well
        # when the band offsets large, i.e. higher-dimensional problems.

        # Extract fluxes
        P = u.reshape(self.shape)
        V = DotDict(
            x = np.zeros((self.Nx+1, self.Ny)),
            y = np.zeros((self.Nx,   self.Ny+1)),  # noqa
        )
        V.x[1:-1, :] = (P[:-1, :] - P[1:, :]) * TX[1:-1, :]
        V.y[:, 1:-1] = (P[:, :-1] - P[:, 1:]) * TY[:, 1:-1]
        return P, V

    def pressure_step(self, S, q):
        """TPFA finite-volume of Darcy: -nabla(K lambda(s) nabla(u)) = q."""
        # Compute K*lambda(S)
        Mw, Mo = self.RelPerm(S)
        Mt = Mw+Mo
        Mt = Mt.reshape(self.shape)
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

        for _iT in range(Nts):
            mw, mo = self.RelPerm(S)         # compute mobilities
            fw = mw/(mw+mo)                  # compute fractional flow
            S = S + (A@fw + fi*dtx)          # update saturation

        return S

    def step(self, S, dt):
        [P, V] = self.  pressure_step(S, self.Q)
        S      = self.saturation_step(S, self.Q, V, dt)
        return S


def simulate(model_step, nSteps, x0, dt=.025, obs=None, pbar=True):

    # Range with or w/o progbar
    rge = np.arange(nSteps)
    if pbar:
        rge = progbar(rge, "Simulation")

    # Init
    xx = np.zeros((nSteps+1,)+x0.shape)
    xx[0] = x0

    # Step
    for iT in rge:
        xx[iT+1] = model_step(xx[iT], dt)

    if obs is None:
        return xx

    # Observe
    yy = np.zeros((nSteps,)+(obs.length,))
    for iT in rge:
        yy[iT] = obs(xx[iT+1])
    return xx, yy
