"""Reservoir simulator: 2D, two-phase, immiscible, incompressible, using TPFA.

Based on Matlab codes from NTNU/Sintef:
http://folk.ntnu.no/andreas/papers/ResSimMatlab.pdf
Translated to python by Patrick N. Raanes.

Implemented with OOP so as to facilitate multiple realisations, by ensuring
that the parameter values of one instance do not influence another instance.
Depending on thread-safety, this might not be necessary, but is usually cleaner
when estimating anything other than the model's input/output (i.e. the state
variables).

Note: Index ordering/labels: `x` is 1st coord., `y` is 2nd.
See `grid.py` for more info.
"""

from functools import wraps

import numpy as np
import scipy.sparse as sparse
from numpy import errstate
from scipy.sparse.linalg import spsolve
# from scipy.sparse.linalg import cg
from struct_tools import DotDict, NicePrint

from simulator.grid import Grid2D
from tools import repeat

# TODO
# - Protect Nx, Ny, shape, etc?
# - Can the cell volumnes (h2) be arrays?


class ResSim(NicePrint, Grid2D):
    """Reservoir simulator.

    Example:
    >>> model = ResSim(Lx=1, Ly=1, Nx=32, Ny=32)
    >>> model.config_wells([[0, 0, 1]], [[1, 1, -1]])
    >>> water_sat0 = np.zeros(model.M)
    >>> saturation = repeat(model.step, 3, water_sat0, 0.025)
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
        """Scale production so as to equal injection.

        Otherwise, model will silently input deficit from SW corner.
        """

        def normalize_wellset(ww):
            ww = np.array(ww, float).T
            ww[0] *= self.Lx
            ww[1] *= self.Ly
            ww[2] /= ww[2].sum()
            return ww.T

        injectors = normalize_wellset(inj)
        producers = normalize_wellset(prod)

        def collocate(wells):
            """Place wells exactly on nodes."""
            for i in range(len(wells)):
                x, y, q = wells[i]
                wells[i, :2] = self.ind2xy(self.xy2ind(x, y))

        collocate(injectors)
        collocate(producers)

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


# Example run
if __name__ == '__main__':
    from matplotlib import pyplot as plt

    import simulator.plotting as plots
    from geostat import gaussian_fields

    model = ResSim(Lx=1, Ly=1, Nx=20, Ny=20)
    plots.model = model

    # Relative coordinates
    injectors = [[0.1, 0.0, 1.0], [0.9, 0.0, 1.0]]
    producers = [[0.1, 0.7, 100.0], [0.9, 1.0, 1.0], [.5, .2, 1]]
    model.config_wells(injectors, producers)

    # Create gridded field -- use e.g. for perm or saturation0
    np.random.seed(3000)
    surf = gaussian_fields(model.mesh(), 1)
    surf = 0.5 + .2*surf
    # surf = truncate_01(surf)
    # surf = sigmoid(surf)
    surf = surf.reshape(model.shape)
    # Insert barrier
    surf[:model.Nx//2, model.Ny//3] = 0.001
    # Set permeabilities to surf.
    model.Gridded.K = np.stack([surf, surf])

    # Define obs operator
    obs_inds = [model.xy2ind(x, y) for (x, y, _) in model.producers]
    def obs(saturation):  # noqa
        return [saturation[i] for i in obs_inds]

    # Simulate
    S0 = np.zeros(model.M)

    # dt=0.025 was used in Matlab code with 64x64 (and 1x1),
    # but I find that dt=0.1 works alright too.
    # With 32x32 I find that dt=0.2 works fine.
    # With 20x20 I find that dt=0.4 works fine.
    T = 28*0.025
    dt = 0.4
    nTime = round(T/dt)
    saturation, production = repeat(model.step, nTime, S0, dt, obs)

    # Animation
    plots.COORD_TYPE = "index"
    animation = plots.dashboard(surf, saturation, production, animate=False)
    plt.pause(.1)
