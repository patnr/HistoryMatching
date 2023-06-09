# -*- coding: utf-8 -*-
# # Production optimisation with ensembles – an interactive tutorial
#
# Copyright Patrick N. Raanes, NORCE, 2023.
#
# This is a self-contained tutorial on production optimisation using ensemble methods.
# - Please have a look at the [history matching (HM) tutorial](MAIN.ipynb)
#   for an introduction to Python, Jupyter notebooks, and this reservoir simulator.

# If you're on **Google Colab**, run the cell below to install the requirements.
# Otherwise (and assuming you have done the installation described in the README),
# you can skip/delete this cell.

## TODO:
# - Use Bezier curves to parametrize well rates ?
# - Make user widget for manual optimisation
# - 1D: Plot ensemble of npv curves
# - Plot ensemble of pdfs of npvs, including
#   strategies: reactive control, nominal optimization, robust optimization
# - Use model with many injectors, producers ? jansen2010closed_prez
# - Cite essen, jansen, chen, fonseca, stordal, raanes

## Imports
remote = "https://raw.githubusercontent.com/patnr/HistoryMatching"
# !wget -qO- {remote}/master/colab_bootstrap.sh | bash -s

from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
# from tools import mpl_setup
# mpl_setup.init()
# from tqdm.auto import tqdm
plt.ion()
np.set_printoptions(precision=6)

import copy
import numpy.random as rnd
import scipy.linalg as sla
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec
# from matplotlib.ticker import LogLocator
from mpl_tools.place import freshfig
# from numpy import sqrt
from struct_tools import DotDict as Dict
# from tqdm.auto import tqdm as progbar

import TPFA_ResSim as simulator
import tools.plotting as plotting
# import tools.localization as loc
from tools import geostat, utils
from tools.utils import apply
# from tools.utils import center


## Aux
def ticklabels_inside(ax, axis='y', al=None, **kwargs):
    """Move ticks -- and their labels -- inside panel. `kwargs` could be e.g. `labelsize`."""
    ax.tick_params(axis=axis, which='both', direction='in', pad=-4, **kwargs)
    if axis == 'y':
        plt.setp(ax.get_yticklabels(), ha=(al or "left"))
    else:
        plt.setp(ax.get_xticklabels(), ha="right", va=(al or "bottom"))


def fig3(*args, figsize=(11, 3), **kwargs):
    fig, _ax = freshfig(*args, figsize=figsize, **kwargs)
    _ax.remove()
    gs = GridSpec(2, 2)
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])
    for ax in (ax1, ax2):
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
    return fig, (ax0, ax1, ax2)


def final_sweep(title="", **kwargs):
    model = model_setup(**kwargs)
    wsats, prods = sim(model, wsat0)
    plotting.single.model = model
    fig, ax = freshfig(f"Final sweep {title}", figsize=(1, .6), rel=True)
    plotting.single.field(ax, wsats[-1], "oil", wells=True, colorbar=True)
    fig.tight_layout()


def plot_path(ax1, ax2, ax3, path, objs=None, color=None):
    # Plot x0
    ax1.plot(*path[0, :2], c=color or 'g', ms=3**2, marker='o')
    # Path line
    ax1.plot(*path.T[:2], c=color or "g")
    # Path scatter and text
    if len(path) >= 2:
        ii = np.logspace(0, np.log10(len(path) - 1),
                         15, endpoint=True, dtype=int)
    else:
        ii = [0]
    cm = plt.get_cmap('viridis_r')(np.linspace(0.0, 1, len(ii)))
    for k, c in zip(ii, cm):
        if color:
            c = color
        x = path[k][:2]
        ax1.text(*x, k, c=c)
        ax1.scatter(*x, s=4**2, color=c, zorder=5)

    if objs is not None:
        # Objective values
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.set_ylabel('Obj')
        ax2.plot(objs, color=color or 'C0', marker='.', ms=3**2)
        # ax2.set_ylim(-objs.max(), -objs.min())
        ax2.grid()

        # Step magnitudes
        ax3.sharex(ax2)
        ax2.tick_params(labelbottom=False)
        xx = np.arange(len(path)-1) + .5
        yy = nnorm(np.diff(path, axis=0), -1)
        ax3.plot(xx, yy, color=color or "C1", marker='.', ms=3**2)
        ax3.set_ylabel('|Step|')
        ax3.grid()
        ax3.set(xlabel="itr")


def atleast_2d(x):
    """Ensure has ens axis."""
    singleton = np.ndim(x) == 1
    x = np.atleast_2d(x)
    return x, singleton


# Could put this in npv() rather than copying the base model
# but it's useful to have a base model in the global namespace
# for plotting purposes.
perm = Dict()

def sample_prior_perm(N):
    lperms = geostat.gaussian_fields(model.mesh, N, r=0.8)
    return lperms

def perm_transf(x):
    return .1 + np.exp(5*x)

plotting.styles["pperm"]["levels"] = np.linspace(-4, 4, 21)
plotting.styles["pperm"]["cticks"] = np.arange(-4, 4+1)

def set_perm(model, log_perm_array):
    """Set perm. in model code (both x and y components)."""
    p = perm_transf(log_perm_array)
    p = p.reshape(model.shape)
    model.Gridded.K = np.stack([p, p])


def nnorm(x, axis=0):
    """L2 norm. Uses `mean` -- not `sum` (as in `nla.norm`)."""
    return np.sqrt(np.mean(x*x, axis))


def cntr(xx):
    return xx - xx.mean(0)


def rinv(A, reg, tikh=True, nMax=None):
    """Reproduces `sla.pinv(..., rtol=reg)` for `tikh=False`."""
    # Decompose
    U, s, VT = sla.svd(A, full_matrices=False)

    # "Relativize" the regularisation param
    reg = reg * s[0]

    # Compute inverse (regularized or truncated)
    if tikh:
        s1 = s / (s**2 + reg**2)
    else:
        s0 = s >= reg
        s1 = np.zeros_like(s)
        s1[s0] = 1/s[s0]

    if nMax:
        s1[nMax:] = 0

    # Re-compose
    return (VT.T * s1) @ U.T


@dataclass
class Momentum:
    """Gradient momentum (provides history memorisation)."""
    # Note: I also tried rolling average with a maxlen deque
    #       but the results of cursory testing were slightly worse.
    b: float = 0.9

    def update(self, v):
        if not hasattr(self, 'val'):
            self.reset()
        self.val *= self.b
        self.val += (1-self.b) * v

    def reset(self, val=0):
        self.val = val


def EnGrad(obj, u, chol, precond=False):
    U = cntr(rnd.randn(N, len(u)) @ chol.T)
    J = cntr(obj(u + U))
    if precond:
        g = U.T @ J / (N-1)
    else:
        g = rinv(U, reg=.1, tikh=True) @ J
    return g


## EnOpt
def EnOpt(obj, u, chol, sign=+1,
    # Step modifiers:
    regulator=None, xSteps=(1,), normed=True, precond=True,
    # Stopping criteria:
    nIter=100, rtol=1e-4):
    """Gradient/steepest *descent* using ensemble (LLS) gradient and backtracking.

    - `rtol` specified how large an improvement is required to update the iterate.
      Large values makes backtracking more reluctant to accept an update,
      resulting in *faster* declaration of convergence.
      Setting to 0 is not recommended, because if the objective function is flat
      in the neighborhood, then the path could just go in circles on that flat.
    """

    def backtrack(base_step):
        """Line search by bisection."""
        for i, xStep in enumerate(xSteps):
            x = path[-1] + sign * xStep * base_step
            J = obj(x)
            if sign*(J - objs[-1]) > atol:
                return x, J, i

    # Init
    if regulator:
        regulator.reset()
    J = obj(u)
    atol = max(1e-8, abs(J)) * rtol
    info, path, objs = {}, [], []

    for itr in range(nIter):
        path.append(u)
        objs.append(J)

        # Compute search direction
        grad = EnGrad(obj, u, chol, precond=precond)
        if normed:
            grad /= nnorm(grad)
        if regulator:
            regulator.update(grad)

        # Update iterate
        if not regulator or not (updated := backtrack(regulator.val)):
            if regulator:
                regulator.reset(grad)
                info.setdefault('resets', []).append(itr)
            # Fallback to pure grad
            if not (updated := backtrack(grad)):
                # Stop if lower J not found
                status = "Converged ✅"
                break
        u, J, i = updated
        info.setdefault('nDeclined', []).append(i)

    else:
        status = "Ran out of iters ❌"

    print(f"{status:<9} {itr=:<5}  {path[-1]=}  {objs[-1]=:.2f}")
    return np.array(path), np.array(objs), info


# ## Model case
def model_setup(**kwargs):
    """Create new model, based on `globals()['model']`."""
    # Init
    modln = copy.deepcopy(model)  # dont overwrite
    # pperm gets special treatment (transformation)
    set_perm(modln, kwargs.pop('perm', pperm))
    # Set other attrs
    for key, val in kwargs.items():
        setattr(modln, key, val)
    # Sanitize
    modln.config_wells(modln.inj_xy, modln.inj_rates,
                       modln.prod_xy, modln.prod_rates)
    return modln


model = simulator.ResSim(Nx=20, Ny=20, Lx=2, Ly=1)
seed = rnd.seed(3)
pperm = sample_prior_perm(1)

xy_4corners = np.dstack(np.meshgrid(
    np.array([.12, .87]) * model.Lx,
    np.array([.12, .87]) * model.Ly
)).reshape((-1, 2))

r0 = 1.5
model = model_setup(
    inj_xy = np.array([model.domain[1]]) / 2,
    inj_rates = r0 * np.ones((1, 1)) / 1,
    prod_rates = r0 * np.ones((4, 1)) / 4,
    prod_xy = xy_4corners,
)
plotting.single.model = model
plotting.single.coord_type = "absolute"


# Simulation
wsat0 = np.zeros(model.Nxy)
T = 1
dt = 0.025
nTime = round(T/dt)

# Like `forward_model`, but w/o setting params
def sim(model, wsat, pbar=False, leave=False):
    """Simulate reservoir."""
    integrator = model.time_stepper(dt)
    wsats = simulator.recurse(integrator, nTime, wsat, pbar=pbar, leave=leave)
    # Extract production time series from water saturation fields
    wells = model.xy2ind(*model.prod_xy.T)
    prods = np.array([wsat[wells] for wsat in wsats])
    return wsats, prods


## Objectives
def npv(**kwargs):
    """Net present value (NPV, i.e. discounted, total oil production) of model config."""
    # Config
    try:
        model = model_setup(**kwargs)
        # Simulate
        wsats, prods = sim(model, wsat0)
    except Exception:
        return 0  # Invalid model params. Penalize. Use `raise` for debugging.
    # Compute "monetary" value
    discounts = .99 ** np.arange(nTime + 1)
    prods = 1 - prods                  # water --> oil
    prods = prods * model.prod_rates.T # volume = saturation * rate
    prods = np.sum(prods, -1)          # sum over wells
    value = prods @ discounts          # sum in time, incld. discount factors
    # Compute cost of water injection
    # PS: We don't bother with cost of water production,
    # since it is implicitly approximated by reduction in oil production.
    inj_rates = model.inj_rates
    if inj_rates.shape[1] == 1:
        inj_rates = np.tile(inj_rates, (1, nTime))
    cost = np.sum(inj_rates, 0)
    cost = cost @ discounts[:-1]
    return value - .4*cost


def npv_in_rates(inj_rates):
    """`npv(inj_rates)`. Input shape `(nEns, nInj)`."""
    inj_rates, singleton = atleast_2d(inj_rates)
    inj_rates = inj_rates.reshape((len(inj_rates), -1, 1))  # (nEns, nInj) --> (nEns, nInj, 1)
    total_rate = np.sum(inj_rates, axis=1).squeeze() # (nEns,)
    prod_rates = 1/4 * (np.ones((1, 4, len(inj_rates))) * total_rate).T
    Js = apply(npv, inj_rates=inj_rates, prod_rates=prod_rates, unzip=False)
    return Js[0] if singleton else Js

def npv_in_injectors(xys):
    """`npv(inj_xy)`. Input shape `(nEns, 2*nInj)`."""
    xys, singleton = atleast_2d(xys)
    xys = xys.reshape((len(xys), -1, 2))  # (nEns, 2*nInj) --> (nEns, nInj, 2)
    Js = apply(npv, inj_xy=xys, unzip=False)
    return Js[0] if singleton else Js


def npv_in_x_of_inj0_with_fixed_y(x):
    """Like `npv_in_injectors` but with `y` fixed. Input shape `(nEns, 1)` or `(1,)`."""
    xs = x
    ys = y * np.ones_like(xs)
    xys = np.hstack([xs, ys])
    Js = npv_in_injectors(xys)
    return Js

def npv_in_injectors_transformed(xys):
    """Like `npv_in_injectors` but with transformation of (x, y)."""
    xys = transform_xys(xys)
    Js = npv_in_injectors(xys)
    return Js

def transform_xys(xys):
    """Transform infinite plane to `(0, Lx) x (0, Ly)`."""
    xys = np.array(xys, dtype=float)

    def realline_to_0L(x, L, compress=1):
        sigmoid = lambda z: 1/(1 + np.exp(-z))
        x = (x - L/2) * L * compress
        return L * sigmoid(x)

    # Loop over (x, y)
    for i0, L in zip([0, 1], model.domain[1]):
        ii = slice(i0, None, 2)
        xys[..., ii] = realline_to_0L(xys[..., ii], L, 1)
    return xys


## Optim params
N = 10
xSteps = [.4 * 1/2**i for i in range(8)]
utils.nCPU = True


if False:
    ## Plot obj_inj_x
    # Make pairs of x-values slightly on each side of cell borders
    d2 = model.hx/2 - 1e-8
    xx = model.mesh[0][:, 0]
    xx = np.ravel((xx - d2, xx + d2), order='F')

    y = model.Ly/2
    npvs = npv_in_x_of_inj0_with_fixed_y(xx[:, None])
    fig, ax = freshfig(f"npv_in_x_of_inj0_with_fixed_y ({y})", figsize=(7, 3))
    ax.plot(xx, npvs, "slategrey", lw=3)
    ax.set(xlabel="x", ylabel="NPV")
    fig.tight_layout()

    ## Optimize obj_inj_x
    L = .3 * np.eye(1)
    shifts = {}
    for i, u0 in enumerate(model.Lx * np.array([[.05, .1, .2, .8, .9, .95]]).T):
        path, objs, info = EnOpt(npv_in_x_of_inj0_with_fixed_y, u0, L,
                                 regulator=Momentum(0), xSteps=xSteps)
        shift = -.3*i  # for visual distinction
        ax.plot(path, objs + shift, '-o', c=f'C{i+1}')


if False:
    ## Compute obj_inj
    X, Y = model.mesh
    XY = np.vstack([X.ravel(), Y.ravel()]).T
    npvs = npv_in_injectors(XY)
    npvs.reshape(model.shape)

    fig, axs = fig3("npv_in_injectors")
    plotting.field(axs[0], npvs, "obj", wells=True, argmax=True, colorbar=True)

    ### Optimize obj_inj
    # - Some get stuck in local minima
    # - Smaller steps towards end
    # - Notice that discreteness of npv_in_injectors function means that
    #   they dont all CV to exactly the same
    L = .1 * np.eye(2)
    for i, color in zip(range(3), ['C0', 'C2', 'C6', 'C7', 'C8', 'C9']):
        u0 = rnd.rand(2) * model.domain[1]
        path, objs, info = EnOpt(lambda u: npv_in_injectors(u), u0, L,
                                 regulator=Momentum(0.3), xSteps=xSteps, precond=False)
        plot_path(*axs, path, objs, color=color)
    fig.tight_layout()

    final_sweep("npv_in_injectors", inj_xy=path[-1].reshape((-1, 2)))


if False:
    ## Optimize 2 inj_xy
    u0 = np.array([0, 0] + [2, 0])
    model = model_setup(
        prod_xy = xy_4corners[:2],
        inj_xy = transform_xys(u0).reshape((2, 2)),
        prod_rates = r0 * np.ones((2, 1)) / 2,
        inj_rates = r0 * np.ones((2, 1)) / 2,
    )
    plotting.single.model = model
    fig, axs = fig3("npv_in_injectors_transformed")
    plotting.field(axs[0], pperm, "pperm", wells=True, colorbar=True)

    L = .1 * np.eye(len(u0))
    path, objs, info = EnOpt(npv_in_injectors_transformed, u0, L,
                             regulator=Momentum(.1), xSteps=xSteps, precond=False, rtol=1e-8)
    path = transform_xys(path)

    plot_path(*axs, path[:, :2], objs, color='C0')
    plot_path(*axs, path[:, 2:], color='C5')
    fig.tight_layout()

    final_sweep("npv_in_injectors_transformed", inj_xy=path[-1].reshape((-1, 2)))

if True:
    # Restore default well config
    model = model_setup(
        inj_xy = np.array([model.domain[1]]) / 2,
        inj_rates = r0 * np.ones((1, 1)) / 1,
        prod_rates = r0 * np.ones((4, 1)) / 4,
        prod_xy = xy_4corners,
    )
    plotting.single.model = model

    xx = np.linspace(0.1, 5, 21)
    objs = npv_in_rates(xx[:, None])
    fig, ax = freshfig("npv_in_rates")
    ax.plot(xx, objs, "slategrey")
    ax.grid()
    ax.set(xlabel="rate", ylabel="NPV")

    for i, u0 in enumerate(np.array([[.1, 5]]).T):
        L = .1 * np.eye(len(u0))
        path, objs, info = EnOpt(npv_in_rates, u0, L, xSteps=xSteps, precond=False, rtol=1e-8)
        ax.plot(path, objs, '-o', color=f'C{i+1}')
