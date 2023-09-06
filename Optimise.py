# -*- coding: utf-8 -*-
# # Production optimisation with ensembles – an interactive tutorial
#
# Copyright Patrick N. Raanes, NORCE, 2023.
#
# This is a self-contained tutorial on production optimisation using ensemble methods.
# - Please have a look at the [history matching (HM) tutorial](HistoryMatch.ipynb)
#   for an introduction to Python, Jupyter notebooks, and this reservoir simulator.
#
# If you're on **Google Colab**, run the cell below to install the requirements.
# Otherwise (and assuming you have done the installation described in the README),
# you can skip/delete this cell.

remote = "https://raw.githubusercontent.com/patnr/HistoryMatching"
# !wget -qO- {remote}/master/colab_bootstrap.sh | bash -s

# ## Imports

import copy

import numpy as np
import numpy.random as rnd
import TPFA_ResSim as simulator
from mpl_tools.place import freshfig
from tqdm.auto import tqdm

import tools.plotting as plotting
from tools import geostat, mpl_setup, utils

mpl_setup.init()
np.set_printoptions(precision=6)

# Could put this in npv() rather than copying the base model
# but it's useful to have a base model in the global namespace
# for plotting purposes.

# ## Define model
# We start with the same settings as in the previous tutorial (on history matching).
# This will serve as our default/base model

model = simulator.ResSim(Nx=20, Ny=20, Lx=2, Ly=1, name="Base model")

# #### Perm
seed = rnd.seed(3)
model.K = .1 + np.exp(5 * geostat.gaussian_fields(model.mesh, 1, r=0.8))

# #### Wells
# List of coordinates (x, y) of the 4 cornerns of the rectangular domain

xy_4corners = np.dstack(np.meshgrid(
    np.array([.12, .87]) * model.Lx,
    np.array([.12, .87]) * model.Ly,
)).reshape((-1, 2))

# Suggested total rate of production (total rate of injection must be the same).

rate0 = 1.5

model.inj_xy  = [[model.Lx/2, model.Ly/2]]
model.prod_xy = xy_4corners
model.inj_rates  = rate0 * np.ones((1, 1)) / 1
model.prod_rates = rate0 * np.ones((4, 1)) / 4

# Plot

fig, ax = freshfig(model.name, figsize=(1, .6), rel=True)
model.plt_field(ax, model.K[0], "perm", wells=True, colorbar=True);
fig.tight_layout()

# ## Define simulations

wsat0 = np.zeros(model.Nxy)
T = 1
dt = 0.025
nTime = round(T/dt)

def simulate(model, wsat, pbar=False, leave=False):
    """Compute evolution in time of reservoir saturation field."""
    integrator = model.time_stepper(dt)
    wsats = simulator.recurse(integrator, nTime, wsat, pbar=pbar, leave=leave)
    # Extract production time series from water saturation fields
    wells = model.xy2ind(*model.prod_xy.T)
    prods = np.array([wsat[wells] for wsat in wsats])
    return wsats, prods


# Let us plot the final sweep of the base model configuration.

def plot_final_sweep(model):
    """Simulate reservoir, plot final oil saturation."""
    title = "Final sweep" + (" -- " + model.name) if model.name else ""
    wsats, prods = simulate(model, wsat0)
    fig, ax = freshfig(title, figsize=(1, .6), rel=True)
    model.plt_field(ax, wsats[-1], "oil", wells=True, colorbar=True)
    fig.tight_layout()

plot_final_sweep(model)

# Unlike the history matching tutorial, we will do several distinct "cases",
# i.e. use different *base* model configuration (which are not changed by our methods).
# We have therefore factored out the convenient parameter setter from the simulator.

def remake(model, **kwargs):
    """Instantiate new model config."""
    model = copy.deepcopy(model)
    for k, v in kwargs.items():
        setattr(model, k, v)
    return model

# Note that setting parameters is generally not a trivial task (unlike here).
# It might involve reshaping arrays, translating units, read/write to file, etc.
# Indeed, from a "task runner" perspective, there is no hard distinction between
# writing parameters and running simulations.
#
# Let's store the base-base model.

model0 = remake(model)

# ### Multiprocessing. Only used in ensemble runs

utils.nCPU = True

def apply2(fun, **kwargs):
    """Fix `unzip`, `leave`. Treat singleton case. Requires ens (2D) input."""
    arg0 = next(iter(kwargs.values()))
    singleton = (np.ndim(arg0) == 1) or len(arg0) == 1
    Js = utils.apply(fun, **kwargs, unzip=False, leave=False)
    return Js[0] if singleton else Js


# ## EnOpt
# EnOpt consists of gradient descent with ensemble gradient estimation.
# We wrap the gradient estimation function in another to fix its configuration parameters,
# (to avoid having to pass them through the caller, i.e. gradient descent).

def nabla_ens(chol=1.0, nEns=10, precond=False, normed=True):
    """Set parameters of `ens_grad`."""
    def ens_grad(obj, u):
        """Compute ensemble gradient (LLS regression) for `obj` centered on `u`."""
        cholT = chol.T if isinstance(chol, np.ndarray) else chol * np.eye(len(u))
        U = rnd.randn(nEns, len(u)) @ cholT
        U, _ = utils.center(U)
        J = obj(u + U, desc=f"ens_grad of {obj.__name__}'s")
        J, _ = utils.center(J)
        if precond:
            g = U.T @ J / (nEns-1)
        else:
            g = utils.rinv(U, reg=.1, tikh=True) @ J
        if normed:
            g /= utils.mnorm(g)
        return g
    return ens_grad

# Another ingredient to successful gradient descent is line search.
#
# Parameters:
# - `sign=+/-1`: max/min-imization.
# - `xSteps`: trial step lengths.
# - `rtol`: convergence criterion.
#   Specifies magnitude of improvement required to accept update of iterate.
#   Larger values ⇒ +reluctance to accept update ⇒ *faster* declaration of convergence.
#   Setting to 0 is not recommended, because if the objective function is flat
#   in the neighborhood, then the path could just go in circles on that flat.

def backtracker(sign=+1, xSteps=tuple(1/2**(i+1) for i in range(8)), rtol=1e-8):
    """Set parameters of `backtrack`."""
    def backtrack(x0, J0, objective, search_direction):
        """Line search by bisection."""
        atol = max(1e-8, abs(J0)) * rtol
        with tqdm(total=len(xSteps), desc="Backtrack", leave=False) as pbar:
            for i, step_length in enumerate(xSteps):
                pbar.update(1)
                dx = sign * step_length * search_direction
                x1 = x0 + dx
                J1 = objective(x1)
                dJ = J1 - J0
                if sign*dJ > atol:
                    pbar.update(len(xSteps))  # needed in Jupyter
                    return x1, J1, dict(nDeclined=i)
    return backtrack

# Other acceleration techniques such as momentum, AdaGrad, and Nesterov
# could also be considered, but do not necessarily fit well together with
# line search.
#
# The following implements gradient descent (GD).

def GD(objective, x, nabla=nabla_ens(), line_search=backtracker(), nIter=100):
    """Gradient (i.e. steepest) descent/ascent."""
    J = objective(x)
    path = [x]
    objs = [J]
    info = []  # ⇒ len+1 == len(path)
    for itr in range(nIter):
        grad = nabla(objective, x)
        if (update := line_search(x, J, objective, grad)):
            x, J, dct = update
            path.append(x)
            objs.append(J)
            info.append(dct)
        else:
            status = "Converged ✅"
            break
    else:
        status = "Ran out of iters ❌"
    print(f"{status:<9} {itr=:<5}  {x=!s}  {J=:.2f}")
    return np.asarray(path), np.asarray(objs), info


# ## NPV

# Convert production saturation time series to cumulative monetary value.

def prod2npv(model, prods):
    """Net present value (NPV), i.e. discounted, total oil production."""
    discounts = .99 ** np.arange(nTime + 1)
    prods = 1 - prods                   # water --> oil
    prods = prods * model.prod_rates.T  # volume = saturation * rate
    prods = np.sum(prods, -1)           # sum over wells
    value = prods @ discounts           # sum in time, incld. discount factors
    # Compute cost of water injection
    # PS: We don't bother with cost of water production,
    # since it is implicitly approximated by reduction in oil production.
    inj_rates = model.inj_rates
    if inj_rates.shape[1] == 1:
        inj_rates = np.tile(inj_rates, (1, nTime))
    cost = np.sum(inj_rates, 0)
    cost = cost @ discounts[:-1]
    return value - .5*cost

# Before applying `prod2npv`, the objective function must first compute the production,
# which entails configuring and simulating the model.

def npv(**kwargs):
    """NPV from model config."""
    try:
        new = remake(model, **kwargs)
        wsats, prods = simulate(new, wsat0)
    except Exception:
        return 0  # Invalid model params. Penalize.
        # Use `raise` for debugging.
    return prod2npv(new, prods)


# ## Optimize x-coordinate of single injector
# Let's try it out with a 1D optimisation case.

def npv_in_x_of_inj0_with_fixed_y(x, desc=None):
    """Optimize x-coordinate of injector."""
    xy = np.zeros((len(x), 1, 2))
    xy[..., 0] = x
    xy[..., 1] = y
    Js = apply2(npv, inj_xy=xy, desc=desc)
    return Js

y = model.Ly/2
obj = npv_in_x_of_inj0_with_fixed_y

# Since our model is so simple, and we only have 1 control parameter,
# we can afford to compute and plot the entire objective.

xx = np.linspace(0, model.Lx, 201)
npvs = obj(np.atleast_2d(xx).T, desc=f"Plot pts. of {obj.__name__}")

# +
# Plot objective
fig, ax = freshfig(f"{obj.__name__}({y})", figsize=(7, 3))
ax.set(xlabel="x", ylabel="NPV")
ax.plot(xx, npvs, "slategrey", lw=3);

# Optimize, plot
u0s = model.Lx * np.array([[.05, .1, .2, .8, .9, .95]]).T
for i, u0 in enumerate(u0s):
    path, objs, info = GD(obj, u0, nabla_ens(.3))
    shift = .3*i  # for visual distinction
    ax.plot(path, objs - shift, '-o', c=f'C{i+1}')
fig.tight_layout()


# -

# Note that the objective functions appears to jump at regular intervals.
# This would be even more apparent (no slanting of the "walls") with a higher resolution.
# The phenomenon is due to the fact that the model always collocates wells with grid nodes.
# Other than this, as we might expect, the objective is nice and convex,
# and EnOpt is able to find the minimum, for several different starting positions,
# without much trouble.

# ## Optimize both coordinates

def npv_in_injectors(xys, desc=None):
    """Optimize (x, y) of injectors."""
    nEns = 1 if xys.ndim == 1 else len(xys)
    shape = (nEns, -1, 2)
    return apply2(npv, inj_xy=xys.reshape(shape), desc=desc)


# Compute entire objective

obj = npv_in_injectors
X, Y = model.mesh
XY = np.vstack([X.ravel(), Y.ravel()]).T
npvs = obj(XY, desc=f"Plot pts. of {obj.__name__}")

# +
# Plot objective
fig, axs = plotting.figure12(obj.__name__)
model.plt_field(axs[0], npvs, "NPV", wells=True, argmax=True, colorbar=True);
fig.tight_layout()

# Optimize, plot
for color in ['C0', 'C2', 'C7', 'C9']:
    u0 = rnd.rand(2) * model.domain[1]
    path, objs, info = GD(obj, u0, nabla_ens(.1))
    plotting.add_path12(*axs, path, objs, color=color)
# -

# ##### Comments
# - Given the zig-zag optimization trajectories we see, it would appear
#   that the EnOpt gradient descent implementation could well benefit
#   from using an "acceleration" technique such as "momentum".
#   See git commit `9937d5b2` for a working implementation.
# - Some get stuck in local minima
# - Smaller steps towards end
# - Notice that discreteness of npv_in_injectors function means that
#   they dont all CV to exactly the same

# Plot of final sweep

plot_final_sweep(remake(model, inj_xy=path[-1].reshape((-1, 2)),
                        name=f"Optimized in {obj.__name__}"))

# ## Optimize coordinates of 2 injectors

# With 2 injectors, it's more interesting (not necessary) to also only have 2 producers. So let's configure our model for that.

model = remake(model,
    name = "Lower 2 corners",
    prod_xy = xy_4corners[:2],
    prod_rates = rate0 * np.ones((2, 1)) / 2,
    inj_rates  = rate0 * np.ones((2, 1)) / 2,
)


# As you might imagine, with the 2 producers at the lower corners,
# the optimal 2 injector positions will be somewhere near the upper edge.
# But boundaries are a problem for basic EnOpt.
# Because of its Gaussian character, it will often sample points outside of the domain.
# This won't crash our optimisation, since the `npv` function
# catches all exceptions and converts them to a penatly,
# but the gradient near the border will then seem to indicate that the border is a bad place to be,
# which is not necessarily the case.
#
# - One quickfix to this problem
# (but more technically demanding that the penalization already in place)
# is to truncate the ensemble members to the valid domain.
# - Another alternative (ref. Mathias) is to use a non-Gaussian generalisation of EnOpt that samples
# from a Beta (e.g.) distribution, and uses a different formula than LLS regression
# to estimate the average gradient. This is one more degree of technical overhead.
# - Another alternative is to transform the control variables
# so that the domain is the whole of $\mathcal{R}^d$.
# This is the approach taken below.
# Note that this is not an approach for constrained optimisation in general:
# we can only constrain the control variables, not functions thereof.

# +
def npv_in_injectors_transformed(xys, desc=None):
    return npv_in_injectors(coordinate_transform(xys), desc=desc)

obj = npv_in_injectors_transformed

# +
# Plot perm field
fig, axs = plotting.figure12(obj.__name__)
model.plt_field(axs[0], model.K[0], "perm", wells=True, colorbar=True)
fig.tight_layout()

# Optimize
def coordinate_transform(xys):
    """Map `ℝ² --> (0, Lx) x (0, Ly)`, with `origin ↦ domain centre`."""
    xys = np.array(xys, dtype=float)
    sigmoid = lambda x, h, w=1: h/(1 + np.exp(-x/w))
    xys[..., 0::2] = sigmoid(xys[..., 0::2], model.Lx)
    xys[..., 1::2] = sigmoid(xys[..., 1::2], model.Ly)
    return xys


u0 = np.array([-1, 0, +1, 0])
path, objs, info = GD(obj, u0, nabla_ens(.1))
path = coordinate_transform(path)

# Plot optimisation trajectory
plotting.add_path12(*axs, path[:, :2], objs, color='C1')
plotting.add_path12(*axs, path[:, 2:], color='C3')
fig.tight_layout()
# -

# Let's plot the final sweep

plot_final_sweep(remake(model, inj_xy=path[-1].reshape((-1, 2)),
                        name=f"Optimzed in {obj.__name__}"))

# ## Optimize single rate

# When setting the injection rate(s), we must also
# set the total production rates to be the same (this is a model constraint).
# When there are multiple producers, we could distribute the production
# differently across the wells (somehow, our choice),
# but here we all set them all to be equal.

def npv_in_rates(inj_rates, desc=None):
    """`npv(inj_rates)`."""
    nEns = 1 if inj_rates.ndim == 1 else len(inj_rates)
    inj_rates = inj_rates.reshape((nEns, -1, 1))  # --> (nEns, nInj, 1)

    nProd = len(model.prod_rates)
    total_rate = np.sum(inj_rates, axis=1).squeeze()
    prod_rates = np.full((nEns, nProd, 1), np.nan)
    prod_rates.T[:] = total_rate/nProd

    return apply2(npv, inj_rates=inj_rates, prod_rates=prod_rates, desc=desc)

obj = npv_in_rates

# Restore default well config

model = model0

# Optimize

xx = np.linspace(0.1, 5, 21)
npvs = obj(np.atleast_2d(xx).T, "Plot pts.")

# +
fig, ax = freshfig(obj.__name__, figsize=(1, .4), rel=True)
ax.grid()
ax.set(xlabel="rate", ylabel="NPV")
ax.plot(xx, npvs, "slategrey")

for i, u0 in enumerate(np.array([[.1, 5]]).T):
    path, objs, info = GD(obj, u0, nabla_ens(.1))
    shift = i+1  # for visual distinction
    ax.plot(path, objs - shift, '-o', color=f'C{i+1}')
fig.tight_layout()
# -

# ## Interactive (manual) optimisation of multiple rates

# Let's make the flow "less orthogonal" by not placing the wells on a rectilinear grid (i.e. the 4 corners).

triangle = [0, 135, -135]
wells = dict(
    inj_xy = ([[model.Lx/2, model.Ly/2]] +
              [utils.xy_p_normed(th + 90, *model.domain[1]) for th in triangle]),
    prod_xy = [utils.xy_p_normed(th - 90, *model.domain[1]) for th in triangle],
    inj_rates  = rate0 * np.ones((4, 1)) / 4,
    prod_rates = rate0 * np.ones((3, 1)) / 3,
)
model = remake(model, **wells)

# Show well config

fig, ax = freshfig("Triangle case", figsize=(1, .6), rel=True)
model.plt_field(ax, model.K[0], "perm", wells=True, colorbar=True);
fig.tight_layout()


# Define function that takes injection rates and computes final sweep, i.e. saturation field.

def final_sweep_given_inj_rates(**kwargs):
    inj_rates = np.array([list(kwargs.values())]).T
    prod_rates = np.ones((3, 1)) / 3 * np.sum(inj_rates)
    new = remake(model, inj_rates=inj_rates, prod_rates=prod_rates)
    wsats, prods = simulate(new, wsat0)
    npv = prod2npv(new, prods)
    print(f"NPV for these injection_rates: {npv:.5f}")
    return wsats[-1]


# By assigning `controls` to this function (the rate of each injector)...

final_sweep_given_inj_rates.controls = dict(
    i0 = (0, 1.4),
    i1 = (0, 1.4),
    i2 = (0, 1.4),
    i3 = (0, 1.4),
)

# ... the following widget allows us to "interactively" (but manually) optimise the rates.
# This is of course only feasible because the model is so simple and runs so fast.

plotting.field_console(model, final_sweep_given_inj_rates, "oil", wells=True, figsize=(1, .6))

# ## Automatic (EnOpt) optimisation
# Run EnOpt (below).

u0 = .7*np.ones(len(model.inj_rates))
path, objs, info = GD(obj, u0, nabla_ens(.1))

# Now try setting
# the resulting suggested values in the interactive widget above.
# Were you able to find equally good settings?
