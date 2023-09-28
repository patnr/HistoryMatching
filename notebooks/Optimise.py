# -*- coding: utf-8 -*-
# # Production optimisation with ensembles – an interactive tutorial
#
# Copyright Patrick N. Raanes, NORCE, 2023.
#
# This is a tutorial on production optimisation using ensemble methods.
# Please also have a look at the [history matching (HM) tutorial](HistoryMatch.ipynb)
# for an introduction to Python, Jupyter notebooks, and this reservoir simulator.
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
from tqdm.auto import tqdm as progbar

import tools.plotting as plotting
from tools import geostat, mpl_setup, utils
from tools.utils import center, apply

mpl_setup.init()
np.set_printoptions(precision=4, sign=' ', floatmode="fixed")

# ## Define model
# We start with the same settings as in the previous tutorial (on history matching).
# This will serve as our default/base model.
# It is convenient to define it in the global namespace.

# #### Grid

model = simulator.ResSim(Nx=20, Ny=20, Lx=2, Ly=1, name="Base model")

# #### Permeability

seed = rnd.seed(3)
model.K = .1 + np.exp(5 * geostat.gaussian_fields(model.mesh, 1, r=0.8))

# #### Wells
# List of coordinates (x, y) of the 4 cornerns of the rectangular domain

near01 = np.array([.12, .87])
xy_4corners = [[x, y]
               for y in model.Ly*near01
               for x in model.Lx*near01]

# Suggested total rate of production.

rate0 = 1.5

# Note that the production and injection rates add up to the same
# (at each time step), as they must (or model will raise an error).

model.inj_xy = [[model.Lx/2, model.Ly/2]]
model.prod_xy = xy_4corners
model.inj_rates  = rate0 * np.ones((1, 1)) / 1
model.prod_rates = rate0 * np.ones((4, 1)) / 4

# #### Plot

fig, ax = freshfig(model.name, figsize=(1, .6), rel=True)
model.plt_field(ax, model.K[0], "perm");
fig.tight_layout()

# #### Simulations

wsat0 = np.zeros(model.Nxy)
T = 1
dt = 0.025
nTime = round(T/dt)

# Let us plot the final sweep of the base model configuration.

def plot_final_sweep(model):
    """Simulate reservoir, plot final oil saturation."""
    wsats = model.sim(dt, nTime, wsat0, pbar=False)
    title = "Final sweep" + (" -- " + model.name) if model.name else ""
    fig, ax = freshfig(title, figsize=(1, .6), rel=True)
    model.plt_field(ax, wsats[-1], "oil")
    fig.tight_layout()

plot_final_sweep(model)

# ## NPV objective function
# The NPV (objective) function,
# similar to the `forward_model` of the history matching tutorial,
# entails configuring and running/simulating the model.
# But the main output is now the economical net value (profit),
# while some other variables are included as diagnostics.
# Also, importantly, note the it's all wrapped in error penalisation.

def npv(model, **params):
    """Discounted net present value (NPV) from model config."""
    try:
        model = remake(model, **params)
        wsats = model.sim(dt, nTime, wsat0, pbar=False)
        # Sum over wells
        prod_total = partial_volumes(model, wsats, "prod").sum(0)
        inj_total  = partial_volumes(model, wsats, "inj").sum(0)
        # Sum in time
        prod_total = prod_total @ discounts
        inj_total = inj_total @ discounts
        # Add up
        value = (price_of_oil * prod_total -
                 price_of_inj * inj_total)
        other = dict(wsats=wsats, prod_total=prod_total, inj_total=inj_total)
    except Exception:
        # Invalid model params ⇒ penalize.
        # Use `raise` for debugging.
        value, other = 0, None
    return value, other

# Note that water injection has a cost.
# Is seems a reasonable simplification to let this serve as a stand-in
# also for the cost of GHG emissions.
# We don't bother with cost of water production,
# since it is implicitly approximated by reduction in oil production.
#
# The following values are not motivated by any realism.
# However, the 1-to-1 relationship implied by mass balance of the simulator
# means that the (volumetric) price of injection must be cheapter than for oil
# in order for production (even at 100% oil saturation) to be profitable.
#
# PS: since they just get defined in the global namespace,
# they cannot be manipulated by our ensemble methods
# (i.e. we cannot be "robust" to price fluctuations).

price_of_inj = 5e3
price_of_oil = 1e4
discounts = .99 ** np.arange(nTime)

# #### Setter
# Unlike the history matching tutorial, we define a parameter setter also
# outside of the forward model. This will be convenient since we will do
# several distinct "cases", i.e. model configurations with differences not
# generated via our methods.  On the other hand, we do not bother to
# implement/support permability parameterisation, as in the previous tutorial.
# Note that setting parameters is not generally as trivial a task as it is here.
# It might involve reshaping arrays, translating units, read/write to file, etc.
# Indeed, from a "task runner" perspective, there is no hard distinction between
# writing parameters and running simulations.

def remake(model, **params):
    """Instantiate new model config."""
    model = copy.deepcopy(model)
    for k, v in params.items():
        setattr(model, k, v)
    return model

# Let's store the base model.

original_model = remake(model)

# #### Extract well flux from saturation fields
# Also contains some minor bookkeeping
# (for example, unify treatment of constant/variable rates).

def partial_volumes(model, wsats, inj_or_prod):
    """Essentially `saturation * rate` for `inj_or_prod` wells."""
    # Saturations
    if inj_or_prod == "prod":
        # Oil (use trapezoid approx)
        well_inds = model.xy2ind(*model.prod_xy.T)
        saturations = 0.5 * (wsats[:-1, well_inds] +
                             wsats[+1:, well_inds])
        saturations = (1 - saturations).T  # water --> oil
    elif inj_or_prod == "inj":
        # Injector uses 100% water
        saturations = 1
    else:
        raise KeyError

    # Rates
    rates = getattr(model, f"{inj_or_prod}_rates")
    if (_nT := rates.shape[1]) == 1 and _nT != nTime:
        # constant-in-time ⇒ replicate for all time steps
        rates = np.tile(rates, (1, nTime))

    volume_per_rate = dt * model.hx * model.hx

    return volume_per_rate * rates * saturations


# ## EnOpt

# #### Multiprocessing
# Ensemble methods are easily parallelizable, achieved hereunder by `apply`
# that we "map" on ensemble simulations.

utils.nCPU = True

# It should be noted that -- in some cases -- it is best to leave the parallelisation to the
# model/simulator/objective function itself since it is "closer to the metal"
# and so can therefore do more speed optimisation. For example if the simulations
# can be vectorized over ensemble members, then multi-threaded `numpy` is likey faster).

# #### Ensemble gradient estimator
# EnOpt consists of gradient descent with ensemble gradient estimation.
# We wrap the gradient estimation function in another to fix its configuration parameters,
# (to avoid having to pass them through the caller, i.e. gradient descent).

def nabla_ens(chol=1.0, nEns=10, precond=False, normed=True):
    """Set parameters of `ens_grad`."""
    pbar = dict(desc="ens_grad", leave=False)
    def ens_grad(obj, u):
        """Compute ensemble gradient (LLS regression) for `obj` centered on `u`."""
        cholT = chol.T if isinstance(chol, np.ndarray) else chol * np.eye(len(u))
        U = rnd.randn(nEns, len(u)) @ cholT
        dU = center(U)[0]
        dJ = apply(obj, u + dU, pbar=pbar)
        if precond:
            g = dU.T @ dJ / (nEns-1)
        else:
            g = utils.rinv(dU, reg=.1, tikh=True) @ dJ
        if normed:
            g /= utils.mnorm(g)
        return g
    return ens_grad

# #### Backtracking
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
        with progbar(total=len(xSteps), desc="Backtrack", leave=False) as pbar:
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

# #### Gradient descent
# Other acceleration techniques (AdaGrad, Nesterov, momentum,
# of which git commit `9937d5b2` contains a working implementation)
# could also be considered, but do not necessarily play nice with line search.
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


# ## Case: Optimize injector location
# Let's try optimising the location (x, y) of the injector well.
#
# The objective function is simply a thin wrapper around `npv`
# which translates its single (vector) input argument into `kwargs`,
# and discards all output except the scalar NPV.

def npv_inj_xy(xys):
    return npv(model, inj_xy=xys)[0]

obj = npv_inj_xy
model = original_model
print(f"Case: '{obj.__name__}' for '{model.name}'")

# The model is sufficiently cheap that we can afford to compute the objective
# over its entire 2D domain, and plot it.

XY = np.stack(model.mesh, -1).reshape((-1, 2))
npvs = apply(obj, XY, pbar="obj(mesh)")
npvs = np.asarray(npvs)

# We have in effect conducted an exhaustive computation of the objective function,
# so that we already know the true, global, optimum:

argmax = npvs.argmax()
print("Global (exhaustive search) optimum:", f"{npvs[argmax]:.4}",
      "at (x={:.2}, y={:.2})".format(*model.ind2xy(argmax)))

# Note that the optimum is not quite in the centre of the domain,
# which is caused by the randomness (including asymmetry) of the permeability field.
#
# Now let's try EnOpt from a few different starting/initial guesses,
# and plot the optimisation paths along with the contours of the objective
# *PS: code for both tasks must be in same cell in order to plot on same figure*.

# +
# Plot objective
fig, axs = plotting.figure12(obj.__name__)
model.plt_field(axs[0], npvs, "NPV", argmax=True, wells=False);

# Optimize, plot paths
for color in ['C0', 'C2', 'C7', 'C9']:
    u0 = rnd.rand(2) * model.domain[1]
    path, objs, info = GD(obj, u0, nabla_ens(.1))
    plotting.add_path12(*axs, path, objs, color=color, labels=False)
fig.tight_layout()
# -

# ##### Comments
# - The increase in objective function for each step is guaranteed by the line search
#   (but could cause getting stuck in a local minimum). It is also what causes
#   the step size to vanish towards later iterations.
# - EnOpt uses gradient descent, which is a "local" optimizer (being gradient based).
#   Therefore it can get stuck in local mimima, for example (depends on the random
#   numbers used) the corner areas outside a producer.
# - Even when they find the global minimum, the optimisation paths don't
#   converge on the exact same point (depending on their starting point / initial guess).
#   This will be further explained in the following case.

# Plot of final sweep of the result of the final optimisation trial.

plot_final_sweep(remake(model, inj_xy=path[-1], name=f"Optimal for {obj.__name__}"))

# ## Case: Optimize x-coordinate of single injector
#
# The setters in `remake` and ResSim simplify much in defining the forward model.
# Still, sometimes we need to pre-process the arguments some more.
# For example, suppose we only want to vary the x-coordinate of the injector,
# while keeping the y-coordinate fixed.

def npv_x_with_fixed_y(xs):
    xys = np.stack([xs, xs], -1) # ⇒ (1d or 2d)
    xys[..., 1] = y  # fix constant value
    return npv(model, inj_xy=xys)[0]

obj = npv_x_with_fixed_y
model = original_model
y = model.Ly/2
print(f"Case: '{obj.__name__}' for '{model.name}'")

# *PS: The use of `...` is a trick that allows operating on the last axis of `xys`,
# which works both when it's 1d and 2d.*
# Also note that we could of course have re-used `npv_inj_xy` to define `npv_x_with_fixed_y`.
# This will be our approach for the subsequent case.

x_grid = np.linspace(0, model.Lx, 201)
npvs = apply(obj, x_grid, pbar="obj(x_grid)")

# +
# Plot objective
fig, ax = freshfig(f"{obj.__name__}({y})", figsize=(7, 3))
ax.set(xlabel="x", ylabel="NPV")
ax.plot(x_grid, npvs, "slategrey", lw=3);

# Optimize, plot paths
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

# ## Case: Optimize coordinates of 2 injectors

# With 2 injectors, it's more interesting (not necessary) to also only have 2 producers.
# So let's configure our model for that, placing the 2 producers at the lower corners.

model = remake(original_model,
    name = "Lower 2 corners",
    prod_xy = xy_4corners[:2],
    prod_rates = rate0 * np.ones((2, 1)) / 2
)

# Now, as you might imagine, the optimal injector positions will be somewhere near the upper edge.
# But boundaries are a problem for basic EnOpt.
# Because of its Gaussian character, its gradient estimation will often sample points
# outside of the domain.
# This won't crash our optimisation, since the `npv` function
# catches all exceptions and converts them to a penalty,
# but the gradient near the border will then seem to indicate that the border is a bad place to be,
# which is not necessarily the case.
#
# - One quickfix to this issue is to truncate the ensemble members to the valid domain.
# - A more sophisticated alternative is to use a non-Gaussian generalisation
#   of EnOpt that samples from a Beta (e.g.) distribution,
#   and uses a different formula than LLS regression to estimate the average gradient (ref. Mathias).
# - The above two solutions are somewhat technically demanding,
#   since they require communicating the boundaries to the gradient estimator.
#   A simpler solution is to transform the control variables
#   so that the domain is the whole of $\mathcal{R}^d$.
#
# Note that these approaches can only constrain the control variables themselves,
# not functions thereof (e.g. there might be rate constraints, while the control
# variables are actually pressures, WBHP).
#
# Below, we take the transformation approach.

def coordinate_transform(xys):
    """Map `ℝ --> (0, L)` with `origin ↦ domain centre`, in both dims (axis 1)."""
    # An alternative to reshape/undo is slicing with 0::2 and 1::2
    xy2d = np.array(xys, dtype=float).reshape((-1, 2))
    xy2d[:, 0] = sigmoid(xy2d[:, 0], model.Lx)  # transform x
    xy2d[:, 1] = sigmoid(xy2d[:, 1], model.Ly)  # transform y
    return xy2d.reshape(np.shape(xys))

def sigmoid(x, height, width=1):
    return height/(1 + np.exp(-x/width))

inj_xys0 = [[-1, 0], [+1, 0]]
model = remake(model,
    inj_xy = coordinate_transform(inj_xys0),
    inj_rates = rate0 * np.ones((2, 1)) / 2,
)

# The objective function is otherwise unchanged.

# +
def npv_xy_transf(xys):
    return npv_inj_xy(coordinate_transform(xys))

obj = npv_xy_transf
print(f"Case: '{obj.__name__}' for '{model.name}'")
# -

# The objective is now a function of `2*nInj = 4` variables.
# It is therefore difficult to plot (requires cross-sections or other projections)
# and anyway computing it would be `nPixels_per_dim^nInj` times more costly.
# We therefore just plot the (known) permeability field along with initial well layout.

# +
# Optimize
u0 = np.ravel(inj_xys0)
path, objs, info = GD(obj, u0, nabla_ens(.1))
path = coordinate_transform(path)

fig, axs = plotting.figure12(obj.__name__)
model.plt_field(axs[0], model.K[0], "perm")

# Plot optimisation trajectory
plotting.add_path12(*axs, path[:, :2], objs, color='C1')
plotting.add_path12(*axs, path[:, 2:], color='C3')
fig.tight_layout()
# -

# Seems reasonable.
# A useful sanity check is provided by inspecting the resulting flow pattern.

plot_final_sweep(remake(model, inj_xy=path[-1], name=f"Optimal for {obj.__name__}"))

# ## Case: Optimize single rate

# When setting the injection rate(s), we must also
# set the total production rates to be the same (this is a model constraint),
# and vice-versa.

def equalize(rates, nWell):
    """Distribute the total rate equally among `nWell`."""
    return np.tile(rates.sum(0) / nWell, (nWell, 1))

# Thus, as above, we need to pre-compute something before calling `npv()`.

# +
def npv_in_inj_rates(inj_rates):
    prod_rates = equalize(inj_rates, model.nProd)
    return npv(model, inj_rates=inj_rates, prod_rates=prod_rates)[0]

obj = npv_in_inj_rates
model = original_model
print(f"Case: '{obj.__name__}' for '{model.name}'")
# -

# Again, we are able and can afford to compute and plot the entire objective.

rate_grid = np.linspace(0.1, 5, 21)
npvs = apply(obj, rate_grid, pbar="obj(rate_grid)")

# It makes sense that there is an optimum sweet spot somewhere in the middle.
# - Little water injection ⇒ little oil production.
# - Much water injection ⇒ very pricey, whereas reservoir contains finite amount of oil.

# +
# Optimize
fig, ax = freshfig(obj.__name__, figsize=(1, .4), rel=True)
ax.grid()
ax.set(xlabel="rate", ylabel="NPV")
ax.plot(rate_grid, npvs, "slategrey")

for i, u0 in enumerate(np.array([[.1, 5]]).T):
    path, objs, info = GD(obj, u0, nabla_ens(.1))
    shift = i+1  # for visual distinction
    ax.plot(path, objs - shift, '-o', color=f'C{i+1}')
fig.tight_layout()
# -

# ## Case: multiple rates (with interactive/manual optimisation)
# The objective is again the npv as a function of the injection rate.

obj = npv_in_inj_rates

# But this time let's have more injectors,
# and therefore also rearrange the producers.

triangle = [0, 135, -135]
wells = dict(
    inj_xy = ([[model.Lx/2, model.Ly/2]] +
              [utils.pCircle(th + 90, *model.domain[1]) for th in triangle]),
    prod_xy = [utils.pCircle(th - 90, *model.domain[1]) for th in triangle],
    inj_rates  = rate0 * np.ones((4, 1)) / 4,
    prod_rates = rate0 * np.ones((3, 1)) / 3,
)
model = remake(model, **wells, name="Triangle case")
print(f"Case: '{obj.__name__}' for '{model.name}'")

# Show well layout

fig, ax = freshfig(model.name, figsize=(1, .6), rel=True)
model.plt_field(ax, model.K[0], "perm");
fig.tight_layout()


# Define function that takes injection rates and computes final sweep, i.e. saturation field.
# Also print (with terminal color codes) the resulting NPV.

def final_sweep_given_inj_rates(**kwargs):
    inj_rates = np.array([list(kwargs.values())]).T
    value, info = npv(model, inj_rates=inj_rates, prod_rates=equalize(inj_rates, model.nProd))
    print("\x1b[45m NPV for these injection_rates:", f"\x1b[30;47;1m{value}\x1b[0m")
    return info['wsats'][-1]


# By assigning `controls` to this function (the rate of each injector)...

final_sweep_given_inj_rates.controls = dict(
    inj0_rate = (0, 1.4),
    inj1_rate = (0, 1.4),
    inj2_rate = (0, 1.4),
    inj3_rate = (0, 1.4),
)

# ... the following widget allows us to "interactively" (but manually) optimize the rates.
# This is of course only feasible because the model is so simple and runs so fast.

plotting.field_console(model, final_sweep_given_inj_rates, "oil", wells=True, figsize=(1, .6))

# #### Automatic (EnOpt) optimisation
# Run EnOpt (below).

u0 = .7*np.ones(model.nInj)
path, objs, info = GD(obj, u0, nabla_ens(.1))

# Now try setting
# the resulting suggested values in the interactive widget above.
# Were you able to find equally good settings?

# ## Case: 5-spot similar to Angga -- Make Pareto front
# Compared to Angga:
#
# - No compressibility
# - 20x20 vs. 60x60
# - Simplified geology (permeability)
# - Injection is constant in time and across wells
#
# Only the 1st item is hard to change.

# +
model = remake(original_model,
    name = "Angga",
    prod_xy = [[model.Lx/2, model.Ly/2]],
    inj_xy = xy_4corners,
    prod_rates  = rate0 * np.ones((1, 1)) / 1,
    inj_rates = rate0 * np.ones((4, 1)) / 4,
)

plot_final_sweep(model)


# +
def npv_in_prod_rates(prod_rates):
    inj_rates = equalize(prod_rates, model.nInj)
    return npv(model, prod_rates=prod_rates, inj_rates=inj_rates)[0]

obj = npv_in_prod_rates
print(f"Case: '{obj.__name__}' for '{model.name}'")
# -

# #### Optimize

fig, ax = freshfig(obj.__name__, figsize=(1, .8), rel=True)
rate_grid = np.logspace(-2, 1, 31)
optimal_rates = []
# cost_multiplier = [.01, .04, .1, .4, .9, .99]
__default__ = price_of_inj
cost_multiplier = np.arange(0.1, 1, 0.1)
for i, xCost in enumerate(cost_multiplier):
    price_of_inj = __default__ * xCost
    npvs = apply(obj, rate_grid, pbar="obj(rate_grid)")
    ax.plot(rate_grid, npvs, label=f"{xCost:.1}")
    path, objs, info = GD(obj, np.array([2]), nabla_ens(.1))
    optimal_rates.append(path[-1])
price_of_inj = __default__  # restore
ax.set_ylim(1e-2)
ax.legend(title="×price_of_inj")
ax.set(xlabel="rate", ylabel="NPV")
fig.tight_layout()
ax.grid()

# #### Pareto front
# Breakdown npv (into emissions and sales) for optima

sales = []
emissions = []
for i, prod_rates in enumerate(optimal_rates):
    inj_rates = equalize(prod_rates, model.nInj)
    value, other = npv(model, prod_rates=prod_rates, inj_rates=inj_rates)
    sales.append(other['prod_total'])
    emissions.append(other['inj_total'])


fig, ax = freshfig("Pareto front (npv-optimal settings for range of price_of_inj)", figsize=(1, .8), rel=True)
ax.grid()
ax.set(xlabel="npv (income only)", ylabel="inj/emissions (expenses)")
ax.plot(sales, emissions, "o-")
fig.tight_layout()
