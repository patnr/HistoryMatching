# -*- coding: utf-8 -*-
# # Production optimisation with ensembles – an interactive tutorial
#
# Copyright Patrick N. Raanes, NORCE, 2023.
#
# This is a tutorial on production optimisation using ensemble methods.
# - Please also have a look at the [history matching (HM) tutorial](HistoryMatch.ipynb)
#   for an introduction to Python, Jupyter notebooks, and the reservoir simulator.
# - You don't need to run all cases (i.e. you can skip to what you like)
#   but this will of course affect the random number generation (rng).
#   Also, it is generally not safe to go back to an earlier section (different rng,
#   different simulator setup, etc) without restarting the kernel/interpreter.

# #### Install
# If you're on **Google Colab**, run the cell below to install the requirements.
# Otherwise (and assuming you have done the installation described in the README),
# you can skip/delete this cell.

remote = "https://raw.githubusercontent.com/patnr/HistoryMatching"
# !wget -qO- {remote}/master/colab_bootstrap.sh | bash -s

# #### Imports

# +
import copy
from dataclasses import dataclass

import numpy as np
import numpy.random as rnd
import TPFA_ResSim as simulator

from tools import geostat, plotting, utils
from tools.utils import center, apply, progbar, mesh2list
# -

# #### Config

plotting.init()
np.set_printoptions(precision=4, sign=' ', floatmode="fixed")

# ## Define model
# We start with the same settings as in the previous tutorial (on history matching).
# This will serve as our default/base model.
# It is convenient to define it in the global namespace.

# #### Grid

model = simulator.ResSim(Nx=20, Ny=20, Lx=2, Ly=1, name="Base model")

# #### Permeability

seed = rnd.seed(23)
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

fig, ax = plotting.freshfig(model.name)
model.plt_field(ax, model.K[0], "perm", grid=True);

# #### Simulations

wsat0 = np.zeros(model.Nxy)
T = 1
dt = 0.025
nTime = round(T/dt)

# Let us plot the final sweep of the base model configuration.

def plot_final_sweep(model):
    """Simulate reservoir, plot final oil saturation."""
    wsats = model.sim(dt, nTime, wsat0, pbar=False)
    _, ax = plotting.freshfig("Final sweep -- " + model.name)
    model.plt_field(ax, wsats[-1], "oil")

plot_final_sweep(model)

# ## NPV objective function
# The NPV (objective) function,
# similar to the `forward_model` of the history matching tutorial,
# entails configuring and running/simulating the model.
# But the main output is now the economical net value (profit),
# while some other variables are included as diagnostics.
# Also, importantly, note that it's all wrapped in error penalisation.

def npv(model, **params):
    """Discounted net present value (NPV) from model config."""
    try:
        model = remake(model, **params)
        wsats = model.sim(dt, nTime, wsat0, pbar=False)
        # Volumes (should NOT scale with model hx*hy)
        inj_volumes = dt * model.actual_rates['inj']  * 1
        oil_volumes = dt * model.actual_rates['prod'] * (1 - prod_sats(model, wsats).T)
        # Sum over wells (axis 0) and time (axis 1)
        oil_total = oil_volumes.sum(0) @ discounts
        inj_total = inj_volumes.sum(0) @ discounts
        # Add up
        value = (+price['oil'] * oil_total
                 -price['inj'] * inj_total)
        # Add other costs
        value -= price['/well'] * np.sum(model.actual_rates['prod'] != 0)
        value -= price['/well'] * np.sum(model.actual_rates['inj'] != 0)
        value -= price['turbo'] * (model.actual_rates['prod'].sum(0) - rate0).clip(0).sum() * dt
        # value -= price['diffs'] * np.abs(np.diff(model.actual_rates['prod'], 1)).clip(max=.2).sum()
        # value -= price['fixed'] * (1 + max(find_shut_ins(model.actual_rates['prod'])))
        other = dict(model=model, wsats=wsats, oil_total=oil_total, inj_total=inj_total)
    except Exception:
        # Invalid model params
        value, other = 0, None  # ⇒ penalize
        # `raise`  # enable post-mort. debug
    return value, other

# Note that water injection has a cost.
# Is seems a reasonable simplification to let this serve as a stand-in
# also for the cost of GHG emissions.
# We don't bother with cost of water *production*,
# since it is implicitly approximated by reduction in oil production.
#
# The following values are not grounded in reality.
# However, the 1-to-1 relationship implied by mass balance of the simulator
# means that the (volumetric) price of injection must be cheapter than for oil
# in order for production (even at 100% oil saturation) to be profitable.

OneYear = .1  # ⇒ 10 years to more-or-less drain using rate0

price = {
    "inj": 50,
    "oil": 100,
    'turbo': 2,
    'diffs': 1,
    "fixed": 0.8 * dt/OneYear,
    '/well': 0.3 * dt/OneYear,
}
discounts = .96 ** (dt/OneYear * np.arange(nTime))

# Note that, being defined in the global namespace,
# and not having implemented any "setter" for them,
# these values cannot be manipulated by our ensemble methods.
# *Therefore, for example, we cannot account for uncertainty/fluctuations in prices.*

# #### Parameter setter

def remake(model, **params):
    """Instantiate new model config."""
    model = copy.deepcopy(model)
    for k, v in params.items():
        setattr(model, k, v)
    return model

# Note that, unlike the history matching tutorial,
# we do not bother to implement/support permability setter, which would contain a few extra steps.
# Also, the parameter setter is factored out of the forward model, which will be convenient
# since we will do several distinct "cases" of model configurations. Let's store the base one.

original_model = remake(model)

# #### Auxiliary function

def sigmoid(x, height, width=1):
    """Centered sigmoid: `S(0) == height/2`, with `S(width) = 0.73 * height`."""
    return height/(1 + np.exp(-x/width))


# #### Extracting well flux from saturation fields

def prod_sats(model, wsats):
    """Saturations at producers, per time interval (⇒ trapezoidal rule)."""
    s = wsats[:, model.xy2ind(*model.prod_xy.T)]
    return (s[:-1] + s[+1:]) / 2


# ## EnOpt
#
# #### Ensemble gradient estimator
# EnOpt consists of gradient descent with ensemble gradient estimation.

@dataclass
class nabla_ens:
    """Ensemble gradient estimate (LLS regression)."""
    chol:    float = 1.0   # Cholesky factor (or scalar std. dev.)
    nEns:    int   = 10    # Size of control perturbation ensemble
    precond: bool  = False # Use preconditioned form?
    # Will be used later:
    robustly:None  = None  # Method of treating robust objectives
    obj_ux:  None  = None  # Conditional objective function
    X:       None  = None  # Uncertainty ensemble

    def eval(self, obj, u, pbar):
        """Estimate `∇ obj(u)`"""
        U = utils.gaussian_noise(self.nEns, len(u), self.chol)
        dU = center(U)[0]
        dJ = self.obj_increments(obj, u, u + dU, pbar)
        if self.precond:
            g = dU.T @ dJ / (self.nEns-1)
        else:
            g = utils.rinv(dU, reg=.1, tikh=True) @ dJ
        return g

    def obj_increments(self, obj, u, U, pbar):
        return apply(obj, U, pbar=pbar)  # don't need to `center`

# Note the use of `apply` (which is a thin wrapper on top of `map` or a for loop)
# to compute `obj(u)` for each `u` in the ensemble `U`,
# which is done using parallelisation by multiprocessing.
# *PS: Colab only gives you 1 CPU, so this has no impact.*

utils.nCPU = "auto"


# #### Backtracking
# Another ingredient to successful gradient descent is line search.
#
# *PS: The `rtol>0` parameter specifies the minimal improvement required
# to accept the updated iterate.
# Larger values ⇒ more reluctance to accept update ⇒ *faster* declaration of convergence.
# Setting to 0 is not recommended because then it will not converge in flat neighborhoods.*
#
# TODO: implement Armijo-Goldstein.

@dataclass
class backtracker:
    """Bisect until sufficient improvement."""
    sign:   int   = +1                                  # Search for max(+1) or min(-1)
    xSteps: tuple = tuple(.5**(i+1) for i in range(8))  # Trial step lengths
    rtol:   float = 1e-8                                # Convergence criterion
    def eval(self, obj, u0, J0, search_direction, pbar):
        atol = max(1e-8, abs(J0)) * self.rtol
        pbar.reset(len(self.xSteps))
        for i, step_length in enumerate(self.xSteps):
            du = self.sign * step_length * search_direction
            u1 = u0 + du
            J1 = obj(u1)
            dJ = J1 - J0
            pbar.update()
            if self.sign*dJ > atol:
                pbar.reset(pbar.total)
                return u1, J1, dict(nDeclined=i)

# Other acceleration techniques (AdaGrad, Nesterov, momentum,
# of which git commit `9937d5b2` contains a working implementation)
# could also be considered, but do not necessarily play nice with line search.

# #### Gradient descent
# The following implements gradient descent (GD).

def GD(objective, u, nabla=nabla_ens(), line_search=backtracker(), nrmlz=True, nIter=100, quiet=False):
    """Gradient (i.e. steepest) descent/ascent."""

    # Reusable progress bars (limits flickering scroll in Jupyter) w/ short np printout
    with (progbar(total=nIter, desc="⏳ GD running", leave=True,  disable=quiet) as pbar_gd,
          progbar(total=10000, desc="→ grad. comp.", leave=False, disable=quiet) as pbar_en,
          progbar(total=10000, desc="→ line_search", leave=False, disable=quiet) as pbar_ls,
          np.printoptions(precision=2, threshold=2, edgeitems=1)):

        states = [[u, objective(u), "{cause for stopping}"]]

        for itr in range(nIter):
            u, J, info = states[-1]
            pbar_gd.set_postfix(u=f"{u}", obj=f"{J:.3g}📈")

            grad = nabla.eval(objective, u, pbar_en)
            if nrmlz:
                grad /= np.sqrt(np.mean(grad**2))
            updated = line_search.eval(objective, u, J, grad, pbar_ls)
            pbar_gd.update()

            if updated:
                states.append(updated)
            else:
                cause = "✅ GD converged"
                break
        else:
            cause = "❌ GD ran out of iters"
        pbar_gd.set_description(cause)

    states[0][-1] = cause
    return [np.asarray(arr) for arr in zip(*states)]  # "transpose"


# ## Sanity check
# It is always wise to do some dead simple testing.
# Let's test `GD` on some [well known](https://en.wikipedia.org/wiki/Test_functions_for_optimization) toy problems.

def obj(u):
    # Center model domain, apply aspect ratio
    u = u - [model.Lx/2, model.Ly/2]
    u = u * [1, getattr(obj, 'aspect', 1)]

    if obj.case == "Quadratic":
        return np.mean(u*u, axis=-1)

    elif obj.case == "Rosenbrock":
        u = u * [4, 4]
        u = u.T
        t1 = u[1:] - u[:-1] * u[:-1]
        t2 = u[:-1] - 1
        return np.sum(100*(t1*t1) + t2*t2, 0)

    elif obj.case == "Rastrigin":
        u = u * [5.12, 5.12]
        return 20 + (u*u - 5*np.cos(2*np.pi*u)).sum(-1)


# Note that this objective function supports ensemble input (`u`) without the use of `apply`.
# Thus -- for this case -- it would have been better had we not used `apply` in `nabla_ens`,
# since multi-threaded `numpy` has less overhead than multiprocessing and is therefore faster.
# Generally speaking, it may *sometimes* be better to leave parallelisation to the
# model/simulator/objective function, since it is "closer to the metal"
# and can therefore do more speed optimisations.
# In fact, due to the overhead of multiprocessing, it is better to make `apply` use a plain for loop for this trivial case,
# which is done by setting `nCPU = False`.

@plotting.interact(case=['Quadratic', 'Rosenbrock', 'Rastrigin'],
                   seed=(1, 10), nTrial=(1, 20), aspect=(-1, 1, .1),
                   nIter=(0, 20), xStep=(0, 30, .1),
                   sdev=(0.01, 5), nEns=(2, 100))
def plot(case, seed=5, nTrial=2, aspect=0, nIter=10, xStep=0,
         sdev=.1, precond=False, nrmlz=True, nEns=10):

    obj.aspect = 10**aspect
    obj.case = case
    fig, axs = plotting.figure12("Toy problems")

    for i in range(nTrial):
        rnd.seed(100*seed + i)
        u0 = rnd.rand(2) * model.domain[1]
        xSteps = [xStep] if xStep else backtracker.xSteps

        utils.nCPU = False  # no multiprocessing
        path, objs, info = GD(obj, u0,
                              nabla_ens(sdev, nEns, precond),
                              backtracker(-1, xSteps),
                              nrmlz=nrmlz, nIter=nIter,
                              quiet=True)
        utils.nCPU = True  # restore

        plotting.add_path12(*axs, path, objs, color=f"C{i}", labels=False)
        axs[1].set_yscale("log")

    model.plt_field(axs[0], obj(mesh2list(*model.mesh)), wells=False, cmap="cividis",
                    norm=plotting.LogNorm() if case in ["Rosenbrock"] else None)


# ## Case: Optimize injector location (x, y)
# Let's try optimising the location (x, y) of the injector well.
# The objective function is simply a thin wrapper around `npv`
# which translates its single (vector) input argument into the appropriate keyword argument,
# and discards all output except the scalar NPV.

def npv_inj_xy(xys):
    return npv(model, inj_xy=xys)[0]

obj = npv_inj_xy
model = original_model

# The model is sufficiently cheap that we can afford to compute the objective
# over its entire 2D domain, and plot it.

npvs = apply(obj, mesh2list(*model.mesh), pbar="obj(mesh)")
npvs = np.asarray(npvs)

# We have in effect conducted an exhaustive computation of the objective function,
# so that we already know the true, global, optimum:

argmax = npvs.argmax()
print("Global (exhaustive search) optimum:", f"{npvs[argmax]:.4}",
      "at (x={:.2}, y={:.2})".format(*model.ind2xy(argmax)))

# Note that the optimum is not quite in the centre of the domain,
# which is caused by the asymmetry of the permeability field.
#
# Now let's try EnOpt from a few different starting/initial guesses,
# and plot the optimisation paths along with the contours of the objective
# *PS: code for both tasks must be in same cell in order to plot on same figure*.

# Optimize, plot paths
fig, axs = plotting.figure12(obj.__name__)
for color in ['C0', 'C2', 'C7', 'C9']:
    u0 = rnd.rand(2) * model.domain[1]
    path, objs, info = GD(obj, u0, nabla_ens(.1))
    plotting.add_path12(*axs, path, objs, color=color, labels=False)
model.plt_field(axs[0], npvs, "NPV", argmax=True, wells=False);

# Note that
#
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
# The setters in `remake` and ResSim simplify defining the objective.
# Still, sometimes we need to pre-process the arguments some more.
# For example, suppose we only want to vary the x-coordinate of the injector(s),
# while keeping the y-coordinate fixed.

# +
def npv_x_with_fixed_y(xs):
    xys = np.stack([xs, xs], -1) # ⇒ (1d or 2d)
    xys[..., 1] = y  # fix constant value
    return npv(model, inj_xy=xys)[0]

y = model.Ly/2
# -

# *PS: The use of `...` is a trick that allows operating on the last axis of `xys`,
# which works both when it's 1d and 2d.*
# Also note that we could of course have re-used `npv_inj_xy` to define `npv_x_with_fixed_y`.
# This will be our approach for the subsequent case.

# +
obj = npv_x_with_fixed_y
model = original_model

x_grid = np.linspace(0, model.Lx, 201)
npvs = apply(obj, x_grid, pbar="obj(x_grid)")

# +
# Plot objective
fig, ax = plotting.freshfig(f"{obj.__name__}({y})", figsize=(7, 3))
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
# The phenomenon is due to the fact that the model always collocates wells with grid cell centres.
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

# Plot optimisation trajectory
fig, axs = plotting.figure12(obj.__name__)
plotting.add_path12(*axs, path[:, :2], objs, color='C1')
plotting.add_path12(*axs, path[:, 2:], color='C3')
model.plt_field(axs[0], model.K[0], "perm");
# -

# Seems reasonable.
# A useful sanity check is provided by inspecting the resulting flow pattern.

plot_final_sweep(remake(model, inj_xy=path[-1], name=f"Optimal for {obj.__name__}"))

# ## Case: Optimize single rate
#
# When setting the injection rate(s), we must also
# set the total production rates to be the same (this is a model constraint),
# and vice-versa.
#
# Thus, as above, we need to pre-compute something before calling `npv()`.

# +
def equalize(rates, nWell):
    """Distribute the total rate equally among `nWell`."""
    return np.tile(rates.sum(0) / nWell, (nWell, 1))

def npv_in_inj_rates(inj_rates):
    prod_rates = equalize(inj_rates, model.nProd)
    return npv(model, inj_rates=inj_rates, prod_rates=prod_rates)[0]

obj = npv_in_inj_rates
model = original_model
# -

# Again we are able and can afford to compute and plot the entire objective.

rate_grid = np.linspace(0.1, 5, 21)
npvs = apply(obj, rate_grid, pbar="obj(rate_grid)")

# It makes sense that there is an optimum sweet spot somewhere in the middle.
# - Little water injection ⇒ little oil production.
# - Much water injection ⇒ very pricey, whereas reservoir contains finite amount of oil.

# +
# Optimize
fig, ax = plotting.freshfig(obj.__name__)
ax.grid()
ax.set(xlabel="rate", ylabel="NPV")
ax.plot(rate_grid, npvs, "slategrey")

for i, u0 in enumerate(np.array([[.1, 5]]).T):
    path, objs, info = GD(obj, u0, nabla_ens(.1))
    shift = i+1  # for visual distinction
    ax.plot(path, objs - shift, '-o', color=f'C{i+1}')
fig.tight_layout()
plotting.show()
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

# Show well layout

fig, ax = plotting.freshfig(model.name)
model.plt_field(ax, model.K[0], "perm");

# Define function that takes injection rates and computes final sweep, i.e. saturation field.
# The plot also displays the resulting NPV.

@plotting.interact(
    inj0_rate = (0, 1.4),
    inj1_rate = (0, 1.4),
    inj2_rate = (0, 1.4),
    inj3_rate = (0, 1.4),
    side="right", wrap=False,
)
def interactive_rate_optim(**kwargs):
    rates = np.array([list(kwargs.values())]).T
    value, info = npv(model, inj_rates=rates, prod_rates=equalize(rates, model.nProd))
    _, ax = plotting.freshfig(f"Interactive controls as for '{obj.__name__}'")
    model.plt_field(ax, info['wsats'][-1], "oil", title=f"Final sweep. Resulting NPV: {value:.2f}")


# #### Automatic (EnOpt) optimisation
# Run EnOpt (below).

u0 = .7*np.ones(model.nInj)
path, objs, info = GD(obj, u0, nabla_ens(.1))
print("Controls suggested by EnOpt:", path[-1])

# Now try setting the resulting suggested values in the interactive widget above.
# Were you able to find better settings?

# ## Case: time-dependent rates

# +
def npv_in_rates(rates, diagnostics=False):
    split_at = nInterval * model.nInj
    inj, prod = rates[:split_at], rates[split_at:]

    inj = rate_transform(inj)
    prod = rate_transform(prod)

    # Balance (at each time) by reducing to lowest
    I, P = inj.sum(0), prod.sum(0)
    inj [:, P<I] *= P[P<I] / I[P<I]
    prod[:, I<P] *= I[I<P] / P[I<P]

    v, o = npv(model, inj_rates=inj, prod_rates=prod)
    return (v, o) if diagnostics else v

obj = npv_in_rates
assert model.name == "Triangle case"
# -

nInterval = 10
rate_min = 0.1
rate_max = 3
def rate_transform(rates):
    """Map `ℝ --> (0, max_rates)`. 'Snap' low rates to 0. Expand in time."""
    duration = int(np.ceil(nTime/nInterval))
    rates = sigmoid(rates, rate_max)
    rates[rates < rate_min] = 0
    rates = rates.reshape((-1, nInterval))
    rates = rates.repeat(duration, 1)[:, :nTime]
    return rates

# Optimize

u0 = -1.4 + 1e-2*rnd.randn(model.nInj + model.nProd, nInterval).ravel()
path, objs, info = GD(obj, u0, nabla_ens(.6, nEns=100))

# Extract diagnostics

value, other = npv_in_rates(path[-1], diagnostics=True)
inj_rates = other['model'].actual_rates['inj']
prod_rates = other['model'].actual_rates['prod']
# prod_rates = other['model'].prod_rates
oil_sats = 1 - prod_sats(model, other['wsats']).T

# #### Plot

# +
fig, (ax1, ax2) = plotting.freshfig("Optimal rates", figsize=(7, 6), nrows=2, sharex=True)
ax_ = ax1.twinx()
for iWell, (rates, satrs) in enumerate(zip(prod_rates, oil_sats)):
    ax1.plot(np.arange(nTime), rates, c=f"C{iWell}", lw=3)
    ax_.plot(np.arange(nTime), satrs, c=f"C{iWell}", lw=1)
ax1.axhline(rate_min, color="k", lw=1, ls="--")
ax1.legend(range(model.nProd), title="Prod. well")
ax1.set_ylabel("Rate")
ax1.grid(True)
ax_.set_ylabel('Saturation')

ax2.invert_yaxis()
for iWell, rates in enumerate(inj_rates):
    ax2.plot(np.arange(nTime), rates, c=f"C{model.nProd + iWell}", lw=3)
ax2.axhline(rate_min, color="k", lw=1, ls="--")
ax2.legend(range(model.nInj), title="Inj. well")
ax2.set(ylabel="Rate", xlabel="Time (index)")
ax2.grid(True)
# -

# Final sweep

_, ax = plotting.freshfig(f"Final sweep -- {obj.__name__}")
model.plt_field(ax, other['wsats'][-1], "oil");


# # Robust optimisation
# Robust optimisation problems have a particular structure,
# namely the objective is an *average*:

def obj(u=None, x=None):
    return np.mean([obj1(u, x) for x in uq_ens])

# Of course, we still have to define the
#
# - ensemble (`uq_ens`) providing the uncertainty quantification (UQ)
#   of some parameter(s), over which the average is computed.
# - conditional objective (`obj1`),
#   thus labelled because it applies to 1 member in `uq_ens`.
#
# This objective function becomes very costly to evaluate,
# since it involves $2 n_{\text{Ens}}$ evaluations/simulations.
# But "there are no problems -- only opportunities" or
# perhaps less pompously "every problem carries the seed of its own solution".
# Indeed we can exploit the structure of the robust-optimisation `obj` above
# by the following patch to the way `nabla_ens` computes the increments of `obj`.

def dJ_robust(self, obj, u, U, pbar):
    if self.robustly == "Paired":
        dJ = apply(self.obj_ux, U, x=self.X, pbar=pbar)

    elif self.robustly == "StoSAG":
        uu = np.tile(u, (self.nEns, 1))  # replicate u
        JU = apply(self.obj_ux, U, x=self.X, pbar=pbar)
        Ju = apply(self.obj_ux, uu, x=self.X, pbar=pbar)
        dJ = np.asarray(JU) - Ju

    elif self.robustly in ["Mean-model", "Fragile"]:
        x1 = np.tile(self.X.mean(0), (self.nEns, 1))  # replicate x
        dJ = apply(self.obj_ux, U, x=x1, pbar=pbar)

    else:  # Regular/"naive" (M*N costly) form
        dJ = apply(obj, U, pbar=pbar)

    return dJ

nabla_ens.obj_increments = dJ_robust

# Note that the computational cost of is $2 n_{\text{Ens}}$ simulations
# rather than $n_{\text{Ens}}^2$ (assuming ensembles of equal size, $n_{\text{Ens}}$).
# For small $n_{\text{Ens}}$ this cost saving is not palpable,
# especially since backtracking will also perform a few iterations of $n_{\text{Ens}}$ evaluations.
# But if $n_{\text{Ens}} > 30$ the cost savings become salient.
#
# PS: The cost of `nabla_ens.obj_increments` could be further reduced to just $n_{\text{Ens}}$ simulations
# by not computing `Ju`, instead obtaining these objective values
# from the latest `backtracker` evaluations.
# However, for the sake of code simplicity,
# we do not implement the necessary intercommunication,
# preferring to keep `GD` and `backtracker` entirely generic,
# i.e. unaware of any conditional objectives.

# ## Case: Optimize injector location (x, y) under uncertain permeability

# ### Uncertainty quantification

nEns = 31
rnd.seed(5)
uq_ens = .1 + np.exp(5 * geostat.gaussian_fields(model.mesh, nEns, r=0.8))

# Plot

plotting.fields(model, uq_ens, "perm");

# ### Conditional objective
# The *conditional* objective consists of the `npv`
# at some `inj_xy=u` for a  given permability `K=x`.

# +
def obj1(u, x):
    return npv(model, inj_xy=u, K=x)[0]

model = original_model
# -

# ### Ensemble of objectives
#
# NB: since it involves $n_{\text{Ens}}$ model simulations for each grid cell,
# computing the ensemble of conditional objective surfaces can take quite long.
# So you should skip these computations if you're on a slow computer.

try:
    import google.colab  # type: ignore
    my_computer_is_fast = False # Colab is slow
except ImportError:
    my_computer_is_fast = True

if my_computer_is_fast:
    npv_mesh = apply(lambda x: [obj1(u, x) for u in mesh2list(*model.mesh)], uq_ens)
    plotting.fields(model, npv_mesh, "NPV", "xy of inj, conditional on perm");

    # Thus we know the global optimum of the total/robust objective.
    npv_avrg = np.mean(npv_mesh, 0)
    argmax = npv_avrg.argmax()
    print("Global (exhaustive search) optimum:",
          f"obj={npv_avrg[argmax]:.4}",
          "(x={:.2}, y={:.2})".format(*model.ind2xy(argmax)))

# ### Optimize, plot paths

# +
fig, axs = plotting.figure12(obj1.__name__)
if my_computer_is_fast:
    model.plt_field(axs[0], npv_avrg, "NPV", argmax=True, wells=False, finalize=False)

    # Use "naive" ensemble gradient
    for color in ['C0', 'C2']:
        u0 = rnd.rand(2) * model.domain[1]
        path, objs, info = GD(obj, u0, nabla_ens(.1, nEns=nEns))
        plotting.add_path12(*axs, path, objs, color=color, labels=False)

# Use StoSAG ensemble gradient
for color in ['C7', 'C9']:
    u0 = rnd.rand(2) * model.domain[1]
    path, objs, info = GD(obj, u0, nabla_ens(.1, nEns=nEns, obj_ux=obj1, X=uq_ens, robustly="StoSAG"))
    plotting.add_path12(*axs, path, objs, color=color, labels=False)

fig.tight_layout();
# -

# Clearly, optimising the full objective with "naive" EnOpt is very costly,
# but gets significantly faster using robust EnOpt (StoSAG), in particular if $n_{\text{Ens}} > 30$.
#
# You may also want to experiment with the other alternatives for `robustly`, i.e. "Mean-model" and "Paired".
#
# Let us store the optimum of the last trial of StoSAG.

ctrl_robust = path[-1]

# ### Nominally (conditionally/individually) optimal controls
#
# It is also (academically) interesting to consider the optimum for the conditional objective,
# i.e. for a single uncertain parameter member/realisation vector in `uq_ens`.
# We can thus generate an ensemble of such nominally optimal control strategies.

ctrl_ens_nominal = []
for x in progbar(uq_ens, desc="Nominal optim."):
    u0 = rnd.rand(2) * model.domain[1]
    path, objs, info = GD(lambda u: obj1(u, x), u0, nabla_ens(.1, nEns=nEns), quiet=True)
    ctrl_ens_nominal.append(path[-1])

# Alternatively, since we have already computed the npv for each pixel/cell
# for each uncertain ensemble member, we can get the global nominal optima.

if my_computer_is_fast:
    ctrl_ens_nominal2 = model.ind2xy(np.asarray(npv_mesh).argmax(axis=1)).T

# The following scatter-plots the nominal optima (`ctrl_ens_nominal`)
# for the injector well. The scatter locations are labelled with the corresponding uncertainty realisation.

fig, ax = plotting.freshfig("Optima")
cmap = plotting.plt.get_cmap('tab20')
lbl_props = dict(fontsize="small", fontweight="bold")
sct_props = dict(s=6**2, edgecolor="w", zorder=3)
lbls = []
for n, (x, y) in enumerate(ctrl_ens_nominal):
    color = cmap(n % 20)
    ax.scatter(x, y, color=color, **sct_props)
    lbls.append(ax.text(x, y, n, c=color, **lbl_props))
    if my_computer_is_fast:
        x2, y2 = ctrl_ens_nominal2[n]
        ax.plot([x, x2], [y, y2], '-', color=color, lw=2)
ax.scatter(*ctrl_robust, s=8**2, color="w")
utils.adjust_text(lbls, precision=.1);
model.plt_field(ax, np.zeros_like(model.mesh[0]), "domain",
                wells=False, colorbar=False, grid=True);

# Also drawn are lines to/from the true/global nominal optima (if `my_computer_is_fast`).
# It can be seen that EnOpt mostly, but not always, finds the global optimum
# for this case.

err = (ctrl_ens_nominal2 - ctrl_ens_nominal)
err /= model.domain[1] # scale to [0, 1]
RMS = np.sqrt(np.mean(err**2, -1))
print(f"Number of significantly suboptimal EnOpt answer: {sum(RMS > 0.1)} of {len(RMS)}")

# ### Histogram (KDE) for each control strategy

# Let us assess (evaluate) the performance of the robust optimal control vector
# for each of the parameter possibilities (i.e. each realisation in `uq_ens`).
# *PS: this was of course already computed as part of the iterative optimisation
# procedure, but was not included it among its outputs.*

npvs_robust = apply(lambda x: obj1(ctrl_robust, x), uq_ens)

# We can do the same for each nominally optimal control vector.
# *PS: we could also do the same for `ctrl_ens_nominal2`.*

npvs_condnl = apply(lambda u: [obj1(u, x) for x in uq_ens], ctrl_ens_nominal)

# Note that `npvs_condnl` is of shape `(nEns, nEns)`,
# the first index (i.e. each row) corresponds to a nominally optimal control parameter vector.
# We can construct a histogram for each one,
# but it's difficult to visualize several histograms together.
# Instead, following [Essen2009](#Essen2009), we use (Gaussian) kernel density estimation (KDE)
# to create a "continuous" histogram, i.e. an approximate probability density.
# The following code is a bit lengthy due to plotting details.

# +
from scipy.stats import gaussian_kde

fig, ax = plotting.freshfig("NPV densities for optimal controls", figsize=(7, 4))
ax.set_xlabel("NPV")
ax.set_ylabel("Density (pdf)");

a, b = np.min(npvs_condnl), np.max(npvs_condnl)
npv_grid = np.linspace(a, b, 100)

lbls = []
for n, npvs_n in enumerate(npvs_condnl):
    color = cmap(n % 20)
    kde = gaussian_kde(npvs_n)
    ax.plot(npv_grid, kde(npv_grid), c=color, lw=1.2, alpha=.7)

    # Label curves
    x = a + n*(b-a)/nEns
    lbls.append(ax.text(x, kde(x).item(), n, c=color, **lbl_props))
    ax.scatter(x, kde(x), s=2, c=color)

# Add robust strategy
ax.plot(npv_grid, gaussian_kde(npvs_robust).evaluate(npv_grid), "w", lw=3)

# Legend showing mean values
leg = (f"         Mean    Min",
       f"Robust:  {np.mean(npvs_robust):<6.3g}  {np.min(npvs_robust):.3g}",
       f"Nominal: {np.mean(npvs_condnl):<6.3g}  {np.min(npvs_condnl):.3g}")
ax.text(.02, .97, "\n".join(leg), transform=ax.transAxes, va="top", ha="left",
        fontsize="medium", fontfamily="monospace", bbox=dict(
            facecolor='lightyellow', edgecolor='k', alpha=0.99,
            boxstyle="round,pad=0.25"))

ax.tick_params(axis="y", left=False, labelleft=False)
ax.set(facecolor="k", ylim=0, xlim=(a, b))
utils.adjust_text(lbls)
fig.tight_layout()
plotting.show()
# -

# # Multi-objective optimisation
# Compared to [Angga2022](#Angga2022) 5-spot case:
#
# - No compressibility
# - Different model rectangle
# - Simplified geology (permeability)
# - Injection is constant in time and across wells
#
# Only the 1st item would be hard to change.

# +
model = remake(original_model,
    name = "Angga2022-5spot",
    prod_xy = [[model.Lx/2, model.Ly/2]],
    inj_xy = xy_4corners,
    prod_rates = rate0 * np.ones((1, 1)) / 1,
    inj_rates = rate0 * np.ones((4, 1)) / 4,
)

plot_final_sweep(model)


# +
def npv_in_prod_rates(prod_rates):
    inj_rates = equalize(prod_rates, model.nInj)
    return npv(model, prod_rates=prod_rates, inj_rates=inj_rates)[0]

obj = npv_in_prod_rates
# -

# ### Optimize

fig, ax = plotting.freshfig(obj.__name__)
rate_grid = np.logspace(-2, 1, 31)
optimal_rates = []
# cost_multiplier = [.01, .04, .1, .4, .9, .99]
__default__ = price['inj']
cost_multiplier = np.arange(0.1, 1, 0.1)
for i, xCost in enumerate(cost_multiplier):
    price['inj'] = __default__ * xCost
    npvs = apply(obj, rate_grid, pbar="obj(rate_grid)")
    ax.plot(rate_grid, npvs, label=f"{xCost:.1}")
    path, objs, info = GD(obj, np.array([2]), nabla_ens(.1))
    optimal_rates.append(path[-1])
price['inj'] = __default__  # restore
ax.set_ylim(1e-2)
ax.legend(title="×price_of_inj")
ax.set(xlabel="rate", ylabel="NPV")
ax.grid()
fig.tight_layout()
plotting.show()

# ### Pareto front
# Breakdown npv (into emissions and sales) for optima

sales = []
emissions = []
for i, prod_rates in enumerate(optimal_rates):
    inj_rates = equalize(prod_rates, model.nInj)
    value, other = npv(model, prod_rates=prod_rates, inj_rates=inj_rates)
    sales.append(other['oil_total'])
    emissions.append(other['inj_total'])


fig, ax = plotting.freshfig("Pareto front (npv-optimal settings for range of price['inj'])")
ax.set(xlabel="npv (income only)", ylabel="inj/emissions (expenses)")
ax.plot(sales, emissions, "o-")
ax.grid()
fig.tight_layout()

# # References

# <a id="Essen2009">[Essen2009]</a>: van Essen, G., M. Zandvliet, P. Van den Hof, O. Bosgra, and J.-D. Jansen. *Robust waterflooding optimization of multiple geological scenarios.* **SPE Journal**, 14(01):202–210, 2009.
#
# <a id="Angga2022">[Angga2022]</a>: Angga, I. G. A. G., M. Bellout, B. S. Kristoffersen, P. E. S. Bergmo, P. A. Slotte, and C. F. Berg. *Effect of CO2 tax on energy use in oil production: waterflooding optimization under different emission costs.* **SN Applied Sciences**, 4(11):313, 2022
#
# <a id="">[Fonseca2017]</a>: Fonseca, R. M., B. Chen, J. D. Jansen, and A. Reynolds. *A stochastic simplex approximate gradient (StoSAG) for optimization under uncertainty.* **International Journal for Numerical Methods in Engineering**, 109(13):1756– 1776, 2017. doi: 10.1002/nme.5342.
#
# <a id="">[Raanes2023]</a>: Raanes, P. N., A. S. Stordal, and R. J. Lorentzen. *Review of ensemble gradients for robust optimisation*, 2023. doi: 10.48550/arXiv.2304.12136
#
# <a id="">[Chen2009]</a>: Chen, Y., D. S. Oliver, and D. Zhang. *Efficient ensemble-based closed- loop production optimization.* **SPE Journal**, 14(04):634–645, 2009. doi: 10.2118/112873-PA.
