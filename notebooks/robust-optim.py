# ## Imports

import copy

import numpy as np
import numpy.random as rnd
import TPFA_ResSim as simulator
from mpl_tools.place import freshfig
from adjustText import adjust_text
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

import tools.plotting as plotting
from tools import geostat, mpl_setup, utils
from tools.utils import center, apply, progbar

mpl_setup.init()
np.set_printoptions(precision=4, sign=' ', floatmode="fixed")

# ## Define model

model = simulator.ResSim(Nx=4, Ny=5, Lx=2, Ly=1, name="Base model")

# +
# seed = rnd.seed(3)
# model.K = .1 + np.exp(5 * geostat.gaussian_fields(model.mesh, 1, r=0.8))
# -

near01 = np.array([.12, .87])
xy_4corners = [[x, y]
               for y in model.Ly*near01
               for x in model.Lx*near01]

rate0 = 1.5

model.inj_xy = [[model.Lx/2, model.Ly/2]]
model.prod_xy = xy_4corners
model.inj_rates  = rate0 * np.ones((1, 1)) / 1
model.prod_rates = rate0 * np.ones((4, 1)) / 4

# +
# fig, ax = freshfig(model.name, figsize=(1, .6), rel=True)
# model.plt_field(ax, model.K[0], "perm");
# fig.tight_layout()
# -

wsat0 = np.zeros(model.Nxy)
T = 1
dt = 0.025
nTime = round(T/dt)

# +
# def plot_final_sweep(model):
#     """Simulate reservoir, plot final oil saturation."""
#     wsats = model.sim(dt, nTime, wsat0, pbar=False)
#     title = "Final sweep" + (" -- " + model.name) if model.name else ""
#     fig, ax = freshfig(title, figsize=(1, .6), rel=True)
#     model.plt_field(ax, wsats[-1], "oil")
#     fig.tight_layout()

# plot_final_sweep(model)
# -

# ## NPV objective function

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

price_of_inj = 5e3
price_of_oil = 1e4
discounts = .99 ** np.arange(nTime)

def remake(model, **params):
    """Instantiate new model config."""
    model = copy.deepcopy(model)
    for k, v in params.items():
        setattr(model, k, v)
    return model

original_model = remake(model)

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

utils.nCPU = True

def color_noise(Z, L):
    """Multiply `Z` (1d or 2d) by Cholesky factor -- or scalar -- `L`."""
    if getattr(L, 'ndim', 0):
        X = Z @ L.T
    else:
        X = Z * L
    return X

def nabla_ens(chol=1.0, nEns=10, precond=False, normed=True, robust=False):
    """Set parameters of `ens_grad`."""
    def ens_grad(obj, u, pbar):
        """Compute ensemble gradient (LLS regression) for `obj` centered on `u`."""
        # Ensemble of controls (U)
        U = color_noise(rnd.randn(nEns, len(u)), chol)
        dU = center(U)[0]
        U = u + dU

        # Increments (dJ)
        if robust:
            u1 = np.tile(u, (nEns, 1))  # replicate u
            JU = apply(obj, U, x=uncertainty_ens, pbar=pbar)
            Ju = apply(obj, u1, x=uncertainty_ens, pbar=pbar)
            dJ = np.asarray(JU) - Ju
        else:
            dJ = apply(obj, U, pbar=pbar)

        if precond:
            g = dU.T @ dJ / (nEns-1)
        else:
            g = utils.rinv(dU, reg=.1, tikh=True) @ dJ

        if normed:
            g /= utils.mnorm(g)

        return g
    return ens_grad

# Note that `ens_grad` is aware that the objective might also accept
# a second input argument, namely the uncertain parameter vector for robust objective problems:
# it implements a separate block to exploit this fact,
# whose computational cost is $2 N$ simulations
# rather than $N^2$ (assuming ensembles of equal size, $N$).
# For small $N$ this cost saving is not palpable,
# especially since backtracking will also perform a few iterations of $N$ evaluations.
# But if $N > 30$ the cost savings become salient.
#
# PS: The cost of `ens_grad` could be further reduced to just $N$ simulations
# by not computing `Ju`, instead obtaining these objective values
# from the latest `backtrack` evaluations.
# However, for the sake of code simplicity,
# we do not implement the necessary intercommunication,
# preferring to keep `GD` and `backtracker` entirely generic,
# i.e. unaware of the robust objective possiblity.

def backtracker(sign=+1, xSteps=tuple(1/2**(i+1) for i in range(8)), rtol=1e-8):
    """Set parameters of `backtrack`."""
    def backtrack(objective, u0, J0, search_direction, pbar):
        """Line search by bisection."""
        atol = max(1e-8, abs(J0)) * rtol
        pbar.reset(len(xSteps))
        for i, step_length in enumerate(xSteps):
            du = sign * step_length * search_direction
            u1 = u0 + du
            J1 = objective(u1)
            dJ = J1 - J0
            pbar.update()
            if sign*dJ > atol:
                pbar.reset(len(xSteps))
                return u1, J1, dict(nDeclined=i)
    return backtrack


def GD(objective, u, nabla=nabla_ens(), line_search=backtracker(), nIter=100, verbose=True):
    """Gradient (i.e. steepest) descent/ascent."""

    # Reusable progress bars (limits flickering scroll in Jupyter) with short np printout
    with (progbar(total=nIter, desc="⏳ GD running", leave=True,  disable=not verbose) as pbar_gd,
          progbar(total=10000, desc="→ ens_grad",    leave=False, disable=not verbose) as pbar_en,
          progbar(total=10000, desc="→ backtrack",   leave=False, disable=not verbose) as pbar_ls,
          np.printoptions(precision=2, threshold=2, edgeitems=1)):

        states = [[u, objective(u), "placeholder for {cause}"]]

        for itr in range(nIter):
            u, J, info = states[-1]
            pbar_gd.set_postfix(u=u, obj=J)

            grad = nabla(objective, u, pbar_en)
            updated = line_search(objective, u, J, grad, pbar_ls)
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

# ## Uncertain ens settings

rnd.seed(5)
nEns = 31
uncertainty_ens = .1 + np.exp(5 * geostat.gaussian_fields(model.mesh, nEns, r=0.8))


def obj(u=None, x=None):
    """Evaluate `obj(u, x)` for different sources of `(u, x)`."""
    if u is None:
        # Used for plotting objective for entire domain
        return np.asarray([obj1(u, x) for u in ctrl_ens])
    elif x is None:
        # Defines robust objective
        return np.mean([obj1(u, x) for x in uncertainty_ens])
    else:
        # A single, conditional objective evaluation
        return obj1(u, x)


# ## Case: Optimize injector location (x, y)

def obj1(u, x):
    """Objective (`npv` in `inj_xy=u`), conditional on given permability (`K=x`)."""
    return npv(model, inj_xy=u, K=x)[0]

model = original_model
# print(f"Case: '{obj.__name__}' for '{model.name}'")

# ### Plot ensemble (perm)

plotting.fields(model, uncertainty_ens, "perm");

# ### Plot ensemble of objectives

print("obj(u=mesh, x=ens)")
ctrl_ens = np.stack(model.mesh, -1).reshape((-1, 2))
npv_ens = apply(obj, x=uncertainty_ens)
plotting.fields(model, npv_ens, "NPV", "xy of inj, conditional on perm");

# ### Total/robust objective
# Note that this can take quite long,
# since it involves $N$ model simulations for each grid cell.

npv_avrg = np.mean(npv_ens, 0) # == np.asarray(apply(obj1, ctrl_ens))
argmax = npv_avrg.argmax()
print("Global (exhaustive search) optimum:", f"{npv_avrg[argmax]:.4}",
      "at (x={:.2}, y={:.2})".format(*model.ind2xy(argmax)))

# #### Optimize, plot paths
# EnOpt can also take quite long, since it involves $N$ model simulations,
# for each control ensemble member (of which we also use $N$),
# and each iteration.

# +
fig, axs = plotting.figure12(obj.__name__)
model.plt_field(axs[0], npv_avrg, "NPV", argmax=True, wells=False);

# Use "naive" ensemble gradient
for color in ['C0', 'C2']:
    u0 = rnd.rand(2) * model.domain[1]
    path, objs, info = GD(obj, u0, nabla_ens(.1, nEns=nEns))
    plotting.add_path12(*axs, path, objs, color=color, labels=False)

# Use StoSAG ensemble gradient
for color in ['C7', 'C9']:
    u0 = rnd.rand(2) * model.domain[1]
    path, objs, info = GD(obj, u0, nabla_ens(.1, robust=True, nEns=nEns))
    plotting.add_path12(*axs, path, objs, color=color, labels=False)

fig.tight_layout()
# -

# Clearly, optimising the full objective is very costly, and will therefore not be pursued further.
# Let us store one of the trials of EnOpt with StoSAG robust ensemble gradient.

robust_ctrl = path[-1]

# ### Nominally (conditionally/individually) optimal controls
#
# It is also interesting to consider the optimum for the conditional objective,
# i.e. given an uncertain parameter member/realisation in `uncertainty_ens`.
# We can thus generate an ensemble of such nominally optimal control strategies.

nominal_ctrl_ens = []
for x in progbar(uncertainty_ens, desc="Nominal optim."):
    u0 = rnd.rand(2) * model.domain[1]
    path, objs, info = GD(lambda u: obj1(u, x), u0, nabla_ens(.1, nEns=nEns), verbose=False)
    nominal_ctrl_ens.append(path[-1])

# #### Plot optima (`nominal_ctrl_ens`)

cmap = plt.get_cmap('tab20')
fig, ax = freshfig("Optima", figsize=(7, 4))
model.plt_field(ax, np.zeros_like(model.mesh[0]), "domain",
                wells=False, colorbar=False, grid=True);
lbl_props = dict(fontsize="large", fontweight="bold")
lbls = []
for n, (x, y) in enumerate(nominal_ctrl_ens):
    color = cmap(n % 20)
    ax.scatter(x, y, s=6**2, color=color, lw=.5, edgecolor="w")
    lbls.append(ax.text(x, y, n, c=color, **lbl_props))
ax.scatter(*robust_ctrl, s=8**2, color="w")
adjust_text(lbls, precision=.1);
fig.tight_layout()

# It can be seen that EnOpt mostly, but not always, finds the global optimum
# for this case.
#
# ### Histogram (KDE) for each control strategy
# We can assess (evaluate) each nominally optimal control vector
# for each realisation in `uncertainty_ens`.

print("obj(ens, ens)")
ctrl_ens = [robust_ctrl] + nominal_ctrl_ens
npvs_ens = apply(obj, x=uncertainty_ens)
npvs_ens = np.asarray(npvs_ens).T
npvs_robust = npvs_ens[0]
npvs_condnl = npvs_ens[1:]

# Note that `npvs_condnl` is of shape `(nEns, nEns)`,
# each row corresponding to a nominally optimal control parameter vector.
# We can construct a histogram for each one,
# but it's difficult to visualize several histograms together.
# Instead, following [Essen2009](#Essen2009), we use (Gaussian) kernel density estimation (KDE)
# to create a "continuous" histogram, i.e. an approximate probability density.

# #### Plot
# The following code is a bit lengthy due to plotting details.

# +
fig, ax = freshfig("NPV densities for optimal controls", figsize=(7, 4))
ax.set_xlabel("NPV")
ax.set_ylabel("Density (pdf)");

a, b = npvs_condnl.min(), npvs_condnl.max()
npv_grid = np.linspace(a, b, 100)

lbls = []
for n, npvs_n in enumerate(npvs_condnl):
    color = cmap(n % 20)
    kde = gaussian_kde(npvs_n)
    ax.plot(npv_grid, kde(npv_grid), c=color, lw=.8, alpha=.7)

    # Label curves
    x = a + n*(b-a)/nEns
    lbls.append(ax.text(x, kde(x).item(), n, c=color, **lbl_props))
    ax.scatter(x, kde(x), s=2, c=color)

# Add robust strategy
ax.plot(npv_grid, gaussian_kde(npvs_robust).evaluate(npv_grid), "w", lw=3)

# Legend showing mean values
leg = (f"         Mean    Min",
       f"Robust:  {npvs_robust.mean():<6.3g}  {npvs_robust.min():.3g}",
       f"Nominal: {npvs_condnl.mean():<6.3g}  {npvs_condnl.min():.3g}")
ax.text(.02, .97, "\n".join(leg), transform=ax.transAxes, va="top", ha="left",
        fontsize="medium", fontfamily="monospace", bbox=dict(
            facecolor='lightyellow', edgecolor='k', alpha=0.99,
            boxstyle="round,pad=0.25"))

ax.tick_params(axis="y", left=False, labelleft=False)
ax.set(facecolor="k", ylim=0, xlim=(a, b))
adjust_text(lbls)
fig.tight_layout()
# -

# ## References

# <a id="Essen2009">[Essen2009]</a>: van Essen, G., M. Zandvliet, P. Van den Hof, O. Bosgra, and J.-D. Jansen. Robust waterflooding optimization of multiple geological scenarios. SPE Journal, 14(01):202–210, 2009.
