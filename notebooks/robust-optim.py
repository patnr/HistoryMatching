# ## Imports

import copy
from dataclasses import dataclass

import numpy as np
import numpy.random as rnd
import TPFA_ResSim as simulator

from tools import geostat, plotting, utils
from tools.utils import center, apply, progbar

plotting.init()
np.set_printoptions(precision=4, sign=' ', floatmode="fixed")

# ## Define model

model = simulator.ResSim(Nx=14, Ny=10, Lx=2, Ly=1, name="Base model")

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
# fig, ax = plotting.freshfig(model.name, figsize=(1, .6), rel=True)
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
#     fig, ax = plotting.freshfig(title, figsize=(1, .6), rel=True)
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

@dataclass
class nabla_ens:
    """Ensemble gradient estimate (LLS regression)."""
    chol:    float = 1.0   # Cholesky factor (or scalar std. dev.)
    nEns:    int   = 10    # Size of control perturbation ensemble
    precond: bool  = False # Use preconditioned form?
    conditional: None = None
    X:           None = None  # Uncertainty ensemble

    def apply(self, obj, u, pbar):
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
        """Compute `dJ := center(obj(U))`."""
        obj1 = self.conditional
        if obj1:
            u1 = np.tile(u, (self.nEns, 1))  # replicate u
            JU = apply(obj1, U, x=self.X, pbar=pbar)
            Ju = apply(obj1, u1, x=self.X, pbar=pbar)
            dJ = np.asarray(JU) - Ju
        else:
            dJ = apply(obj, U, pbar=pbar)  # can omit `center`
        return dJ

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
# from the latest `backtracker` evaluations.
# However, for the sake of code simplicity,
# we do not implement the necessary intercommunication,
# preferring to keep `GD` and `backtracker` entirely generic,
# i.e. unaware of the robust objective possiblity.

@dataclass
class backtracker:
    """Bisect until improvement."""
    sign:   int   = +1                                  # Search for max(+1) or min(-1)
    xSteps: tuple = tuple(.5**(i+1) for i in range(8))  # Trial step lengths
    rtol:   float = 1e-8                                # Convergence criterion
    def apply(self, obj, u0, J0, search_direction, pbar):
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


def GD(objective, u, nabla=nabla_ens(), line_search=backtracker(), nrmlz=True, nIter=100, verbose=True):
    """Gradient (i.e. steepest) descent/ascent."""

    # Reusable progress bars (limits flickering scroll in Jupyter) with short np printout
    with (progbar(total=nIter, desc="⏳ GD running", leave=True,  disable=not verbose) as pbar_gd,
          progbar(total=10000, desc="→ grad. comp.", leave=False, disable=not verbose) as pbar_en,
          progbar(total=10000, desc="→ line_search", leave=False, disable=not verbose) as pbar_ls,
          np.printoptions(precision=2, threshold=2, edgeitems=1)):

        states = [[u, objective(u), "placeholder for {cause}"]]

        for itr in range(nIter):
            u, J, info = states[-1]
            pbar_gd.set_postfix(u=u, obj=J)

            grad = nabla.apply(objective, u, pbar_en)
            if nrmlz:
                grad /= np.sqrt(np.mean(grad**2))
            updated = line_search.apply(objective, u, J, grad, pbar_ls)
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

# ### Plot ensemble (perm)

plotting.fields(model, uncertainty_ens, "perm");

# ### Robust obj

def obj(u=None, x=None):
    return np.mean([obj1(u, x) for x in uncertainty_ens])


# ## Case: Optimize injector location (x, y)
# The *conditional* objective consists of the `npv`
# at some `inj_xy=u` for a (i.e. "one" whence "`obj1`") given permability `K=x`.

def obj1(u, x):
    return npv(model, inj_xy=u, K=x)[0]

model = original_model
# print(f"Case: '{obj1.__name__}' for '{model.name}'")

# ### Plot ensemble of objectives
#
# NB: since it involves $N$ model simulations for each grid cell,
# this can take quite long.

print("obj1(u=mesh, x=ens)")
XY = np.stack(model.mesh, -1).reshape((-1, 2))
npv_mesh = apply(lambda x: [obj1(u, x) for u in XY], uncertainty_ens)
plotting.fields(model, npv_mesh, "NPV", "xy of inj, conditional on perm");

# ### Total/robust global optimum

npv_avrg = np.mean(npv_mesh, 0)
argmax = npv_avrg.argmax()
print("Global (exhaustive search) optimum:",
      f"obj={npv_avrg[argmax]:.4}",
      "(x={:.2}, y={:.2})".format(*model.ind2xy(argmax)))

# #### Optimize, plot paths
# EnOpt therefore becomes much slower,
# since each of its $N$ control ensemble members at a given iteration
# requires $M$ model simulations.

# +
fig, axs = plotting.figure12(obj1.__name__)
model.plt_field(axs[0], npv_avrg, "NPV", argmax=True, wells=False);

# Use "naive" ensemble gradient
for color in ['C0', 'C2']:
    u0 = rnd.rand(2) * model.domain[1]
    path, objs, info = GD(obj, u0, nabla_ens(.1, nEns=nEns))
    plotting.add_path12(*axs, path, objs, color=color, labels=False)

# Use StoSAG ensemble gradient
for color in ['C7', 'C9']:
    u0 = rnd.rand(2) * model.domain[1]
    path, objs, info = GD(obj, u0, nabla_ens(.1, nEns=nEns, conditional=obj1, X=uncertainty_ens))
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

# Alternatively, we can get the globally nominally optima since we have already computed the npv for each pixel/cell for each uncertain ensemble member.

nominal_ctrl_ens_global = model.ind2xy(np.asarray(npv_mesh).argmax(axis=1)).T

# #### Plot optima (`nominal_ctrl_ens`)

cmap = plotting.plt.get_cmap('tab20')
fig, ax = plotting.freshfig("Optima", figsize=(7, 4))
model.plt_field(ax, np.zeros_like(model.mesh[0]), "domain",
                wells=False, colorbar=False, grid=True);
lbl_props = dict(fontsize="large", fontweight="bold")
lbls = []
for n, (x, y) in enumerate(nominal_ctrl_ens):
    color = cmap(n % 20)
    ax.scatter(x, y, s=6**2, color=color, lw=.5, edgecolor="w")
    lbls.append(ax.text(x, y, n, c=color, **lbl_props))
    if True:
        x2, y2 = nominal_ctrl_ens_global[n]
        ax.plot([x, x2], [y, y2], '-', color=color, lw=.5)
ax.scatter(*robust_ctrl, s=8**2, color="w")
utils.adjust_text(lbls, precision=.1);
fig.tight_layout()

# It can be seen that EnOpt mostly, but not always, finds the global optimum
# for this case.
#
# ### Histogram (KDE) for each control strategy
# We can assess (evaluate) each nominally optimal control vector
# for each realisation in `uncertainty_ens`.

print("obj1(ctrl_ens, uncertainty_ens)")
npvs_condnl = apply(lambda u: [obj1(u, x) for x in uncertainty_ens], nominal_ctrl_ens)
npvs_robust = apply(lambda x: obj1(robust_ctrl, x), uncertainty_ens)

# Note that `npvs_condnl` is of shape `(nEns, nEns)`,
# each row corresponding to a nominally optimal control parameter vector.
# We can construct a histogram for each one,
# but it's difficult to visualize several histograms together.
# Instead, following [Essen2009](#Essen2009), we use (Gaussian) kernel density estimation (KDE)
# to create a "continuous" histogram, i.e. an approximate probability density.

# #### Plot
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
    ax.plot(npv_grid, kde(npv_grid), c=color, lw=.8, alpha=.7)

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
# -

# ## References

# <a id="Essen2009">[Essen2009]</a>: van Essen, G., M. Zandvliet, P. Van den Hof, O. Bosgra, and J.-D. Jansen. Robust waterflooding optimization of multiple geological scenarios. SPE Journal, 14(01):202–210, 2009.
