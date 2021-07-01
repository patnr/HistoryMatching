# -*- coding: utf-8 -*-
# # Tutorial on ensemble history matching and optimisation
#
# Copyright Patrick N. Raanes, NORCE, 2020.
#
# This (Jupyter/Python) notebook is a self-contained tutorial on
# history matching (HM) using ensemble methods.
# Please do not hesitate to file issues on GitHub,
# or submit pull requests.

# ## The Jupyter notebook format
# Notebooks combine **cells** of code (Python) with cells of text (markdown).
# For example, try to edit the cell below to insert your name, and then run it.

name = "Batman"
print("Hello world! I'm " + name)

# You will likely be more efficient if you know these **keyboard shortcuts**:
#
# | Navigate                      |    | Edit              |    | Exit           |    | Run                              |    | Run & advance                     |
# | -------------                 | -- | ----------------- | -- | --------       | -- | -------                          | -- | -------------                     |
# | <kbd>↓</kbd> and <kbd>↑</kbd> |    | <kbd>Enter</kbd>  |    | <kbd>Esc</kbd> |    | <kbd>Ctrl</kbd>+<kbd>Enter</kbd> |    | <kbd>Shift</kbd>+<kbd>Enter</kbd> |
#
# When you open a notebook it starts a **session (kernel/runtime)** of Python
# in the background.  All of the code cells (in a given notebook) are connected
# (they use the same kernel and thus share variables, functions, and classes).
# Thus, the **order** in which you run the cells matters.  One thing you must
# know is how to **restart** the session, so that you can start over. Try to
# locate this option via the top menu bar.

# There is a huge amount of libraries available in **Python**,
# including the popular `numpy` and `matplotlib/pyplot` packages.
# These are imported (and abbreviated) as `np`, and `mpl` and `plt`.
# Try them out by running the following, which illustrates some algebra
# using syntax reminiscent of Matlab.

import numpy as np
from matplotlib import pyplot as plt
from tools import mpl_setup
mpl_setup.init()

# Use numpy's arrays for vectors and matrices. Example constructions:
a  = np.arange(10)  # Alternatively: np.array([0,1,2,3,4,5,6,7,8,9])
Id = 2*np.eye(10)   # Alternatively: np.diag(2*np.ones(10))

print("Indexing examples:")
print("a         =", a)
print("a[3]      =", a[3])
print("a[0:3]    =", a[0:3])
print("a[:3]     =", a[:3])
print("a[3:]     =", a[3:])
print("a[-1]     =", a[-1])
print("Id[:3,:3] =", Id[:3, :3], sep="\n")

print("\nLinear algebra examples:")
print("100+a  =", 100+a)
print("Id@a   =", Id@a)
print("Id*a   =", Id*a, sep="\n")

plt.title("Plotting example")
plt.ylabel("$i \\, x^2$")
for i in range(4):
    plt.plot(i * a**2, label="i = %d" % i)
plt.legend();

# Run the following cells to import yet more tools.

import copy
import numpy.random as rnd
import scipy.linalg as sla
from IPython.display import display
from ipywidgets import interact
from matplotlib.ticker import LogLocator
from mpl_tools.place import freshfig
from numpy import sqrt
from struct_tools import DotDict as Dict
from tqdm.auto import tqdm as progbar

# ## Model and case specification

# For exact reproducibility of our problem/case, we set the random generator seed.

seed = rnd.seed(4)  # very easy
# seed = rnd.seed(5)  # hard
# seed = rnd.seed(6)  # very easy
# seed = rnd.seed(7)  # easy

# Our reservoir simulator takes up about 100 lines of python code. This may seem borderline too simple, but serves the purpose of *illustrating* the main features of the history matching process. Indeed, we do not detail the code here, but simply import it from the accompanying python modules. We also import some associated tools, e.g. for plotting, whose details we shall not belabour. If you want to inspect/modify this code, have a look in the git repository.

import simulator
import simulator.plotting as plots
from tools import geostat, misc
from tools.misc import center

# In short, the model is a 2D, two-phase, immiscible, incompressible simulator using two-point flux approximation (TPFA) discretisation. It was translated from the Matlab code here http://folk.ntnu.no/andreas/papers/ResSimMatlab.pdf

model = simulator.ResSim(Nx=20, Ny=20, Lx=2, Ly=1)

# Plot configuration
plots.model = model
plots.field.coord_type = "absolute"
plots.field.levels = np.linspace(-3.8, 3.8, 21)
plots.field.ticks = np.arange(-3, 4)
plots.field.cmap = "jet"

# The following declares some data containers to help us keep organised.
# The names have all been shortened to 4 characters, but this is just
# to obtain more convenient code alignment for readability.

# +
# Permeability
perm = Dict()

# Production
prod = Dict(
    past = Dict(),
    futr = Dict(),
)

# Water saturation
wsat = Dict(
    init = Dict(),
    past = Dict(),
    futr = Dict(),
)
# -

# Technical note: This data hierarchy is convienient in *this* notebook/script, especially for plotting purposes. For example, we can with ease refer to `wsat.past.Truth` and `wsat.past.Prior`. The former will be a numpy array of shape `(nTime, M)` where `M = model.M`, and the latter will have shape `(N, nTime, M)` where `N` is the size of the ensemble. However, in other implementations, different choices of data structure may be more convenient, e.g. where the different components of the unknowns are merely concatenated along the last axis, rather than being kept in separate dicts.

# #### Permeability sampling
# We will estimate the log permeability field.
# We parameterize the permeability parameters via some transform, which becomes part of the forward model. We term the parameterized permeability fields "pre-permeability". *If* we use the exponential, then we will we working with log-permeabilities. At any rate, the transform should be chosen so that the parameterized permeabilities are suited for ensemble methods, i.e. are distributed as a Gaussian.  But this consideration must be weighted against the fact that that nonlinearity (which is also a difficulty for ensemble methods) in the transform might add to the nonlinearity of the total/composite forward model.  In any case, since this is a synthetic case, we can freely choose *both* the distribution of the parameterized permeabilities, *and* the transform.  Here we use Gaussian fields, and a "perturbed" exponential function (to render the problem a little more complex).

def sample_prior_perm(N=1):
    lperms = geostat.gaussian_fields(model.mesh(), N, r=0.8)
    return lperms

def perm_transf(x):
    return .1 + np.exp(5*x)
    # return 1000*np.exp(3*x)

# For any type of parameter, one typically has to write a "setter" function that takes the vector of parameter parameter values, and applies it to the specific model implementation. We could merge this functionality with `perm_transf` (and indeed the "setter" function is also part of the composite forward model) but it is convenient to separate these implementation specifics from the mathematics going on in `perm_transf`.

def set_perm(model, log_perm_array):
    """Set perm. in model code. Duplicates the perm. values in x- and y- dir."""
    p = perm_transf(log_perm_array)
    p = p.reshape(model.shape)
    model.Gridded.K = np.stack([p, p])

# Now we are in position to sample the permeability of the (synthetic) truth.

perm.Truth = sample_prior_perm()
set_perm(model, perm.Truth)

# #### Well specification
# In this model, wells are represented simply by point **sources** and **sinks**. This is of course incredibly basic and not realistic, but works for our purposes. So all we need to specify is their placement and flux (which we will not vary in time). The code below puts wells on a grid. Try `print(grid2)` to see how to easily specify another well configuration.
#
# Since the **boundary conditions** are Dirichlet, specifying *zero flux*, and the fluid is incompressible, the total of the source terms must equal that of the sinks. This is ensured by the `config_wells` function used below.

grid1 = [.1, .9]
grid2 = np.dstack(np.meshgrid(grid1, grid1)).reshape((-1, 2))
rates = np.ones((len(grid2), 1))  # ==> all wells use the same (constant) rate
model.config_wells(
    # Each row in `inj` and `prod` should be a tuple: (x, y, rate),
    # where x, y ∈ (0, 1) and rate > 0.
    inj  = [[0.50, 0.50, 1.00]],
    prod = np.hstack((grid2, rates)),
);

# #### Plot true field
# Let's take a moment to visualize the model permeability field, and the well locations.

fig, ax = freshfig("True perm. field", figsize=(1.5, 1), rel=1)
# cs = plots.field(ax, perm.Truth)
cs = plots.field(ax, perm_transf(perm.Truth),
                 locator=LogLocator(), cmap="viridis", levels=10)
plots.well_scatter(ax, model.producers, inj=False)
plots.well_scatter(ax, model.injectors, inj=True)
fig.colorbar(cs);


# #### Define obs operator
# The data will consist in the water saturation of at the well locations, i.e. of the production. I.e. there is no well model. It should be pointed out, however, that ensemble methods technically support observation models of any complexity, though your accuracy mileage may vary (again, depending on the incurred nonlinearity and non-Gaussianity). Furthermore, it is also no problem to include time-dependence in the observation model.

obs_inds = [model.xy2ind(x, y) for (x, y, _) in model.producers]
def obs_model(water_sat):
    return water_sat[obs_inds]

# #### Simulation to generate the synthetic truth evolution and data

wsat.init.Truth = np.zeros(model.M)
T     = 1
dt    = 0.025
nTime = round(T/dt)
(wsat.past.Truth,
 prod.past.Truth) = misc.repeat(model.step, nTime, wsat.init.Truth, dt, obs_model)

# ##### Animation
# Run the code cells below to get an animation of the oil saturation evolution. Injection/production wells are marked with triangles pointing down/up.

# %%capture
animation = plots.dashboard(perm.Truth, wsat.past.Truth, prod.past.Truth);

# Note: can take up to a minute to appear
animation

# #### Noisy obs
# In reality, observations are never perfect. To emulate this, we corrupt the observations by adding a bit of noise.

prod.past.Noisy = prod.past.Truth.copy()
nProd           = len(model.producers)  # num. of obs (each time)
R               = 1e-3 * np.eye(nProd)
for iT in range(nTime):
    prod.past.Noisy[iT] += sqrt(R) @ rnd.randn(nProd)


# Plot of observations (and their noise):

fig, ax = freshfig("Observations")
plots.production1(ax, prod.past.Truth, prod.past.Noisy);

# ## Prior
# The prior ensemble is generated in the same manner as the (synthetic) truth, using the same mean and covariance.  Thus, the members are "statistically indistinguishable" to the truth. This assumption underlies ensemble methods.

N = 200
perm.Prior = sample_prior_perm(N)

# Note that field (before transformation) is Gaussian with (expected) mean 0 and variance 1.
print("Prior mean:", np.mean(perm.Prior))
print("Prior var.:", np.var(perm.Prior))

# Let us inspect the parameter values in the form of their histogram.

fig, ax = freshfig("Perm. -- marginal distribution", figsize=(1.5, .7), rel=1)
bins = np.linspace(*plots.field.levels[[0, -1]], 32)
for label, perm_field in perm.items():
    ax.hist(perm_transf(perm_field.ravel()),
            perm_transf(bins),
            # "Downscale" counts by N, coz `density=1` "fails" with log-scale.
            weights=(np.ones(model.M*N)/N if label != "Truth" else None),
            label=label, alpha=0.3)
ax.set(xscale="log", xlabel="Permeability", ylabel="Count")
ax.legend()
fig.tight_layout()

# Since the x-scale is logarithmic, the prior's histogram should look Gaussian if `perm_transf` is purely exponential. By contrast, the historgram of the truth is from a single (spatially extensive) realisation, and therefore will contain significant sampling "error".

# Below we can see some realizations (members) from the ensemble.

plots.fields(plots.field, perm.Prior, "Prior");

# #### Eigenvalue spectrum
# In practice, of course, we would not be using an explicit `Cov` matrix when generating the prior ensemble, because it would be too large.  However, since this synthetic case in being made that way, let's inspect its spectrum.

U, svals, VT = sla.svd(perm.Prior)
ii = 1 + np.arange(len(svals))
fig, ax = freshfig("Spectrum of prior cov.", figsize=(1.6, .7), rel=1)
ax.loglog(ii, svals)
# ax.semilogx(ii, svals)
ax.grid(True, "minor", axis="x")
ax.grid(True, "major", axis="y")
ax.set(xlabel="eigenvalue #", ylabel="variance");

# With our limited ensemble size, we see no clear cutoff index. In other words, we are not so fortunate that the prior is implicitly restricted to some subspace that is of lower rank than our ensemble. This is a very realistic situation, and indicates that localisation (implemented further below) will be very beneficial.

# ## Assimilation

# ### Propagation
# Ensemble methods obtain observation-parameter sensitivities from the covariances of the ensemble run through the ("forward") model. Note that this is "embarrasingly parallelizable", because each iterate is complete independent (requires no communication) from the others. We take advantage of this through multiprocessing.

# Set (int) number of CPU cores to use. Set to False when debugging.
multiprocess = False

def forward_model(nTime, *args, desc=""):
    """Create the (composite) forward model, i.e. forecast. Supports ensemble input.

    This is a composite function.  The main work consists of running the
    reservoir simulator for each realisation in the ensemble.  However, the
    simulator only inputs/outputs state variables, so we also have to take
    the necessary steps to set the parameter values
    """

    def run1(estimable):
        """Forward model for a *single* member/realisation."""
        # Avoid the risk (difficult to diagnose with multiprocessing) that
        # the parameter values of one member overwrite those of another.
        # Alternative: re-initialize the model.
        model_n = copy.deepcopy(model)

        # Unpack variables
        wsat0, perm, *rates = estimable

        # Set production rates, if provided.
        if rates:
            # The historical rates (couble be, but) are not unknowns;
            # Instead, this "setter" is provided for the purpose
            # of optimising future production.
            model_n.producers[:, 2] = rates[0]
            model_n.config_wells(model_n.injectors, model_n.producers, remap=False)

        # Set permeabilities
        set_perm(model_n, perm)

        # Run simulator
        wsats, prods = misc.repeat(
            model_n.step, nTime, wsat0, dt, obs_model, pbar=False)

        return wsats, prods

    # Compose ensemble. This packing is a technicality necessary for
    # the syntax of `map`, used instead of a `for`-loop for multiprocessing.
    E = zip(*args)  # Tranpose args (so that member_index is 0th axis)

    # Dispatch jobs
    desc = " ".join(["Ens.simul.", desc])
    if multiprocess:
        from p_tqdm import p_map
        n = None if isinstance(multiprocess, bool) else multiprocess
        Ef = list(p_map(run1, list(E), num_cpus=n, desc=desc))
    else:
        Ef = list(progbar(map(run1, E), desc, N))

    # Transpose (to unpack)
    # Here we output everything, but really we need only emit
    # - The state at the final time, for restarts (predictions).
    # - The observations (for the assimilation update).
    # - The variables used for production optimisation
    #   (in this case the same as the obs, namely the production).
    saturation, production = zip(*Ef)

    return np.array(saturation), np.array(production)

# Note that the forward model not only takes an ensemble of permeability fields, but also an ensemble of initial water saturations. This is not because the initial saturations are uncertain (unknown); indeed, this here case study assumes that it is perfectly known, and equal to the true initial water saturation (a constant field of 0). Therefore, the initial water saturation is set to the true value for each member (giving it uncertainty 0).

wsat.init.Prior = np.tile(wsat.init.Truth, (N, 1))

# So why does `forward_model` have saturation as an input and output? Because the posterior of this state (i.e. time-dependent, prognostic) variable *does* depend on the method used for the conditioning, and will later be used to restart the simulations so as to generate future predictions.

# Now, let's run the forward model on the prior.

(wsat.past.Prior,
 prod.past.Prior) = forward_model(nTime, wsat.init.Prior, perm.Prior)

# ### Localisation
# Localisation invervenes to fix-up the estimated correlations before they are used. It is a method of injecting prior information (distant points are likely not strongly codependent) that is not *encoded* in the ensemble (usually due to their finite size). Defining an effective localisation mask or tapering function can be a difficult task.

# #### Correlation plots
# The conditioning "update" of ensemble methods is often formulated in terms of a "**Kalman gain**" matrix, derived so as to achieve a variety of optimality properties (see e.g. [[Jaz70]](#Jaz70)): in the linear-Gaussian case, to compute the correct posterior moments; in the linear (non-Gaussian) case, to compute the BLUE, or achieve orthogonality of the posterior error and innovation; in the non-linear, non-Gaussian case, the ensemble version can be derived as linear regression (with some tweaks) from the perturbed observations to the unknowns.
#
# Another way to look at it is to ask "what does it do?". Heuristically, this may be answered in 3 points:
#
# - It uses *estimated correlation* coefficients to establish relationships between observations and unknowns. For example, if there is no correlation, there will be no update (even for iterative methods).
# - It takes into account the variables' scales *and* relative uncertainties, via their variances. Hence why it works with covariances, and not just correlations. One of the main advantages of ensemble methods is that the estimation inherently provides reduced-rank representations of covariance matrices.
# - It takes into account the "intermingling" of correlations. For example, two measurements/observations that are highly correlated (when including both prior and observation errors) will barely contribute more than either one.
#
# In summary, it is useful to investigate the correlation relations of the ensemble, especially for the prior.

# ##### Auto-correlation for `wsat`
# First, as a sanity check, it is useful to plot the correlation of the saturation field at some given time vs. the production at the same time. The correlation should be maximal (1.00) at the location of the well in question. Let us verify this: zoom-in several times, centering on the green star, to verify that it lies on top of the well of that panel.

iT = -1
xx = wsat.past.Prior[:, iT]
yy = prod.past.Prior[:, iT].T
corrs = [misc.corr(xx, y) for y in yy]
fig, axs, _ = plots.fields(plots.corr_field, corrs, f"Saturation vs. obs (time {iT})");
# Add wells and maxima
for i, (ax, well) in enumerate(zip(axs, model.producers)):
    plots.well_scatter(ax, well[None, :], inj=False, text=str(i))
    ax.plot(*model.ind2xy(corrs[i].argmax()), "g*", ms=12, label="max")


# ##### Correlation vs unknowns (pre-permeability)
# The following plots the correlation fields for the unknown field (pre-permeability) vs the productions at a given time.

fig, axs, _ = plots.fields(plots.corr_field, corrs, "Pre-perm vs. obs.");  # Init fig
#
@interact(time_index=(0, nTime-1))
def _plot(time_index=nTime//2):
    xx = perm.Prior
    yy = prod.past.Prior[:, time_index].T
    with np.errstate(divide="ignore", invalid="ignore"):
        corrs = [misc.corr(xx, y) for y in yy]
        for i, (ax, corr, well) in enumerate(zip(axs, corrs, model.producers)):
            ax.clear()
            plots.corr_field(ax, corr)
            plots.well_scatter(ax, well[None, :], inj=False, text=str(i))
            ax.plot(*model.ind2xy(corr.argmax()), "g*", ms=12, label="max")

# Use the interative slider below the plot to walk through time. Note that
#
# - The variances in the initial productions (when the slider is all the way to the left) are zero, yielding nan's and blank plots.
# - The maximum is not quite superimposed with the well in question.
# - The correlation fields grow stronger in time. This is because it takes time for the permeability field to impact the flow at the well. TODO: improve reasoning.
# - The opposite corner of a given well is anti-correlated with it. This makes sense, since larger permeability in the opposite corner to a well will subtract from its production.

# ### Ensemble smoother

# #### Why smoothing?
# Why do we only use smoothers (and not filters) for history matching?
# When ensemble methods were first being used for history matching, it was
# though that filtering, rather than smoothing, should be used.
# Filters sequentially assimilate the time-series data,
# running the model simulator in between each observation time,
# (re)starting each step from saturation fields that
# have been conditioned on all of the data up until that point.
# Typically, the filters would be augmented with parameter fields (time-independent unknowns) as well. Either way, re-starting the simulator with ensemble-updated fields tends to be problematic, because the updated members might not be physically realistic and realisable, causing the simulator's solver to slow down or fail to converge. This issue is generally aggravated by not having run the simulator from time 0, since the linear updates provided by the ensemble will yield saturation fields that differ from those obtained by re-running the simulator. Therefore, updating the unknowns only once, using all of the observations, is far more convenient.

class ES_update:
    """Update/conditioning (Bayes' rule) of an ensemble, given a vector of obs.

    Implements the "ensemble smoother" (ES) algorithm,
    with "perturbed observations".
    NB: obs_err_cov is treated as diagonal. Alternative: use `sla.sqrtm`.

    Why have we chosen to use a class (and not a function)?
    Because this allows storing `KGdY`, for later use.
    This "on-the-fly" application follows directly from state-augmentation formalism.

    NB: some of these formulae appear transposed, and reversed,
    compared to (EnKF) literature standards. The reason is that
    we stack the members as rows instead of the conventional columns.
    Rationale: https://nansencenter.github.io/DAPPER/dapper/index.html#conventions
    """

    def __init__(self, obs_ens, observations, obs_err_cov):
        """Prepare the update."""
        Y, _        = center(obs_ens, rescale=True)
        obs_cov     = obs_err_cov*(N-1) + Y.T@Y
        obs_pert    = rnd.randn(N, len(observations)) @ sqrt(obs_err_cov)
        innovations = observations - (obs_ens + obs_pert)

        # (pre-) Kalman gain * Innovations
        # Also called the X5 matrix by Evensen'2003.
        self.KGdY = innovations @ sla.pinv2(obs_cov) @ Y.T

    def __call__(self, E):
        """Do the update."""
        return E + self.KGdY @ center(E)[0]

# #### Compute

# +
def ravel_time(x, undo=False):
    """Ravel/flatten the last two axes, or undo this operation."""
    if undo:
        *N, ab = x.shape
        return x.reshape(N + [nTime, ab//nTime])
    else:
        *N, a, b = x.shape
        return x.reshape(N + [a*b])

# Pre-compute
ES = ES_update(
    obs_ens      = ravel_time(prod.past.Prior),
    observations = ravel_time(prod.past.Noisy),
    obs_err_cov  = sla.block_diag(*[R]*nTime),
)
# -

# Apply
perm.ES = ES(perm.Prior)

# #### Plot ES
# Let's plot the updated, initial ensemble.

plots.fields(plots.field, perm.ES, "ES (posterior)");

# We will see some more diagnostics later.

# ### Iterative ensemble smoother

# #### Why iterate?
# Because of the non-linearity of the forward model.

def IES_analysis(w, T, Y, dy):
    """Compute the ensemble analysis."""
    N = len(Y)
    Y0       = sla.pinv(T) @ Y        # "De-condition"
    V, s, UT = misc.svd0(Y0)          # Decompose
    Cowp     = misc.pows(V, misc.pad0(s**2, N) + N-1)
    Cow1     = Cowp(-1.0)             # Posterior cov of w
    grad     = Y0@dy - w*(N-1)        # Cost function gradient
    dw       = grad@Cow1              # Gauss-Newton step
    T        = Cowp(-.5) * sqrt(N-1)  # Transform matrix
    return dw, T


def IES(ensemble, observations, obs_err_cov, stepsize=1, nIter=10, wtol=1e-4):
    """Iterative ensemble smoother."""
    E = ensemble
    y = observations
    N = len(E)
    N1 = N - 1
    Rm12T = np.diag(sqrt(1/np.diag(obs_err_cov)))  # TODO?

    # Init
    stat = Dict(dw=[], rmse=[], stepsize=[],
                obj=Dict(lklhd=[], prior=[], postr=[]))

    # Init ensemble decomposition.
    X0, x0 = center(E)    # Decompose ensemble.
    w      = np.zeros(N)  # Control vector for the mean state.
    T      = np.eye(N)    # Anomalies transform matrix.

    for itr in range(nIter):
        # Compute rmse (vs. Truth)
        stat.rmse += [misc.RMS(perm.Truth, E).rmse]

        # Forecast.
        _, Eo = forward_model(nTime, wsat.init.Prior, E, desc=f"Iter #{itr}")
        Eo = ravel_time(Eo)

        # Prepare analysis.
        Y, xo  = center(Eo)         # Get anomalies, mean.
        dy     = (y - xo) @ Rm12T   # Transform obs space.
        Y      = Y        @ Rm12T   # Transform obs space.

        # Diagnostics
        stat.obj.prior += [w@w * N1]
        stat.obj.lklhd += [dy@dy]
        stat.obj.postr += [stat.obj.prior[-1] + stat.obj.lklhd[-1]]

        reject_step = itr > 0 and stat.obj.postr[itr] > np.min(stat.obj.postr)
        if reject_step:
            # Restore prev. ensemble, lower stepsize
            stepsize   /= 10
            w, T        = old  # noqa
        else:
            # Store current ensemble, boost stepsize
            old         = w, T
            stepsize   *= 2
            stepsize    = min(1, stepsize)

            dw, T = IES_analysis(w, T, Y, dy)

        stat.dw += [dw@dw / N]
        stat.stepsize += [stepsize]

        # Step
        w = w + stepsize*dw
        E = x0 + (w + T)@X0

        if stepsize * np.sqrt(dw@dw/N) < wtol:
            break

    # The last step must be discarded,
    # because it cannot be validated without re-running the model.
    w, T = old
    E = x0 + (w+T)@X0

    return E, stat


# #### Apply the IES

perm.IES, stats_IES = IES(
    ensemble     = perm.Prior,
    observations = ravel_time(prod.past.Noisy),
    obs_err_cov  = sla.block_diag(*[R]*nTime),
    stepsize=1,
)


# #### Plot IES
# Let's plot the updated, initial ensemble.

plots.fields(plots.field, perm.IES, "IES (posterior)");

# The following plots the cost function(s) together with the error compared to the true (pre-)perm field as a function of the iteration number. Note that the relationship between the (total, i.e. posterior) cost function  and the RMSE is not necessarily monotonic. Re-running the experiments with a different seed is instructive. It may be observed that the iterations are not always very successful.

fig, ax = freshfig("IES Objective function")
ls = dict(postr="-", prior=":", lklhd="--")
for name, J in stats_IES.obj.items():
    ax.plot(np.sqrt(J), color="b", ls=ls[name], label=name)
ax.set_xlabel("iteration")
ax.set_ylabel("RMS mismatch", color="b")
ax.tick_params(axis='y', labelcolor="b")
ax.legend()
ax2 = ax.twinx()  # axis for rmse
ax2.set_ylabel('RMS error', color="r")
ax2.plot(stats_IES.rmse, color="r")
ax2.tick_params(axis='y', labelcolor="r")

# ### Diagnostics
# In terms of root-mean-square error (RMSE), the ES is expected to improve on the prior. The "expectation" wording indicates that this is true on average, but not always. To be specific, it means that it is guaranteed to hold true if the RMSE is calculated for infinitely many experiments (each time simulating a new synthetic truth and observations from the prior). The reason for this is that the ES uses the Kalman update, which is the BLUE (best linear unbiased estimate), and "best" means that the variance must get reduced. However, note that this requires the ensemble to be infinitely big, which it most certainly is not in our case. Therefore, we do not need to be very unlucky to observe that the RMSE has actually increased. Despite this, as we will see later, the data match might yield a different conclusions concerning the utility of the update.

# #### RMS summary

print("Stats vs. true field")
misc.RMS_all(perm, vs="Truth")

# #### Plot of means
# Let's plot mean fields.
#
# NB: Caution! Mean fields are liable to smoother than the truth. This is a phenomenon familiar from geostatistics (e.g. Kriging). As such, their importance must not be overstated (they're just one estimator out of many). Instead, whenever a decision is to be made, all of the members should be included in the decision-making process. This does not mean that you must eyeball each field, but that decision analyses should be based on expected values with respect to ensembles.

perm_means = Dict({k: perm[k].mean(axis=0) for k in perm})

plots.fields(plots.field, perm_means, "Means");

# ### Past production (data mismatch)
# In synthetic experiments such as this one, is is instructive to computing the "error": the difference/mismatch of the (supposedly) unknown parameters and the truth.  Of course, in real life, the truth is not known.  Moreover, at the end of the day, we mainly care about production rates and saturations.  Therefore, let us now compute the "residual" (i.e. the mismatch between predicted and true *observations*), which we get from the predicted production "profiles".

(wsat.past.ES,
 prod.past.ES) = forward_model(nTime, wsat.init.Prior, perm.ES)

(wsat.past.IES,
 prod.past.IES) = forward_model(nTime, wsat.init.Prior, perm.IES)

# It is Bayesian(ally) consistent to apply the pre-computed ES gain to any un-conditioned ensemble, e.g. that of the prior's production predictions. This can be seen (by those familiar with that trick) by state augmentation. This provides another posterior approximation of the production history -- one which doesn't require running the model again (in contrast to what we did for `prod.past.(I)ES` immediately above). Since it requires 0 iterations, let's call this "ES0". Let us try that as well.

prod.past.ES0 = ravel_time(ES(ravel_time(prod.past.Prior)), undo=True)

# #### Plot them all together:

v = plots.productions(prod.past, "Past")
display(v)

# #### RMS summary

print("Stats vs. past production (i.e. NOISY observations)")
misc.RMS_all(prod.past, vs="Noisy")

# Note that the data mismatch is significantly reduced. This may be the case even if the updated permeability field did not have a reduced rmse (overall, relative to that of the prior prior). The "direct" forecast (essentially just linear regression) may achieve even lower rmse, but generally, less realistic production plots.


# ##### Comment on prior
# Note that the prior "surrounds" the data. This the likely situation in our synthetic case, where the truth was generated by the same random draw process as the ensemble.
#
# In practice, this is often not the case. If so, you might want to go back to your geologists and tell them that something is amiss. You should then produce a revised prior with better properties.
#
# Note: the above instructions sound like statistical heresy. We are using the data twice over (on the prior, and later to update/condition the prior). However, this is justified to the extent that prior information is difficult to quantify and encode. Too much prior adaptation, however, and you risk overfitting! Indeed, it is a delicate matter.

# ##### Comment on posterior
# If the assumptions (statistical indistinguishability, Gaussianity) are not too far off, then the ensemble posteriors (ES, EnKS, ES0) should also surround the data, but with a tighter fit.

# ## Prediction
# We now prediction the future by forecasting from the current (present-time) ensembles.
#
# Note that we must use the current saturation in the "restart" for the predictive simulations. Since the estimates of the current saturation depend on the assumed permeability field, these estimates are also "posterior", and depend on the conditioning method used. For convenience, we first extract the slice of the current saturation fields (which is really the only one we make use of among those of the past), and plot the mean fields.

wsat.curnt = Dict({k: v[..., -1, :] for k, v in wsat.past.items()})
wsat_means = Dict({k: np.atleast_2d(v).mean(axis=0) for k, v in wsat.curnt.items()})
plots.fields(plots.oilfield, wsat_means, "Means");

# Now we predict.

print("Future/prediction")

(wsat.futr.Truth,
 prod.futr.Truth) = misc.repeat(model.step, nTime, wsat.curnt.Truth, dt, obs_model)

(wsat.futr.Prior,
 prod.futr.Prior) = forward_model(nTime, wsat.curnt.Prior, perm.Prior)

(wsat.futr.ES,
 prod.futr.ES) = forward_model(nTime, wsat.curnt.ES, perm.ES)

(wsat.futr.IES,
 prod.futr.IES) = forward_model(nTime, wsat.curnt.IES, perm.IES)

prod.futr.ES0 = ravel_time(ES(ravel_time(prod.futr.Prior)), undo=True)

# ### Diagnostics

# #### Plot future production

v = plots.productions(prod.futr, "Future");
display(v)

# #### RMS summary

print("Stats vs. (supposedly unknown) future production")
misc.RMS_all(prod.futr, vs="Truth")


# ## Robust optimisation
# NB: This section is very unfinished, and should not be seen as a reference.

# This section uses EnOpt to optimise the controls: the relative rates of production of the wells (again, for simplicity, these will be constant in time).

# Ojective function definition: total oil from production wells. This objective function takes an ensemble (`*E`) of unknowns (`wsat, perm`) and controls (`rates`) and outputs the corresponding ensemble of total oil productions.

def total_oil(E, rates):
    # bounded = np.all((0 < rates) & (rates < 1), axis=1)
    wsat, prod = forward_model(nTime, *E, rates)
    return np.sum(prod, axis=(1, 2))

# Define step modifier to improve on "vanilla" gradient descent.

def GDM(beta1=0.9):
    """Gradient descent with (historical) momentum."""
    grad1 = 0

    def set_historical(g):
        nonlocal grad1
        grad1 = beta1*grad1 + (1-beta1)*g

    def step(g):
        set_historical(g)
        return grad1

    return step

# Define EnOpt

def EnOpt(obj, E, ctrls, C12, stepsize=1, nIter=10):
    N = len(E[0])
    stepper = GDM()

    # Diagnostics
    print("Initial controls:", ctrls)
    repeated = np.tile(ctrls, (N, 1))
    J = obj(E, repeated).mean()
    print("Total oil (mean) for initial guess: %.3f" % J)

    for _itr in progbar(range(nIter), desc="EnOpt"):
        Eu = ctrls + rnd.randn(N, len(ctrls)) @ C12.T
        Eu = Eu.clip(1e-5)

        Ej = obj(E, Eu)
        # print("Total oil (mean): %.3f"%Ej.mean())

        Xu = center(Eu)[0]
        Xj = center(Ej)[0]

        G  = Xj.T @ Xu / (N-1)

        du = stepper(G)
        ctrls  = ctrls + stepsize*du
        ctrls  = ctrls.clip(1e-5)

    # Diagnostics
    print("Final controls:", ctrls)
    repeated = np.tile(ctrls, (N, 1))
    J = obj(E, repeated).mean()
    print("Total oil (mean) after optimisation: %.3f" % J)

    return ctrls

# Run EnOpt

rnd.seed(3)
# ctrls0  = model.producers[:, 2]
ctrls0  = rnd.rand(nProd)
ctrls0 /= sum(ctrls0)
C12     = 0.03 * np.eye(nProd)
E       = wsat.curnt.ES, perm.ES
# E       = wsat.curnt.IES, perm.IES
ctrls   = EnOpt(total_oil, E, ctrls0, C12, stepsize=10)


# ### Final comments
# It is instructive to run this notebook/script again, but with a different random seed. This will yield a different truth, and noisy production data, and so a new case/problem, which may be more, or less, difficult.
#
# Another alternative is to only re-run the notebook cells starting from where the prior was sampled. Thus, the truth and observations will not change, yet because the prior sample will change, the results will change. If this change is significant (which can only be asserted by re-running the experiments several times), then you can not have much confidence in your result. In order to fix this, you must increase the ensemble size (to reduce sampling error), or play with the tuning parameters such as the localisation radius (or more generally, improve your localisation implementation).
#
# Such re-running of the synthetic experiments is similar in aim to statistical cross-validation. Note that it may (and should) also be applied in real applications! Of couse, then the truth is unknown. But even though the truth is unknown, a synthetic truth can be sampled from the prior uncertainty. And at the very least, the history matching and optimisation methods should yield improved performance (reduced errors and increased NPV) in the synthetic case.

# ## References
# <a id="Jaz70">[Jaz70]</a>: Jazwinski, A. H. 1970. *Stochastic Processes and Filtering Theory*. Vol. 63. Academic Press.
