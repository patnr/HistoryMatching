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
# | Navigate                      |    | Edit              |    | Exit           |    | Run                              |    | Run & go to next                  |
# | -------------                 | -- | ----------------- | -- | --------       | -- | -------                          | -- | -----------------                 |
# | <kbd>↓</kbd> and <kbd>↑</kbd> |    | <kbd>Enter</kbd>  |    | <kbd>Esc</kbd> |    | <kbd>Ctrl</kbd>+<kbd>Enter</kbd> |    | <kbd>Shift</kbd>+<kbd>Enter</kbd> |
#
# When you open a notebook it starts a **session (kernel/runtime)** of Python
# in the background.  All of the code cells (in a given notebook) are connected
# (they use the same kernel and thus share variables, functions, and classes).
# Thus, the **order** in which you run the cells matters.  One thing you must
# know is how to **restart** the session, so that you can start over. Try to
# locate this option via the top menu bar.

# There is a huge amount of libraries available in **Python**,
# including the popular `scipy` (with `numpy` at its core) and `matplotlib` packages.
# These are imported (and abbreviated) as `sp`, `np`, and `mpl` and `plt`.
# Try them out by running the following, which illustrates some algebra
# using syntax reminiscent of Matlab.

import numpy as np
from matplotlib import pyplot as plt
import mpl_setup
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

# ## Load tools

# Run the following cells to import some tools.

from copy import deepcopy

import scipy.linalg as sla
from matplotlib import ticker
from mpl_tools.place import freshfig
from numpy.random import randn
from numpy import sqrt
from struct_tools import DotDict as Dict
from tqdm.auto import tqdm as progbar
from IPython.display import display
from ipywidgets import interact

# The next cell imports the model and associate tools.

import geostat
import simulator
import simulator.plotting as plots
import tools
from tools import center, mean0

# The following declares some data containers to help us keep organised.

# +
# Permeability
perm = Dict()

# Production
prod = Dict(
    past   = Dict(),
    future = Dict(),
)

# Water saturation
wsat = Dict(
    initial = Dict(),
    past    = Dict(),
    future  = Dict(),
)
# -

# Technical note: This data hierarchy is convienient in *this* notebook/script, especially for plotting purposes. For example, we can with ease refer to `wsat.past.Truth` and `wsat.past.Prior`. The former will be a numpy array of shape `(nTime, M)` where `M = model.M`, and the latter will have shape `(N, nTime, M)` where `N` is the size of the ensemble. However, in other implementations, different choices of data structure may be more convenient, e.g. where the different components of the unknowns are merely concatenated along the last axis, rather than being kept in separate dicts.

# Finally, for reproducibility, we set the random generator seed.

seed = np.random.seed(4)  # very easy
# seed = np.random.seed(5)  # hard
# seed = np.random.seed(6)  # very easy
# seed = np.random.seed(7)  # easy

# ## Model and case specification
# The reservoir model, which takes up about 100 lines of python code, is outrageously simple, serving the purpose of *illustrating* the history matching process. The model is a 2D, two-phase, immiscible, incompressible simulator using TPFA discretisation. It was translated from the Matlab code here http://folk.ntnu.no/andreas/papers/ResSimMatlab.pdf

model = simulator.ResSim(Nx=20, Ny=20, Lx=2, Ly=1)

# The model comes with a set of plot functionality for convenience.
# Here are some relevant configuration defaults.

plots.model = model
plots.field.coord_type = "absolute"
plots.field.levels = np.linspace(-3.5, 3.5, 21)
plots.field.cmap = "jet"

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
# In this model, wells are represented simply by point **sources** and **sinks**. This is of course incredibly basic and not realistic, but works for our purposes. So all we need to specify is their placement and flux (which we will not vary in time). The code below puts wells on a grid. Try `print(wlGrid)` to see how to easily specify another well configuration.
#
# Since the **boundary conditions** are Dirichlet, specifying *zero flux*, and the fluid is incompressible, the total of the source terms must equal that of the sinks. This is ensured by the `config_wells` function used below.

grid1d = [.1, .9]
wlGrid = np.dstack(np.meshgrid(grid1d, grid1d)).reshape((-1, 2))
rates = np.ones((len(wlGrid), 1))  # ==> all wells use the same (constant) rate
model.config_wells(
    # Each row in `inj` and `prod` should be a tuple: (x, y, rate),
    # where x, y ∈ (0, 1) and rate > 0.
    inj  = [[0.50, 0.50, 1.00]],
    prod = np.hstack((wlGrid, rates)),
);

# #### Plot true field
# Let's take a moment to visualize the model permeability field, and the well locations.

fig, ax = freshfig("True perm. field", figsize=(1.2, .7), rel=1)
# cs = plots.field(ax, perm.Truth)
cs = plots.field(ax, perm_transf(perm.Truth),
                 locator=ticker.LogLocator(), cmap="viridis",
                 levels=10)
plots.well_scatter(ax, model.producers, inj=False)
plots.well_scatter(ax, model.injectors, inj=True)
fig.colorbar(cs)
plt.pause(.1)


# #### Define obs operator
# The data will consist in the water saturation of at the well locations, i.e. of the production. I.e. there is no well model. It should be pointed out, however, that ensemble methods technically support observation models of any complexity, though your accuracy mileage may vary (again, depending on the incurred nonlinearity and non-Gaussianity). Furthermore, it is also no problem to include time-dependence in the observation model.

obs_inds = [model.xy2ind(x, y) for (x, y, _) in model.producers]
def obs_model(water_sat):
    return water_sat[obs_inds]

# #### Simulation to generate the synthetic truth evolution and data

wsat.initial.Truth = np.zeros(model.M)
T     = 1
dt    = 0.025
nTime = round(T/dt)
(wsat.past.Truth,
 prod.past.Truth) = tools.repeat(model.step, nTime, wsat.initial.Truth, dt, obs_model)

# ##### Animation
# Run the code cells below to get an animation of the oil saturation evolution. Injection (resp. production) wells are marked with triangles pointing down (resp. up).

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
    prod.past.Noisy[iT] += sqrt(R) @ randn(nProd)


# Plot of observations (and their noise):

fig, ax = freshfig("Observations")
hh_y = plots.production1(ax, prod.past.Truth, obs=prod.past.Noisy)
plt.pause(.1)

# ## Prior
# The prior ensemble is generated in the same manner as the (synthetic) truth, using the same mean and covariance.  Thus, the members are "statistically indistinguishable" to the truth. This assumption underlies ensemble methods.

N = 200
perm.Prior = sample_prior_perm(N)

# Note that field (before transformation) is Gaussian with (expected) mean 0 and variance 1.
print("Prior mean:", np.mean(perm.Prior))
print("Prior var.:", np.var(perm.Prior))

# Let us inspect the parameter values in the form of their histogram.

fig, ax = freshfig("Perm. -- marginal distribution", figsize=(1.3, .5), rel=1)
for label, data in perm.items():

    ax.hist(
        perm_transf(data.ravel()),
        perm_transf(np.linspace(-3, 3, 32)),
        # "Downscale" ens counts by N. Necessary because `density` kw
        # doesn't work "correctly" with log-scale.
        weights = (np.ones(model.M*N)/N if label != "Truth" else None),
        label=label, alpha=0.3)

    ax.set(xscale="log", xlabel="Permeability", ylabel="Count")
    ax.legend();
plt.pause(.1)

# The above histogram should be Gaussian histogram if perm_transf is purely exponential:

# Below we can see some realizations (members) from the ensemble.

plots.fields(plots.field, perm.Prior, "Prior");

# #### Eigenvalue spectrum
# In practice, of course, we would not be using an explicit `Cov` matrix when generating the prior ensemble, because it would be too large.  However, since this synthetic case in being made that way, let's inspect its spectrum.

U, svals, VT = sla.svd(perm.Prior)
ii = 1 + np.arange(len(svals))
fig, ax = freshfig("Spectrum of true cov.", figsize=(1.3, .5), rel=1)
ax.loglog(ii, svals)
# ax.semilogx(ii, svals)
ax.grid(True, "minor", axis="x")
ax.grid(True, "major", axis="y")
ax.set(xlabel="eigenvalue #", ylabel="variance");
plt.pause(.1)

# With our limited ensemble size, we see no clear cutoff index. In other words, we are not so fortunate that the prior is implicitly restricted to some subspace that is of lower rank than our ensemble. This is a very realistic situation, and indicates that localisation (implemented further below) will be very beneficial.

# ## Assimilation

# ### Exc (optional)
# Before going into iterative methods, we note that
# the ensemble smoother (ES) is favoured over the ensemble Kalman smoother (EnKS) for history matching. This may come as a surprise, because the EnKS processes the observations sequentially, like the ensemble Kalman filter (EnKF), not in the batch manner of the ES. Because sequential processing is more gradual, one would expect it to achieve better accuracy than batch approaches. However, the ES is preferred because the uncertainty in the state fields is often of less importance than the uncertainty in the parameter fields. More imperatively, (re)starting the simulators (e.g. ECLIPSE) from updated state fields (as well as parameter fields) is a troublesome affair; the fields may have become "unphysical" (or "not realisable") because of the ensemble update, which may hinder the simulator from producing meaningful output (it may crash, or have trouble converging). On the other hand, by going back to the parameter fields before geological modelling (using fast model update (FMU)) tends to yield more realistic parameter fields. Finally, since restarts tend to yield convergence issues in the simulator the following inequality is usually large.
#
# $$
# \begin{align}
# 	\max_n \sum_{t}^{} T_t^n < \sum_{t}^{} \max_n T_t^n,
# 	\label{eqn:max_sum_sum_max}
# \end{align}
# $$
# skjervheim2011ensemble
# Here, $T_t^n$

# ### Propagation
# Ensemble methods obtain observation-parameter sensitivities from the covariances of the ensemble run through the model. Note that this for-loop is "embarrasingly parallelizable", because each iterate is complete indepdendent (requires no communication) from the others.

multiprocess = False  # multiprocessing?

def forward_model(nTime, *args, desc="Ens. run"):
    """Create the (composite) forward model, i.e. forecast. Supports ensemble input.

    This is a composite function.  The main work consists of running the
    reservoir simulator for each realisation in the ensemble.  However, the
    simulator only inputs/outputs state variables, so we also have to take
    the necessary steps to set the parameter values
    """

    def forecast1(estimable):
        """Forward model for a single member/realisation."""
        # Avoid the risk (difficult to diagnose with multiprocessing) that
        # the parameter values of one member overwrite those of another.
        # Alternative: re-initialize the model.
        model_n = deepcopy(model)

        # Unpack/unbundle variables
        wsat0, perm, *q_prod = estimable

        # If provided: set production rates
        if q_prod:
            model_n.producers[:, 2] = q_prod[0]
            model_n.config_wells(model_n.injectors, model_n.producers, remap=False)

        # Set permeabilities
        set_perm(model_n, perm)

        # Run simulator
        wsats, prods = tools.repeat(
            model_n.step, nTime, wsat0, dt, obs_model, pbar=False)

        return wsats, prods

    # Compose ensemble. This packing/bundling is a technicality
    # necessary for the syntax of `map` (necessary for multiprocessing).
    E = zip(*args)  # Tranpose args (so that member_index is 0th axis)

    # Dispatch jobs
    if multiprocess:
        import multiprocessing_on_dill as mpd
        with mpd.Pool() as pool:
            Ef = list(progbar(pool.imap(forecast1, E), total=N, desc=desc))
    else:
        Ef = list(progbar(map(forecast1, E), total=N, desc="Members"))

    # Transpose (to unpack)
    # Here we output everything, but really we need only emit
    # - The state at the final time, for restarts (predictions).
    # - The observations (for the assimilation update).
    # - The variables used for production optimisation
    #   (in this case the same as the obs, namely the production).
    saturation, production = zip(*Ef)

    return np.array(saturation), np.array(production)

# Note that the forward model not only takes an ensemble of permeability fields, but also an ensemble of initial water saturations. This is not because the initial saturations are uncertain (unknown); indeed, this case study assumes that it is perfectly known (i.e. equal to the true initial water saturation, which is a constant field of 0). Therefore, the initial water saturation is set to the true value for each member (giving it uncertainty 0).

wsat.initial.Prior = np.tile(wsat.initial.Truth, (N, 1))

# So why does the `forward_model` take saturation as an input? Because the posterior of this state (i.e. time-dependent) variable does depend on the method used for the conditioning, and will later be used to restart the simulations so as to generate future predictions.

# Now we run the forward model.

(wsat.past.Prior,
 prod.past.Prior) = forward_model(nTime, wsat.initial.Prior, perm.Prior)

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
corrs = [tools.corr(xx, y) for y in yy]
fig, axs, _ = plots.fields(plots.corr_field, corrs, f"Saturation vs. obs (time {iT})");
# Add wells and maxima
for i, (ax, well) in enumerate(zip(axs, model.producers)):
    plots.well_scatter(ax, well[None, :], inj=False, text=str(i))
    ax.plot(*model.ind2xy(corrs[i].argmax()), "g*", ms=12, label="max")


# ##### Correlation vs unknowns (pre-permeability)
# The following plots the correlation fields for the unknown field (pre-permeability) vs the productions at a given time. Use the interative slider below the plot to walk through time. Note that

# - The variances in the initial productions (when the slider is all the way to the left) are zero, yielding nan's and blank plots.
# - The maximum is not quite superimposed with the well in question.
# - The correlation fields grow stronger in time. This is because it takes time for the permeability field to impact the flow at the well. TODO: improve reasoning.
# - The opposite corner of a given well is anti-correlated with it. This makes sense, since larger permeability in the opposite corner to a well will subtract from its production.

fig, axs, _ = plots.fields(plots.corr_field, corrs, "Pre-perm vs. obs.");  # Init fig
#
@interact(time_index=(0, nTime-1))
def _plot(time_index=nTime//2):
    xx = perm.Prior
    yy = prod.past.Prior[:, time_index].T
    with np.errstate(divide="ignore", invalid="ignore"):
        corrs = [tools.corr(xx, y) for y in yy]
        for i, (ax, corr, well) in enumerate(zip(axs, corrs, model.producers)):
            ax.clear()
            plots.corr_field(ax, corr)
            plots.well_scatter(ax, well[None, :], inj=False, text=str(i))
            ax.plot(*model.ind2xy(corr.argmax()), "g*", ms=12, label="max")

# ### Ensemble smoother

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

    def __init__(self, obs_ens, observation, obs_err_cov):
        """Prepare the update."""
        Y, _        = center(obs_ens, rescale=True)
        obs_cov     = obs_err_cov*(N-1) + Y.T@Y
        obs_pert    = randn(N, len(observation)) @ sqrt(obs_err_cov)
        innovations = observation - (obs_ens + obs_pert)

        # (pre-) Kalman gain * Innovations
        # Also called the X5 matrix by Evensen'2003.
        self.KGdY = innovations @ sla.pinv2(obs_cov) @ Y.T

    def __call__(self, E):
        """Do the update."""
        return E + self.KGdY @ center(E)[0]

# #### Update
ES = ES_update(
    obs_ens     = prod.past.Prior.reshape((N, -1)),
    observation = prod.past.Noisy.reshape(-1),
    obs_err_cov = sla.block_diag(*[R]*nTime),
)

# Apply update
perm.ES = ES(perm.Prior)

# #### Plot ES
# Let's plot the updated, initial ensemble.

plots.fields(plots.field, perm.ES, "ES (posterior)");

# We will see some more diagnostics later.

# ### Iterative ensemble smoother
# The following is (almost) all that distinguishes all of the fully-Bayesian iterative ensemble smoothers in the literature.

def iES_flavours(w, T, Y, Y0, dy, Cowp, za, N, nIter, itr, MDA, flavour):
    N1 = N - 1
    Cow1 = Cowp(1.0)

    if MDA:  # View update as annealing (progressive assimilation).
        Cow1 = Cow1 @ T  # apply previous update
        dw = dy @ Y.T @ Cow1
        if 'PertObs' in flavour:   # == "ES-MDA". By Emerick/Reynolds
            D   = mean0(randn(*Y.shape)) * sqrt(nIter)
            T  -= (Y + D) @ Y.T @ Cow1
        elif 'Sqrt' in flavour:    # == "ETKF-ish". By Raanes
            T   = Cowp(0.5) * sqrt(za) @ T
        elif 'Order1' in flavour:  # == "DEnKF-ish". By Emerick
            T  -= 0.5 * Y @ Y.T @ Cow1
        Tinv = np.eye(N)  # [as initialized] coz MDA does not de-condition.

    else:  # View update as Gauss-Newton optimzt. of log-posterior.
        grad  = Y0@dy - w*za                  # Cost function gradient
        dw    = grad@Cow1                     # Gauss-Newton step
        # ETKF-ish". By Bocquet/Sakov.
        if 'Sqrt' in flavour:
            # Sqrt-transforms
            T     = Cowp(0.5) * sqrt(N1)
            Tinv  = Cowp(-.5) / sqrt(N1)
            # Tinv saves time [vs tinv(T)] when Nx<N
        # "EnRML". By Oliver/Chen/Raanes/Evensen/Stordal.
        elif 'PertObs' in flavour:
            if itr == 0:
                D = mean0(randn(*Y.shape))
                iES_flavours.D = D
            else:
                D = iES_flavours.D
            gradT = -(Y+D)@Y0.T + N1*(np.eye(N) - T)
            T     = T + gradT@Cow1
            # Tinv= tinv(T, threshold=N1)  # unstable
            Tinv  = sla.inv(T+1)           # the +1 is for stability.
        # "DEnKF-ish". By Raanes.
        elif 'Order1' in flavour:
            # Included for completeness; does not make much sense.
            gradT = -0.5*Y@Y0.T + N1*(np.eye(N) - T)
            T     = T + gradT@Cow1
            Tinv  = sla.pinv2(T)

    return dw, T, Tinv

# This outer function loops through the iterations, forecasting, de/re-composing the ensemble, performing the linear regression, validating step, and making statistics.

def IES(ensemble, observation, obs_err_cov,
        flavour="Sqrt", MDA=False, bundle=False,
        stepsize=1, nIter=10, wtol=1e-4):

    E = ensemble
    N = len(E)
    N1 = N - 1
    Rm12T = np.diag(sqrt(1/np.diag(obs_err_cov)))  # TODO?

    stats = Dict()
    stats.J_lklhd  = np.full(nIter, np.nan)
    stats.J_prior  = np.full(nIter, np.nan)
    stats.J_postr  = np.full(nIter, np.nan)
    stats.rmse     = np.full(nIter, np.nan)
    stats.stepsize = np.full(nIter, np.nan)
    stats.dw       = np.full(nIter, np.nan)

    if bundle:
        if isinstance(bundle, bool):
            EPS = 1e-4  # Sakov/Boc use T=EPS*eye(N), with EPS=1e-4, but I ...
        else:
            EPS = bundle
    else:
        EPS = 1.0  # ... prefer using  T=EPS*T, yielding a conditional cloud shape

    # Init ensemble decomposition.
    X0, x0 = center(E)    # Decompose ensemble.
    w      = np.zeros(N)  # Control vector for the mean state.
    T      = np.eye(N)    # Anomalies transform matrix.
    Tinv   = np.eye(N)
    # Explicit Tinv [instead of tinv(T)] allows for merging MDA code
    # with iEnKS/EnRML code, and flop savings in 'Sqrt' case.

    for itr in range(nIter):
        # Reconstruct smoothed ensemble.
        E = x0 + (w + EPS*T)@X0
        stats.rmse[itr] = tools.RMS(perm.Truth, E).rmse

        # Forecast.
        E_state, E_obs = forward_model(nTime, wsat.initial.Prior, E, desc=f"Iteration {itr}")
        E_obs = E_obs.reshape((N, -1))

        # Undo the bundle scaling of ensemble.
        if EPS != 1.0:
            E     = tools.inflate_ens(E, 1/EPS)
            E_obs = tools.inflate_ens(E_obs, 1/EPS)

        # Prepare analysis.Ç
        y      = observation        # Get current obs.
        Y, xo  = center(E_obs)      # Get obs {anomalies, mean}.
        dy     = (y - xo) @ Rm12T   # Transform obs space.
        Y      = Y        @ Rm12T   # Transform obs space.
        Y0     = Tinv @ Y           # "De-condition" the obs anomalies.

        # Set "cov normlzt fctr" za ("effective ensemble size")
        # => pre_infl^2 = (N-1)/za.
        za = N1
        if MDA:
            # inflation (factor: nIter) of the ObsErrCov.
            za *= nIter

        # Compute Cowp: the (approx) posterior cov. of w
        # (estiamted at this iteration), raised to some power.
        V, s, UT = tools.svd0(Y0)
        def Cowp(expo): return (V * (tools.pad0(s**2, N) + za)**-expo) @ V.T

        # TODO: NB: these stats are only valid for Sqrt
        stat2 = Dict(
            J_prior = w@w * N1,
            J_lklhd = dy@dy,
        )
        # J_posterior is sum of the other two
        stat2.J_postr = stat2.J_prior + stat2.J_lklhd
        # Take root, insert for [itr]:
        for name in stat2:
            stats[name][itr] = sqrt(stat2[name])

        # Accept previous increment? ...
        if (not MDA) and itr > 0 and stats.J_postr[itr] > np.nanmin(stats.J_postr):
            # ... No. Restore previous ensemble & lower the stepsize (don't compute new increment).
            stepsize   /= 10
            w, T, Tinv  = old  # noqa
        else:
            # ... Yes. Store this ensemble, boost the stepsize, and compute new increment.
            old         = w, T, Tinv
            stepsize   *= 2
            stepsize    = min(1, stepsize)
            dw, T, Tinv = iES_flavours(w, T, Y, Y0, dy, Cowp, za, N, nIter, itr, MDA, flavour)

        stats.      dw[itr] = dw@dw / N
        stats.stepsize[itr] = stepsize

        # Step
        w = w + stepsize*dw

        if stepsize * np.sqrt(dw@dw/N) < wtol:
            break

    stats.nIter = itr + 1

    if not MDA:
        # The last step (dw, T) must be discarded,
        # because it cannot be validated without re-running the model.
        w, T, Tinv  = old

    # Reconstruct the ensemble.
    E = x0 + (w+T)@X0

    return E, stats


# #### Apply the IES

perm.IES, stats_IES = IES(
    ensemble    = perm.Prior,
    observation = prod.past.Noisy.reshape(-1),
    obs_err_cov = sla.block_diag(*[R]*nTime),
    flavour="Sqrt", MDA=False, bundle=False, stepsize=1,
)


# #### Plot IES
# Let's plot the updated, initial ensemble.

plots.fields(plots.field, perm.IES, "IES (posterior)");

# The following plots the cost function(s) together with the error compared to the true (pre-)perm field as a function of the iteration number. Note that the relationship between the (total, i.e. posterior) cost function  and the RMSE is not necessarily monotonic. Re-running the experiments with a different seed is instructive. It may be observed that the iterations are not always very successful.

fig, ax = freshfig("IES Objective function")
ls = dict(J_prior=":", J_lklhd="--", J_postr="-")
for name, J in stats_IES.items():
    try:
        ax.plot(J, color="b", label=name.split("J_")[1], ls=ls[name])
    except IndexError:
        pass
ax.set_xlabel("iteration")
ax.set_ylabel("RMS mismatch", color="b")
ax.tick_params(axis='y', labelcolor="b")
ax.legend()
ax2 = ax.twinx()  # axis for rmse
ax2.set_ylabel('RMS error', color="r")
ax2.plot(stats_IES.rmse, color="r")
ax2.tick_params(axis='y', labelcolor="r")
plt.pause(.1)

# ### Diagnostics
# In terms of root-mean-square error (RMSE), the ES is expected to improve on the prior. The "expectation" wording indicates that this is true on average, but not always. To be specific, it means that it is guaranteed to hold true if the RMSE is calculated for infinitely many experiments (each time simulating a new synthetic truth and observations from the prior). The reason for this is that the ES uses the Kalman update, which is the BLUE (best linear unbiased estimate), and "best" means that the variance must get reduced. However, note that this requires the ensemble to be infinitely big, which it most certainly is not in our case. Therefore, we do not need to be very unlucky to observe that the RMSE has actually increased. Despite this, as we will see later, the data match might yield a different conclusions concerning the utility of the update.

# #### RMS summary

print("Stats vs. true field")
tools.RMS_all(perm, vs="Truth")

# #### Plot of means
# Let's plot mean fields.
#
# NB: Caution! Mean fields are liable to smoother than the truth. This is a phenomenon familiar from geostatistics (e.g. Kriging). As such, their importance must not be overstated (they're just one estimator out of many). Instead, whenever a decision is to be made, all of the members should be included in the decision-making process. This does not mean that you must eyeball each field, but that decision analyses should be based on expected values with respect to ensembles.

perm_means = Dict({k: perm[k].mean(axis=0) for k in perm})

plots.fields(plots.field, perm_means);

# ### Past production (data mismatch)
# In synthetic experiments such as this one, is is instructive to computing the "error": the difference/mismatch of the (supposedly) unknown parameters and the truth.  Of course, in real life, the truth is not known.  Moreover, at the end of the day, we mainly care about production rates and saturations.  Therefore, let us now compute the "residual" (i.e. the mismatch between predicted and true *observations*), which we get from the predicted production "profiles".

(wsat.past.ES,
 prod.past.ES) = forward_model(nTime, wsat.initial.Prior, perm.ES)

(wsat.past.IES,
 prod.past.IES) = forward_model(nTime, wsat.initial.Prior, perm.IES)

# We can also apply the ES update directly to the production data of the prior, which doesn't require running the model again (in contrast to what we had to do immediately above). Let us try that as well.

def with_flattening(fun):
    """Redefine `fun` so that it first flattens the input."""
    def fun2(xx):
        shape = xx.shape
        xx = xx.reshape((shape[0], -1))
        yy = fun(xx)
        return yy.reshape(shape)
    return fun2

prod.past.ES0 = with_flattening(ES)(prod.past.Prior)


# #### Plot them all together:

v = plots.productions(prod.past, "Past")
display(v)

# #### RMS summary

print("Stats vs. past production (i.e. NOISY observations)")
tools.RMS_all(prod.past, vs="Noisy")

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
plots.fields(plots.oilfield, wsat_means);

# Now we predict.

print("Future/prediction")

(wsat.future.Truth,
 prod.future.Truth) = tools.repeat(model.step, nTime, wsat.curnt.Truth, dt, obs_model)

(wsat.future.Prior,
 prod.future.Prior) = forward_model(nTime, wsat.curnt.Prior, perm.Prior)

(wsat.future.ES,
 prod.future.ES) = forward_model(nTime, wsat.curnt.ES, perm.ES)

(wsat.future.IES,
 prod.future.IES) = forward_model(nTime, wsat.curnt.IES, perm.IES)

prod.future.ES0 = with_flattening(ES)(prod.future.Prior)

# ### Diagnostics

# #### Plot future production

v = plots.productions(prod.future, "Future");
display(v)

# #### RMS summary

print("Stats vs. (supposedly unknown) future production")
tools.RMS_all(prod.future, vs="Truth")

# ## Robust optimisation

# This section uses EnOpt to optimise the controls: the relative rates of production of the wells (again, for simplicity, these will be constant in time).

# Cost function definition: total oil from production wells. This cost function takes for an ensemble of (wsat, perm) and controls (Q_prod) and outputs the corresponding ensemble of total oil productions.

def total_oil(E, Eu):
    wsat, perm = E
    wsat, prod = forward_model(nTime, wsat, perm, Eu)
    return np.sum(prod, axis=(1, 2))

# Define step modulator by adding momentum to vanilla gradient descent.

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

def EnOpt(wsats0, perms, u, C12, stepsize=1, nIter=10):
    N = len(wsats0)
    E = wsats0, perms

    stepper = GDM()

    print("Initial controls:", u)
    J = total_oil(E, np.tile(u, (N, 1))).mean()
    print("Total oil, averaged, initial: %.3f" % J)

    for _itr in progbar(range(nIter), desc="EnOpt"):
        Eu = u + randn(N, len(u)) @ C12.T
        Eu = Eu.clip(1e-5)

        Ej = total_oil(E, Eu)
        # print("Approx. total oil, average: %.3f"%Ej.mean())

        Xu = center(Eu)[0]
        Xj = center(Ej)[0]

        G  = Xj.T @ Xu / (N-1)

        du = stepper(G)
        u  = u + stepsize*du
        u  = u.clip(1e-5)

    print("Final controls:", u)
    J = total_oil(E, np.tile(u, (N, 1))).mean()
    print("Total oil, averaged, final: %.3f" % J)
    return u

# Run EnOpt

# u0  = model.producers[:, 2]
u0  = np.random.rand(nProd)
u0 /= sum(u0)
C12 = 0.03 * np.eye(nProd)
u   = EnOpt(wsat.past.ES[:, -1, :], perm.ES, u0, C12, stepsize=10)
# u   = EnOpt(wsat.past.IES[:, -1, :], perm.IES, u0, C12, stepsize=10)


# ### Final comments
# Note that result may depend on the sampling of the prior ensemble,
# as well as other random numbers. This effect should be minimized,
# and tested by repeat experiments. It should also be ensured that the errors decrease, and NPV increase. This is similar to cross validation. Note that such synthetic experiments are fully possible in the real world as well. Even though the truth is unknown, a synthetic truth can be sampled from the prior uncertainty. And at the very least, the methods should yield improved performance in the synthetic case.

# ## References
# <a id="Jaz70">[Jaz70]</a>: Jazwinski, A. H. 1970. *Stochastic Processes and Filtering Theory*. Vol. 63. Academic Press.
