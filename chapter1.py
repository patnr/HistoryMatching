# -*- coding: utf-8 -*-
# # Tutorial on ensemble history matching and optimisation (TODO)
#
# Copyright Patrick N. Raanes, NORCE, 2020.
#
# This (Jupyter/Python) notebook presents
# a tutorial on history matching (HM) using ensemble methods.
#
# This is a work in progress.
# Details may be lacking.
# Don't hesitate to send me an email with any questions you have.

# ## Jupyter notebooks
# the format used for these tutorials.
# Notebooks combine **cells** of code (Python) with cells of text (markdown).
# The exercises in these tutorials only require light Python experience.
# For example, edit the cell below (double-click it),
# insert your name,
# and run it (press "Run" in the toolbar).

name = "Batman"
print("Hello world! I'm " + name)
for i, c in enumerate(name):
    print(i, c)

# You will likely be more efficient if you know these
# **keyboard shortcuts** to interact with cells:
#
# | Navigate                      |    | Edit              |    | Exit           |    | Run                              |    | Run & go to next                  |
# | -------------                 | -- | ----------------- | -- | --------       | -- | -------                          | -- | -----------------                 |
# | <kbd>↓</kbd> and <kbd>↑</kbd> |    | <kbd>Enter</kbd>  |    | <kbd>Esc</kbd> |    | <kbd>Ctrl</kbd>+<kbd>Enter</kbd> |    | <kbd>Shift</kbd>+<kbd>Enter</kbd> |
#
# When you open a notebook it starts a **session (kernel/runtime)**
# of Python in the background.
# All of the Python code cells (in a given notebook) are connected
# (they use the same Python kernel and thus share variables, functions, and classes).
# Thus, the **order** in which you run the cells matters.

# One thing you must know is how to **restart** the Python session,
# which clears all of your variables, functions, etc,
# so that you can start over.
# Test this now by going through the top menu bar:
# `Kernel` → `Restart & Clear Output`.
# But rembember to run the above cell again!

# There is a huge amount of libraries available in **Python**,
# including the popular `scipy` (with `numpy` at its core) and `matplotlib` packages.
# These are imported (and abbreviated) as `sp`, `np`, and `mpl` and `plt`.
# Try them out by running the following, which illustrates some algebra
# using syntax reminiscent of Matlab.

import numpy as np
from matplotlib import pyplot as plt

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

# ## Setup
# Run this to download and install the project's requirements:

# !wget -qO- https://raw.githubusercontent.com/patricknraanes/HistoryMatching/master/colab_bootstrap.sh | bash -s

# Run the following cells to import some tools...

from copy import deepcopy

import numpy as np
import scipy.linalg as sla
from matplotlib import pyplot as plt
from mpl_tools.misc import freshfig
from numpy.random import randn
from patlib.dict_tools import DotDict
from tqdm.auto import tqdm as progbar

# and the model, ...

import geostat
import simulator
import simulator.plotting as plots
from simulator import simulate
from tools import RMS, center


plots.COORD_TYPE = "rel"
plots.setup()

# ... and initialize some data containers.

# +
# Permeability
perm = DotDict()

# Production (water saturation)
prod = DotDict(
    past=DotDict(),
    future=DotDict(),
)

# Water saturation
wsat = DotDict(
    initial=DotDict(),
    past=DotDict(),
    future=DotDict(),
)
# -

# Enable exact reproducibility by setting random generator seed.

# seed = np.random.seed(10)  # easy
# seed = np.random.seed(22)  # easy
# seed = np.random.seed(30)  # easy
seed = np.random.seed(5)  # harder

# ## Model and case specification
# The reservoir model, which takes up about 100 lines of python code,
# is a 2D, two-phase, immiscible, incompressible simulator using TPFA.
# It was translated from the matlab code here
# http://folk.ntnu.no/andreas/papers/ResSimMatlab.pdf
#
# We will estimate the log permeability field and (TODO).
# The data will consist in the production saturations.

model = simulator.ResSim(Nx=20, Ny=20, Lx=2, Ly=1)

# #### Permeability sampling
# We work with log permeabilities, which can (in principle) be Gaussian.

def sample_log_perm(N=1):
    lperms = geostat.gaussian_fields(model.mesh(), N, 0.8)
    return lperms

def log2perm(lperms):
    # return np.exp(3*lperms)
    return 0.5 + .1*lperms

# The transformation of the parameters to model input is effectively part of the forward model.

def set_perm(model, log_perm_array):
    # p = np.exp(3*log_perm_array)
    p = 0.5 + .1*log_perm_array
    p = p.reshape(model.shape)
    model.Gridded.K = np.stack([p, p])

# Here we sample the permeabilitiy of the (synthetic) truth.

perm.Truth = sample_log_perm()
set_perm(model, perm.Truth)

# #### Well specification
# We here specify the wells as point sources and sinks,
# giving their placement and flux.
#
# The boundary conditions are of the Dirichlet type, specifying zero flux.
# The source terms must therefore equal the sink terms.
# This is ensured by the `init_Q` function used below.

# +
# Manual well specification
# model.init_Q(
#     #     x    y     rate
#     inj =[
#         [0.50, 0.50, 1.00],
#     ],
#     prod=[
#         [0.10, 0.10, 1.00],
#         # [0.10, 0.50, 1.00],
#         [0.10, 0.90, 1.00],
#         # [0.50, 0.10, 1.00],
#         # [0.50, 0.50, 1.00],
#         # [0.50, 0.90, 1.00],
#         [0.90, 0.10, 1.00],
#         # [0.90, 0.50, 1.00],
#         [0.90, 0.90, 1.00],
#     ]
# );

# Wells on a grid
well_grid = np.linspace(0.1, .9, 6)
well_grid = np.meshgrid(well_grid, well_grid)
well_grid = np.stack(well_grid + [np.ones_like(well_grid[0])])
well_grid = well_grid.T.reshape((-1, 3))
model.init_Q(
    inj =[[0.50, 0.50, 1.00]],
    prod=well_grid
);

# # Random setting
# model.init_Q(
#     inj =rand(1, 3),
#     prod=rand(8, 3)
# );
# -

# #### Plot true field

fig, ax = freshfig(110)
cs = plots.field(model, ax, perm.Truth)
# cs = plots.field(model, ax, log2perm(perm.Truth), locator=ticker.LogLocator())
plots.well_scatter(model, ax, model.producers, inj=False)
plots.well_scatter(model, ax, model.injectors, inj=True)
fig.colorbar(cs)
fig.suptitle("True field");


# #### Define obs operator

obs_inds = [model.xy2ind(x, y) for (x, y, _) in model.producers]
def obs(water_sat):
    return [water_sat[i] for i in obs_inds]
obs.length = len(obs_inds)

# #### Simulation to generate the synthetic truth evolution and data

wsat.initial.Truth = np.zeros(model.M)
# wsat.initial.Truth = log2perm(perm.Truth.squeeze())  # TODO rm
nTime = 50
dt = 0.025
wsat.past.Truth, prod.past.Truth = simulate(
    model.step, nTime, wsat.initial.Truth, dt, obs)

# ##### Animation
# Run the code cell below to get an animation of the oil saturation evoluation.
# Injection (resp. production) wells are marked with triangles pointing down (resp. up).
#
# <mark><font size="-1">
# <em>Note:</em> takes a while to load.
# </font></mark>

from matplotlib import rc
rc('animation', html="jshtml")
ani = plots.dashboard(model, wsat.past.Truth, prod.past.Truth, animate=True, title="Truth");
ani


# #### Noisy obs
# In reality, observations are never perfect.
# To reflect this, we corrupt the observations by adding a bit of noise.

prod.past.Noisy = prod.past.Truth.copy()
nProd = len(model.producers)  # num. of obs (per time)
R = 1e-6 * np.eye(nProd)
for iT in range(nTime):
    prod.past.Noisy[iT] += R @ randn(nProd)


# Plot of observations (and their noise):

fig, ax = freshfig(120)
hh_y = plots.production1(ax, prod.past.Truth, obs=prod.past.Noisy)

# ## Prior
# The prior ensemble is generated in the same manner as the (synthetic) truth,
# using the same mean and covariance.
# Thus, the members are "statistically indistinguishable" to the truth.
# This assumption underlies ensemble methods.

N = 100
perm.Prior = sample_log_perm(N)

# Depending on the parameter type, transformations may be in order that will yield non-Gaussian distributions. Inspect the density by the histograms below.

fig, ax = freshfig(130, figsize=(12, 3))
for label, data in perm.items():
    ax.hist(data.ravel(), label=label, alpha=0.4, density=True)
ax.set(ylabel="rel. frequency")
ax.legend();

# Below we can see some realizations (members) from the ensemble.

plots.subplots(model, 140, plots.field, perm.Prior,
               figsize=(14, 5), title="Prior -- some realizations");

# #### Eigenvalue specturm
# In practice, of course, we would not be using an explicit `Cov` matrix when generating the prior ensemble, because it would be too large.  However, since this synthetic case in being made that way, let's inspect its spectrum.

U, svals, VT = sla.svd(perm.Prior)
ii = 1+np.arange(len(svals))
fig, ax = freshfig(150, figsize=(12, 5))
ax.loglog(ii, svals)
# ax.semilogx(ii, svals)
ax.grid(True, "minor", axis="x")
ax.grid(True, "major", axis="y")
ax.set(xlabel="eigenvalue #", ylabel="var.",
       title="Spectrum of initial, true cov");

# Finally, we set the prior for the state variable to a single (i.e. deterministic) field. This means that there is no uncertainty in the state variable.

wsat.initial.Prior = np.tile(wsat.initial.Truth, (N, 1))
# wsat.initial.Prior = log2perm(perm.Prior)  # TODO rm


# ## Assimilation

# #### Propagation
# Ensemble methods obtain observation-parameter sensitivities from the covariances of the ensemble run through the model. Note that this for-loop is "embarrasingly parallelizable", because each iterate is complete indepdendent (requires no communication) from the others.

def forecast(nSteps, wsats0, perms):
    """Forecast for an ensemble."""

    # Allocate
    production = np.zeros((N, nSteps, nProd))
    saturation = np.zeros((N, nSteps+1, model.M))

    for n, (wsat0, perm) in enumerate(progbar(list(zip(wsats0, perms)), "Members")):

        # Set ensemble
        model_n = deepcopy(model)
        set_perm(model_n, perm)

        # Simulate
        s, p = simulate(model_n.step, nSteps, wsat0, dt, obs, pbar=False)

        # Write
        # Note: we only really need the last entry in the saturation series.
        production[n] = p
        saturation[n] = s

    return saturation, production

wsat.past.Prior, prod.past.Prior = forecast(
    nTime, wsat.initial.Prior, perm.Prior)

# ### Ensemble smoother

def ES(ensemble, obs_ens, observation, obs_err_cov, infl=1.0):
    """Update/conditioning (Bayes' rule) for an ensemble,

    according to the "ensemble smoother" algorithm,

    given a (vector) observations an an ensemble (matrix).

    NB: obs_err_cov is treated as diagonal.
    Alternative: use `sla.sqrtm`.
    """

    Y           = infl*center(obs_ens)
    obs_cov     = obs_err_cov*(N-1) + Y.T@Y
    obs_pert    = randn(N, len(observation)) @ np.sqrt(obs_err_cov)
    innovations = observation - (obs_ens + obs_pert)

    # (pre-) Kalman gain * Innovations
    KGdY = innovations @ sla.pinv2(obs_cov) @ Y.T
    # Note: formula is transposed, and reversed (vs. literature standards),
    # because the members are here stacked as rows (vs. columns).

    E = ensemble

    # Inflate
    E = E.mean(axis=0) + infl*(E - E.mean(axis=0))

    # Update
    E = E + KGdY @ center(E)
    return E


# ### Iterative ensemble smoother
# TODO

# #### Update

perm.ES = ES(
    ensemble    = perm.Prior,
    obs_ens     = prod.past.Prior.reshape((N, -1)),
    observation = prod.past.Noisy.reshape(-1),
    obs_err_cov = sla.block_diag(*[R]*nTime),
)

# Let's plot the updated, initial ensemble.

plots.subplots(model, 160, plots.field, perm.ES,
               figsize=(14, 5), title="ES posterior -- some realizations");


# #### Diagnostics

print("Stats vs. true field")
print("Prior: ", RMS(perm.Truth, perm.Prior))
print("ES   : ", RMS(perm.Truth, perm.ES))

# #### Plot of means
# Let's plot mean fields.
#
# NB: Caution! Mean fields are liable to be less rugged than the truth.
# As such, their importance must not be overstated
# (they're just one esitmator out of many).
# Instead, whenever a decision is to be made,
# all of the members should be included in the decision-making process.

perm._means = DotDict((k, perm[k].mean(axis=0)) for k in perm
                      if not k.startswith("_"))

plots.subplots(model, 170, plots.field, perm._means,
               figsize=(14, 4), title="Particular fields.");

# ## Correlations
# NB: Correlations are just one part of the update (gain) operation (matrix),
# which also involves:
# - the sensitivity matrix (the obs. operator)
# - the inter-dependence of elements.
# - the relative sizes of the errors (prior vs. likelihood,
#   as well as one vector element vs. another).

# +
# Not very interesting coz its perm-perm
# iWell = 2
# xy_coord = model.producers[iWell, :2]
# plots.correlation_fields(
#    model, 180, intersect(perm, ["Prior", "ES"]),
#    xy_coord, "Initial corr.")
# -


# ## Past production (data mismatch)

# We already have the past true and prior production profiles.
# Let's add to that the production profiles of the posterior.

wsat.past.ES, prod.past.ES = forecast(nTime, wsat.initial.Prior, perm.ES)

# Plot them all together:

plots.productions(190, prod.past, figsize=(14, 5), title="-- Past");

# ##### Comment on prior
# Note that the prior "surrounds" the data. This the likely situation in our synthetic case, where the truth was generated by the same random draw process as the ensemble.
#
# In practice, this is often not the case. If so, you might want to go back to your geologists and tell them that something is amiss. You should then produce a revised prior with better properties.
#
# Note: the above instructions sound like statistical heresy. We are using the data twice over (on the prior, and later to update/condition the prior). However, this is justified to the extent that prior information is difficult to quantify and encode. Too much prior adaptation, however, and you risk overfitting! Ineed, it is a delicate matter.

# ##### Comment on posterior
# If the assumptions (statistical indistinguishability, Gaussianity) are not too far off, then the ensemble posteriors (ES, EnKS, ES_direct) should also surround the data, but with a tighter fit.

# #### Data mismatch

print("Stats vs. past production (i.e. observations)")
print("Prior: ", RMS(prod.past.Noisy, prod.past.Prior))
print("ES   : ", RMS(prod.past.Noisy, prod.past.ES))

# Note that the standard deviation is much smaller than the RMSE. This may be remedied by inflation (which won't necessarily help with the RMSE), localisation (a powerful fix) and a bigger ensemble size (simple, but costly).

# ## Prediction
# We now prediction the future production (and saturation fields) by forecasting using the (updated) estimates.

wsat.future.Truth, prod.future.Truth = simulate(
    model.step, nTime, wsat.past.Truth[-1], dt, obs)

wsat.future.Prior, prod.future.Prior = forecast(
    nTime, wsat.past.Prior[:, -1, :], perm.Prior)

wsat.future.ES, prod.future.ES = forecast(
    nTime, wsat.past.ES[:, -1, :], perm.ES)

# #### Plot future production

plots.productions(200, prod.future, figsize=(14, 5), title="-- Future");

print("Stats vs. (supposedly unknown) future production")
print("Prior: ", RMS(prod.future.Truth, prod.future.Prior))
print("ES   : ", RMS(prod.future.Truth, prod.future.ES))
