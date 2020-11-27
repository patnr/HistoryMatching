# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:nomarker
#     text_representation:
#       extension: .py
#       format_name: nomarker
#       format_version: '1.0'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Tutorial on history matching using ensemble methods
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
# | Navigate                      | Edit              | Exit           | Run                              | Run & go to next                  |
# | -------------                 | : ------------- : | -------------  | : --- :                          | : ------------- :                 |
# | <kbd>↓</kbd> and <kbd>↑</kbd> | <kbd>Enter</kbd>  | <kbd>Esc</kbd> | <kbd>Ctrl</kbd>+<kbd>Enter</kbd> | <kbd>Shift</kbd>+<kbd>Enter</kbd> |
#
# When you open a notebook it starts a **session (kernel/runtime)**
# of Python in the background.
# All of the Python code cells (in a given notebook) are connected
# (they use the same Python kernel and thus share variables, functions, and classes).
# Thus, the **order** in which you run the cells matters. For example:

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
import matplotlib as mpl
from matplotlib import pyplot as plt

# Use numpy's arrays for vectors and matrices. Example constructions:
a = np.arange(10)  # Alternatively: np.array([0,1,2,3,4,5,6,7,8,9])
I = 2*np.eye(10)   # Alternatively: np.diag(2*np.ones(10))

print("Indexing examples:")
print("a        =", a)
print("a[3]     =", a[3])
print("a[0:3]   =", a[0:3])
print("a[:3]    =", a[:3])
print("a[3:]    =", a[3:])
print("a[-1]    =", a[-1])
print("I[:3,:3] =", I[:3, :3], sep="\n")

print("\nLinear algebra examples:")
print("100+a =", 100+a)
print("I@a   =", I@a)
print("I*a   =", I*a, sep="\n")

plt.title("Plotting example")
plt.ylabel("$i \\, x^2$")
for i in range(4):
    plt.plot(i * a**2, label="i = %d" % i)
plt.legend();

# ## Import model
# Run the following cells to import the model and associated tools,
# and initialize some data containers.

import scipy.linalg as sla
from numpy.random import randn
from pylib.std import DotDict
from tqdm.auto import tqdm as progbar
from mpl_tools.misc import freshfig, fig_placement_load, is_notebook_or_qt

import model
import plots
import random_fields
from tools import center, Stats

if is_notebook_or_qt:
    mpl.rcParams.update({'font.size': 13})
    mpl.rcParams["figure.figsize"] = [8, 6]
else:
    plt.ion()
    if mpl.get_backend() != 'MacOSX':
        fig_placement_load()

np.random.seed(4)

# Containers for organizing production and saturation data
#prod = DotDict
#satu = DotDict
#for dct in [prod, satu]:
#    for timespan in ["initial", "past", "present", "future"]:
#        dct[timespan] = DotDict
#        for conditioning in ["Truth", "Prior", "ES", "ES_direct", "EnKS"]:
#             DotDict[timespan][conditioning] = None
prod = DotDict(
    initial = DotDict(), # Estimates at time 0
    past    = DotDict(), # Estimates from 0 to nTime (the present)
    present = DotDict(), # Estimates at nTime, but without re-runs
    future  = DotDict(), # Predictions for time > nTime
)
satu = DotDict(
    initial = DotDict(),
    past    = DotDict(),
    present = DotDict(),
    future  = DotDict(),
)

# ## Model and case
# The reservoir model, which takes up about 100 lines of python code,
# is a 2D, two-phase, immiscible, incompressible simulator using TPFA.
# It was translated from the matlab code here
# http://folk.ntnu.no/andreas/papers/ResSimMatlab.pdf
#
# The model supports inhomogeneous permeabilities and porosities,
# but to keep things as simple as possible we will
# only be estimating the water saturation.
# Specifically, we will focus on the initial saturation,
# as is commonplace in ensemble HM.
# The data will be production saturations.
#
# The boundary conditions are of the Dirichlet type, specifying zero flux.
# The source terms must therefore equal the sink terms.
# This is ensured by the `init_Q` function below.

# #### Well specification
# We here specify the wells as point sources and sinks,
# giving their placement and flux.

model.init_Q(
    # NB: seed dependent!
    # inj =rand(2,3),
    # prod=rand(4,3)

    inj =[
        # x    y     rate
        [0.10, 0.30, 1.00],
        [0.90, 0.70, 1.00]],
    prod=[
        [0.10, 0.10, 1.00],
        [0.10, 0.50, 1.00],
        [0.10, 0.90, 1.00],
        # [0.50, 0.10, 1.00],
        # [0.50, 0.50, 1.00],
        # [0.50, 0.90, 1.00],
        [0.90, 0.10, 1.00],
        [0.90, 0.50, 1.00],
        [0.90, 0.90, 1.00],
    ]
);


# #### Surface of water saturation -- True distribution
# The truth is sampled from a Gaussian field with a given variogram,
# yielding the following moments

Mean = 0.5
Cov = 0.3**2 * random_fields.gen_cov(model.grid, radius=0.5)


# We use this to sample the true saturation ("truth") ...

C12 = sla.sqrtm(Cov).real.T
satu.initial.Truth = 0.5 + randn(model.M) @ C12

# which we then truncate to [0,1]

satu.initial.Truth = model.truncate_01(satu.initial.Truth)


# #### Simulation to generate the synthetic truth

dt = 0.025  # time step (external)
nTime = 28  # number of steps

# Note that the model outputs not just the "state" (the saturation)
# but also the "observations" (the production).
# We store these in our data containers.

satu.past.Truth, prod.past.Truth = \
    model.simulate(nTime, satu.initial.Truth, dt)

# ##### Animation
# Plotted below is the oil saturation before/after production.
# Injection (resp. production) wells are marked with triangles pointing down (resp. up).
#
#
# <mark><font size="-1">
# <em>Warning:</em> takes a while to load.
# </font></mark>
#


ani = plots.animate1(satu.past.Truth, prod.past.Truth)
plots.display(ani)


# #### Noisy obs
# In reality, observations are never perfect.
# To reflect this, we corrupt the observations by adding a bit of noise.


prod.past.obs  = prod.past.Truth.copy()
nProd = len(model.producers)  # num. of obs (per time)
R = 0.1**2 * np.eye(nProd)
for iT in range(nTime):
    prod.past.obs[iT] += R @ randn(nProd)


fig, ax = freshfig(2)
hh_y = plots.production1(ax, prod.past.Truth, obs=prod.past.obs)


# ## Initial ensemble
# The initial ensemble is generated in the same manner as the (synthetic) truth,
# using the same mean and covariance.
# Thus, the members are "statistically indistinguishable" to the truth.
# This assumption underlies ensemble methods.

N = 40
satu.initial.Prior = 0.5 + randn(N, model.M) @ C12
satu.initial.Prior = model.truncate_01(satu.initial.Prior)


# Another assumption underying ensemble methods is that of Gaussianity (Normality).
# However, saturation, porosity, and other variables
# (that we may be subjecting to estimation)
# may only be valid within a certain range.
# Indeed, that is why we are truncating the truth and ensemble.
#
# This makes the field non-Gaussian,
# implying an additional approximation to the ensemble methods.

plots.hists(65, satu.initial, "Water saturation")


# Note: the "Truth" histogram is liable to vary significantly
# (from one run of the notebook to the next)
# because of a relatively small (effective/de-correlated) sample size.
# So its shape varies.
# On the other hand, the ensemble histogram pretty much always
#  its Gaussian origins

# Below we can see some realizations (members) from the ensemble.


plots.oilfield_realizations(23, satu.initial, "Prior")


# #### Eigenvalue specturm
# In practice, of course, we would not be using an explicit `Cov` matrix
# when generating the prior ensemble, because it would be too large.
# However, since this synthetic case in being made that way,
# let's inspect its spectrum.

eigs = sla.eigvalsh(Cov)[::-1]
ii = 1+np.arange(len(eigs))
fig, ax = freshfig(21)
# ax.loglog(ii,eigs)
ax.semilogx(ii, eigs)
ax.grid(True, "minor", axis="x")
ax.grid(True, "major", axis="y")
ax.set(xlabel="eigenvalue #", ylabel="var.",
       title="Spectrum of initial, true cov");

# It appears that the spectrum tails off around $N=30$,
# so maybe this ensemble size will suffice.
# However, try plots the above using `loglog` instead of `semilogx`,
# and you might not be so convinced.
# Nevertheless, as we shall see, it does seem to yield tolerable results,
# even without localization.

# ## Assimilation
# We now forecast the prior ensemble from the initial time
# until the current time.
# Note that this for-loop is "embarrasingly parallelizable",
# because each iterate is complete indepdendent
# (requires no communication) from the others.

def forecast_ensemble(nSteps, sat0):
    satu = np.zeros((nSteps, N, model.M))
    prod = np.zeros((nSteps, N, nProd))
    for n, xn in enumerate(progbar(sat0)):
        s, p = model.simulate(nSteps, xn, dt, pbar=False)
        satu[:, n, :] = s
        prod[:, n, :] = p
    return satu, prod

satu.past.Prior, prod.past.Prior = forecast_ensemble(nTime, satu.initial.Prior)


# ### Assimilation with ES
# First, we assimilate (history match) using the batch method: ensemble smoother (ES).

# This involves "stringing-out / concatenating" a multivariate timeseries
# into one, long vector.
#
# - Mathematically, we often call this augmentation approach.
# - Programmatically, we call it "flattening/(un)ravelling".
#   It is implemented in the cell below.


class EnUpdate:
    """Prepare the update/conditioning (Bayes' rule) for an ensemble,

    given a (vector) observations an an ensemble (matrix).

    NB: obs_err_cov is treated as diagonal.
    Alternative: use `sla.sqrtm`.
    """

    def __init__(self, obs, obs_ens, obs_err_cov):
        Y           = center(obs_ens)
        obs_cov     = obs_err_cov*(N-1) + Y.T@Y
        obs_pert    = randn(N, len(obs)) @ np.sqrt(obs_err_cov)
        innovations = obs - (obs_ens + obs_pert)

        # (pre-) Kalman gain * Innovations
        self.KGdY = innovations @ sla.pinv2(obs_cov) @ Y.T
        # Note: formula is transposed, and reversed (vs. literature standards),
        # because the members are here stacked as rows (vs. columns).

    def apply(self, E):
        # Update
        E = E + self.KGdY @ center(E)

        # Post-process
        E = model.truncate_01(E)

        # Should also inflate & rotate?

        return E


def unfold_time(EE):
    nTime, N, M = EE.shape
    return EE.swapaxes(0, 1).reshape((N, nTime*M))

def as_batch(apply, EE):
    nTime, N, M = EE.shape

    # Concatenate time to make batch
    E = unfold_time(EE)

    # Apply update to batch
    E = apply(E)

    # Re-shape into time-series
    EE = E.reshape(N, nTime, -1).swapaxes(0, 1)

    return EE

ES_update = EnUpdate(
    prod.past.obs.ravel(),
    unfold_time(prod.past.Prior),
    sla.block_diag(*[R]*nTime))

# The main job of ensemble methods in history matching is
# to update (i.e. computed the approximate posterior of)
# the initial conditions (saturation) and any model parameters,
# here termed `satu.initial.ES` (updated ensemble for t=0).

satu.initial.ES = ES_update.apply(satu.initial.Prior)

# However, ensemble methods are very extensible (same as linear regression).
# For the purposes of this tutorial, we compute several by-products.
# One of these is `satu.present.ES`: updated ensemble at the current time.

satu.present.ES = ES_update.apply(satu.past.Prior[-1])

# Let's plot the updated, initial ensemble.

plots.oilfield_realizations(27, satu.initial, "ES")


# An updated estimate of the production can be obtained
# by re-running the simulation model.

satu.past.ES, prod.past.ES = forecast_ensemble(nTime, satu.initial.ES)

# Alternatively, we might update the production profiles "directly".
# We will revisit this strategy below, when discussing prediction.

prod.past.ES_direct = as_batch(ES_update.apply, prod.past.Prior)


# ### Assimilation with EnKS
# Next, we assimilate (history match) using the EnKS.

satu.initial.EnKS = satu.initial.Prior.copy()
E                 = satu.initial.Prior.copy()
Eobs              = np.zeros((N, nProd))

for iT in progbar(range(nTime)):

    # Forecast
    for n, xn in enumerate(E):
        E[n], Eobs[n] = model.simulate(1, xn, dt)

    # Analysis
    Seq_update        = EnUpdate(prod.past.obs[iT], Eobs, R)
    E                 = Seq_update.apply(E)
    satu.initial.EnKS = Seq_update.apply(satu.initial.EnKS)


satu.present.EnKS = E.copy()

plots.oilfield_realizations(28, satu.initial, "EnKS")


# An updated estimate of the production can be obtained by
# re-running the simulation model.

satu.past.EnKS, prod.past.EnKS = forecast_ensemble(nTime, satu.initial.EnKS)


# ### Compare error in initial saturation
# In this synthetic case, we have access to the truth (that generated the observations).
# We can then compare the actual error.
# Contrary to just using the data mismatch, overfitting to data will get penalized!

print("Prior: ", Stats(satu.initial.Truth, satu.initial.Prior))
print("ES   : ", Stats(satu.initial.Truth, satu.initial.ES))
print("EnKS : ", Stats(satu.initial.Truth, satu.initial.EnKS))


# Let's plot mean fields.
#
# NB: Caution! Mean fields are liable to be less rugged than the truth.
# As such, their importance must not be overstated
# (they're just one esitmator out of many).
# Instead, whenever a decision is to be made,
# all of the members should be included in the decision-making process.

plots.oilfield_means(25, satu.initial)


# ### Correlation fields
# Correlation fields (vs. a given variable, i.e. point) show where
# the ensemble update can have an effect (vs. data at that point).
#
# NB: Correlations are just one part of the update (gain) operation (matrix),
# which also involves:
# - the sensitivity matrix (the obs. operator)
# - the inter-dependence of elements.
# - the relative sizes of the errors (prior vs. likelihood,
#   as well as one vector element vs. another).
#
# Plot of correlation fields vs. a specific point (black point).

iWell = 0
xy_coord = model.producers[iWell, :2]
plots.correlation_fields(22, satu.initial, xy_coord, "Initial")
plots.correlation_fields(24, satu.present, xy_coord, "Present")


# ## Diagnostics

# ### Plot past production

# shown = ["Truth", "obs", "Prior", "ES", "ES_direct", "EnKS"]
shown = ["Truth", "obs", "Prior", "ES"]
shown = {k: v for k, v in prod.past.items() if k in shown}
plots.productions(74, shown, "past")


# ##### Comment on the prior
# Note that the prior "surrounds" the data.
# This the likely situation in our synthetic case,
# where the truth was generated by the same random draw process as the ensemble.
#
# In practice, this is often not the case.
# If so, you might want to go back to your geologists and tell them
# that something is amiss.
# You should then produce a revised prior with better properties.
#
# Note: the above instructions sound like statistical heresy.
# We are using the data twice over
# (on the prior, and later to update/condition the prior).
# However, this is justified to the extent that prior information is difficult
# to quantify and encode.
# Too much prior adaptation, however, and you risk overfitting!
# Ineed, it is a delicate matter.
#
#
# ##### Comment on the posteriors
# If the assumptions (statistical indistinguishability, Gaussianity)
# are not too far off, then the ensemble posteriors (ES, EnKS, ES_direct)
# should also surround the data, but with a tighter fit.

# ### Prediction
# We now prediction the future production (and saturation fields)
# by forecasting from the (updated) present saturations.
# An alternative way is to forecast all the way from the
# (updated) initial saturations.

satu.future.Truth, prod.future.Truth = model.   simulate(nTime, satu.past.Truth[-1], dt)
satu.future.Prior, prod.future.Prior = forecast_ensemble(nTime, satu.past.Prior[-1])
satu.future.ES,    prod.future.ES    = forecast_ensemble(nTime, satu.present.ES)
satu.future.EnKS,  prod.future.EnKS  = forecast_ensemble(nTime, satu.present.EnKS)

# Instead of running the model on the posterior/conditonal/updated saturation,
# we could also condition the future, prior production data directly.
# Unlike the "past" case, this usually yields worse RMSE values
# than using the model on the updated saturation ensemble,
# because reservoir dynamics are nonlinear.

prod.future.ES_direct = as_batch(ES_update.apply, prod.future.Prior)


# ### Plot future production
# Let's see if the posterior predictions are better
# (improved data match) than the prior predictions.

# shown = ["Truth", "obs", "Prior", "ES", "ES_direct", "EnKS"]
shown = ["Truth", "obs", "Prior", "ES"]
shown = {k: v for k, v in prod.future.items() if k in shown}
plots.productions(75, shown, "Future")


# <!--
# # ! Perhaps the ensemble spread is too large for history matching methods to be
# # ! effective (because they produce too nonlinear behaviours). In that case, we
# # ! might adjust our test case by reducing the initial (prior) ensemble spread,
# # ! also adjusting its mean towards the truth. A less artifical means is Kriging
# # ! (geostatistics), illustrated below. However, with the default parameters,
# # ! this adjustment is not necessary, but is left for completeness.
# # !
# # ! The obs. locations used for Kriging are marked with white dots.
# # !
# # ! kriging_inds = linspace(0, model.M-1, 10).astype(int)
# # ! kriging_obs = satu.initial.Truth[kriging_inds]
# # ! Cxy = Cov[:,kriging_inds]
# # ! Cyy = Cov[kriging_inds][:,kriging_inds]
# # ! Reg = Cxy @ nla.pinv(Cyy)
# # ! Kriged = satu.initial.Truth.mean() + Reg @ (kriging_obs-satu.initial.Truth.mean())
# # !
# # ! print("Error for Krig.: %.4f"%norm(satu.initial.Truth-Kriged))
# # !
# # ! Eb = Kriged + 0.4*center(satu.initial.Prior)
# # ! fig, axs = freshfig(24,nrows=3,ncols=4,sharex=True,sharey=True)
# # ! axs[0,0].plot(*array([ind2xy(j) for j in kriging_inds]).T, 'w.',ms=10)
# # ! plots.realizations(axs,Eb,"Krig/Prior")
# # !
# # ! Eb = satu.initial.Prior + (Reg @
# # !     (kriging_obs-satu.initial.Prior[:,kriging_inds]).T).T
# # ! Eb = satu.initial.Prior.copy()
# # !
# # ! Eb = model.truncate_01(Eb)
# # -->
