# -*- coding: utf-8 -*-
# # History matching and optimisation with ensembles – an interactive tutorial
#
# Copyright Patrick N. Raanes, NORCE, 2020.
#
# This is a self-contained tutorial on history matching (HM) and optimisation using ensemble methods.
# - If you "run all" then you will burn through it in 5 min.
# - For a more detailed reading, expect to spend around 5 hours.
# - The code emphasises simplicity, not generality.
# - Do not hesitate to file issues on
#   [GitHub](https://github.com/patnr/HistoryMatching),
#   or submit pull requests.

# ## Python in Jupyter

# **Jupyter notebooks** combine **cells/blocks** of code (Python) and text (markdown).
#
# For example, try to **edit** the cell below to insert your name, and then **run** it.

# name = "Batman"
# print("Hello world! I'm", name)

# Next, as an exercise, try to **insert** a new cell here, and compute `23/3`

# You will likely be more efficient if you know these **keyboard shortcuts**:
#
# | Navigate                      |    | Edit              |    | Exit           |    | Run & advance                     |
# | -------------                 | -- | ----------------- | -- | --------       | -- | -------------                     |
# | <kbd>↓</kbd> and <kbd>↑</kbd> |    | <kbd>Enter</kbd>  |    | <kbd>Esc</kbd> |    | <kbd>Shift</kbd>+<kbd>Enter</kbd> |
#
# When you open a notebook it starts a **session (interpreter/kernel/runtime)** of
# Python in the background.  All of the code cells (in a given notebook) are connected
# (share kernel and thus share variables, functions, and classes).  Thus, the **order**
# in which you run the cells matters.  One thing you must know is how to **restart** the
# session, so that you can start over. Try to locate this option via the menu bar at the
# top.

# If you're on **Google Colab**, run the cell below to install the requirements.
# Otherwise (and assuming you have done the installation described in the README),
# you can skip/delete this cell.

remote = "https://raw.githubusercontent.com/patnr/HistoryMatching"
# !wget -qO- {remote}/master/colab_bootstrap.sh | bash -s

# There is a huge amount of libraries available in **Python**,
# including the popular `numpy (np)` and `matplotlib/pyplot (mpl/plt)` packages.
# Try them out by running in the next few cells following,
# which illustrates some algebra using syntax reminiscent of Matlab.

import numpy as np
from matplotlib import pyplot as plt
from tools import plotting

plotting.init()

# Use numpy arrays for vectors, matrices. Examples:
a = np.arange(10)  # OR: np.array([0,1,2,3,4,5,6,7,8,9])
Id = 2 * np.eye(10)  # OR: np.diag(2*np.ones(10))

# print("Indexing examples:")
# print("a         =", a)
# print("a[3]      =", a[3])
# print("a[0:3]    =", a[0:3])
# print("a[:3]     =", a[:3])
# print("a[3:]     =", a[3:])
# print("a[-1]     =", a[-1])
# print("Id[:3,:3] =", Id[:3, :3], sep="\n")

# print("Linear algebra examples:")
# print("100 + a =", 100 + a)
# print("Id @ a  =", Id @ a)
# print("Id * a  =", Id * a, sep="\n")

# fig, ax = plotting.freshfig("Plotting example", figsize=(6, 3))
# ax.set_ylabel("$i \\, x^2$")
# for i in range(4):
#     ax.plot(i * a**2, label="i = %d" % i)
# ax.legend()
# plt.show()

# Run the following cells to import yet more tools.

import copy
import numpy.random as rnd
import scipy.linalg as sla
from numpy import sqrt
from struct_tools import DotDict as Dict

# ## Problem case (simulator, truth, obs)

# For exact reproducibility of our problem/case, we set the random generator seed.

seed = rnd.seed(1)

# Our reservoir simulator takes up about 100 lines of python code. This may seem
# outrageously simple, but serves the purpose of *illustrating* the main features of
# the history matching process. Indeed, we do not detail the simulator code here, but
# simply import it from the accompanying python modules, together with the associated
# plot functionality, the (geostatistical) random field generator, and some linear
# algebra. Hence our focus and code will be of aspects directly related to the history
# matching and optimisation process.

import TPFA_ResSim as simulator
import tools.localization as loc
from tools import geostat, plotting, utils
from tools.utils import center, apply, emph

# In short, the model is a 2D, two-phase, immiscible, incompressible simulator using
# two-point flux approximation (TPFA) discretisation. It was translated from the Matlab
# code here http://folk.ntnu.no/andreas/papers/ResSimMatlab.pdf

model = simulator.ResSim(Nx=20, Ny=20, Lx=2, Ly=1)

# The following declares some data containers to help us keep organised.
# The names have all been shortened to 4 characters, but this is just
# to obtain more convenient code alignment and readability.

# +
# Permeability
perm = Dict()

# Production
prod = Dict(
    past=Dict(),
    futr=Dict(),
)

# Water saturation
wsat = Dict(
    past=Dict(),
    futr=Dict(),
)
# -

# Technical note: This data hierarchy is convienient in *this* notebook/script,
# especially for plotting purposes. For example, we can with ease refer to
# `wsat.past.Truth` and `wsat.past.Prior`. The former will be a numpy array of shape
# `(nTime, model.Nxy)` and the latter will have shape `(N, nTime, model.Nxy)` where
# `N` is the size of the ensemble. However, in other implementations, different choices
# for the data structure may be more convenient, e.g. where the different types of
# the unknowns are merely concatenated along the last axis, rather than being kept in
# separate dicts.

# #### The unknown: permeability
# We will estimate the log permeability field.  We *parameterize* the permeability,
# meaning that they are defined via some transform (function), which becomes part of the
# forward model. We term the parameterized permeability fields "pre-permeability".
# *If* we use the exponential, then we will be working with log-permeabilities.
# Here we use an almost-exponential transform (presumably making the problem slightly trickier).


def perm_transf(x):
    return 0.1 + np.exp(5 * x)
    # return 1000*np.exp(3*x)


# In any case, the transform should be chosen so that the parameterized permeabilities
# are suited for ensemble methods, i.e. are distributed as a Gaussian.  But this
# consideration must be weighted against the fact that that nonlinearity (another
# difficulty for ensemble methods) in the transform might add to the nonlinearity of
# the total/composite forward model.
# Since this is a synthetic case, we can freely choose the distribution of the
# parameterized permeabilities, which we take to be Gaussian.


def sample_prior_perm(N):
    lperms = geostat.gaussian_fields(model.mesh, N, r=0.8)
    return lperms


# For many kinds of parameters, one typically has to write "setter" functions that take
# the vector of parameter parameter values, and apply it to the specific model implementation.


def set_perm(model, log_perm_array):
    """Set perm. in model code (both x and y components)."""
    p = perm_transf(log_perm_array)
    p = p.reshape(model.shape)
    model.K = np.stack([p, p])


perm.Truth = sample_prior_perm(1)
set_perm(model, perm.Truth)

# #### Wells

# In this model, wells are represented simply by point **sources** and **sinks**.
# This is of course incredibly basic and not realistic, but works for our purposes.
# So all we need to specify is their placement and flux (which we will not vary in time).
# The code below generates the (x,y) coordinates of a point near each of the 4 corners.

near01 = np.array([0.12, 0.87])
xy_4corners = [[x, y]
               for y in model.Ly*near01
               for x in model.Lx*near01]  # fmt: skip

# Since the **boundary conditions** are Dirichlet, specifying *zero flux*, and the fluid
# is incompressible, the total of the source terms must equal that of the sinks.
# If this is not the case, the model will raise an error when run.

nPrd = len(xy_4corners)
model.prd_xy = xy_4corners
model.inj_xy = [[model.Lx / 2, model.Ly / 2]]
model.inj_rates = [[1]]
model.prd_rates = np.ones((nPrd, 1)) / nPrd

# As detailed in the model docs, when `inj_rates.shape[1] == 1` (as above),
# the rates do not vary in time.

# #### Plot
# Let's take a moment to visualize the (true) model permeability field,
# and the well locations. Note that they have all been collocated to cell centres.

# fig, ax = plotting.freshfig("True field")
# # model.plt_field(ax, perm.Truth, "pperm")
# model.plt_field(ax, perm_transf(perm.Truth), "perm", grid=True);  # fmt: skip

# #### Observation operator
# The data will consist in the water saturation of at the production well locations.
# I.e. there is no well model. It should be pointed out, however, that ensemble
# methods technically support (though your accuracy mileage may vary, again, depending
# on the incurred nonlinearity and non-Gaussianity) observation models of any complexity.

prod_inds = model.xy2ind(*model.prd_xy.T)


def obs_model(water_sat):
    return water_sat[prod_inds]


# #### Simulation
# The following generates the synthetic truth evolution and data.

T = 1
dt = 0.025
nTime = round(T / dt)

wsat0 = np.zeros(model.Nxy)
wsat.past.Truth = model.sim(dt, nTime, wsat0)
prod.past.Truth = np.array([obs_model(x) for x in wsat.past.Truth[1:]])

# #### Animation
# Run the code cells below to get an animation of the oil saturation evolution.
# Injection/production wells are marked with triangles pointing down/up.
# The (untransformed) pre-perm field is plotted, rather than the actual permeability.

# %%capture
# animation = model.anim(wsat.past.Truth, prod.past.Truth);  # fmt: skip

# Note: can take up to a minute to appear
# animation

# #### Noisy obs
# In reality, observations are never perfect. To emulate this, we corrupt the
# observations by adding a bit of noise.

prod.past.Noisy = prod.past.Truth.copy()
R = 1e-3 * np.eye(nPrd)
for iT in range(nTime):
    prod.past.Noisy[iT] += sqrt(R) @ rnd.randn(nPrd)


# Plot of observations (and their noise):

# fig, ax = plotting.freshfig("Observations")
# model.plt_production(ax, prod.past.Truth, prod.past.Noisy);  # fmt: skip

# Note that several observations are above 1,
# which is "unphysical" or not physically "realisable".

# ## Prior

# The prior ensemble is generated in the same manner as the (synthetic) truth, using the
# same mean and covariance.  Thus, the members are "statistically indistinguishable" to
# the truth. This assumption underlies ensemble methods.
#
# In practice, "encoding" prior information, from a range of experts, and prior
# information sources, in such a way that it is useful for decision making analyses (and
# history matching), is a formidable tecnnical task, typically involving multiple
# different types of modelling.  Nevertheless it is crucial, and must be performed with
# care.

N = 200
perm.Prior = sample_prior_perm(N)

# # Note that field (before transformation) is Gaussian with (expected) mean 0 and variance 1.
# print("Prior mean:", np.mean(perm.Prior))
# print("Prior var.:", np.var(perm.Prior))

# #### Histogram
# Let us inspect the parameter values in the form of their histogram.
# Note that the histogram of the truth is simply counting the values of a single field,
# whereas the histogram of the ensemble counts the values of `N` fields.

# fig, ax = plotting.freshfig("Perm. distribution", figsize=(7, 3))
# bins = np.linspace(*plotting.styles["pperm"]["levels"][[0, -1]], 32)
# for label, perm_field in perm.items():
#     x = perm_field.ravel()
#     ax.hist(
#         perm_transf(x),
#         perm_transf(bins),
#         # Divide counts by N to emulate `density=1` for log-scale.
#         weights=(np.ones_like(x) / N if label != "Truth" else None),
#         label=label,
#         alpha=0.3,
#     )
# ax.set(xscale="log", xlabel="Permeability", ylabel="Count")
# ax.legend()
# fig.tight_layout()
# plt.show()

# Since the x-scale is logarithmic, the prior's histogram should look Gaussian if
# `perm_transf` is purely exponential. By contrast, the historgram of the truth is from
# a single (spatially extensive) realisation, and therefore will contain significant
# sampling "error".

# #### Field plots
# Below we can see some (pre-perm) realizations (members) from the ensemble.

# plotting.fields(model, perm.Prior, "pperm", "Prior");  # fmt: skip

# #### Variance/Spectrum
# In practice we would not be using an explicit `Cov` matrix when generating
# the prior ensemble, because it would be too large. However, we could inspect the spectrum.

# U, svals, VT = sla.svd(perm.Prior)
# plotting.spectrum(svals, "Prior cov.");  # fmt: skip

# ## Forward model

# In order to (begin to attempt to) solve the *inverse problem*,
# we first have to be able to solve the *forward problem*.
# Indeed, ensemble methods obtain observation-parameter sensitivities
# from the covariances of the ensemble run through the ("forward") model.

# #### Function composition

# The forward model is generally a composite function.
# In our simple case, it only consists of two steps:
#
# - Setting the permeability *parameter* field.
#   Note that setting parameters is not generally as trivial a task as it is here.
#   It might involve reshaping arrays, translating units, read/write to file, etc.
#   Indeed, from a "task runner" perspective, there is no hard distinction between
#   writing parameters and running simulations.
# - Running the reservoir simulator.
#
# Stitching together a composite model is not usually a pleasant task.
# Some tools like [ERT](https://github.com/equinor/ert) can make it a little easier.


def comp1(perm, wsat0=wsat0):
    """Composite forward/forecast/prediction model (for 1 realisation)."""
    new_model = copy.deepcopy(model)  # don't overwrite truth model
    set_perm(new_model, perm)  # set parameters
    wsats = new_model.sim(dt, nTime, wsat0, pbar=False)  # run simulator
    prods = np.array([obs_model(x) for x in wsats[1:]])  # extract prod time series
    return wsats, prods


# Note that the input to `forward_model` can contain **not only** permeability fields,
# but **also** the state (i.e. time-dependent, prognostic) variable, i.e. water saturations.
# Why? Because further below we'll be "restarting" (running) the simulator
# from a later point in time (to generate future predictions) in which case
# the saturation fields (which is also among the outputs) will depend on the
# given permeability field, and hence vary from realisation to realisation.
# Thus, the state need only be outputted at the *final* time (for future prediction),
# but for diagnostic purposes we emit the full time series of wsats.

# #### Parallelize

# A huge technical advantage of ensemble methods is that they are
# "embarrasingly parallelizable", because each member simulation
# is completely independent (requires no communication) from the others.


def forward_model(*args, leave=True, desc="Ens-run", **kwargs):
    """Parallelize forward model (`comp1`)."""
    pbar = dict(leave=leave, desc=desc)
    output = apply(comp1, *args, pbar=pbar, **kwargs)
    return [np.asarray(y) for y in zip(*output)]


# Configure the number of CPUs to use in `apply`. Can set to an `int` or False.

utils.nCPU = "auto"

# #### Run

# Now that our forward model is ready, we can make prior estimates of the saturation
# evolution and production. This is interesting in and of itself and, as we'll see
# later, is part of the assimilation process.

(wsat.past.Prior,
 prod.past.Prior) = forward_model(perm.Prior)  # fmt: skip

# #### Flattening the time dimension

# We have organised our ensemble data in 3D arrays,
# with *time* along the second-to-last axis.
# Ensemble methods have no notion of 3D arrays, however, so we need to
# be able to flatten the time and space dimensions (as well as to undo this).
# Providing we stick to a given array axis ordering,
# here is a convenient function for juggling the array shapes.


def vect(x, undo=False):
    """Unravel/flatten the last two axes. Assumes axis `-2` has length `nTime`."""
    # Works both for ensemble (3D) and single-realisation (2D) arrays.
    if undo:
        *N, ab = x.shape
        return x.reshape(N + [nTime, ab // nTime])
    else:
        *N, a, b = x.shape
        return x.reshape(N + [a * b])


# Similarly, we need to specify the observation error covariance matrix for the
# flattened observations.

augmented_obs_error_cov = sla.block_diag(*[R] * nTime)

# ## Correlation study (*a-priori*)

# #### The mechanics of the Kalman gain

# The conditioning "update" of ensemble methods is often formulated in terms of a
# "**Kalman gain**" matrix, derived so as to achieve a variety of optimality properties
# (see e.g. [[Jaz70]](#Jaz70)):
# - in the linear-Gaussian case, to compute the correct posterior moments;
# - in the linear (not-necessarily-Gaussian) case, to compute the [BLUE/MMSE](https://en.wikipedia.org/wiki/Kalman_filter#Kalman_gain_derivation),
#   which is akin to achieving orthogonality of the posterior error and innovation;
# - in the non-linear, non-Gaussian case, the *ensemble* Kalman gain can be derived as
#   linear regression (with some tweaks) from the noisy obs. to the unknowns.
#
# Another way to look at it is to ask "what does it do?"
# Heuristically, this may be answered as follows:
#
# - It uses correlation coefficients to establish relationships between
#   observations and unknowns. For example, if there is no correlation,
#   the unknowns do not get updated.
# - It takes into account the "intermingling" of correlations. For example, two
#   measurements/observations that are highly correlated
#   (when including both prior uncertainty and observation noise)
#   will barely have more impact than either one alone.
# - It also takes into account the variables' variance,
#   and thereby the their relative uncertainties
#   (in fact linear least-squares regression is a straighforward
#   combination of variances and correlation coefficients).
#   For example, if two variables have equal correlation with an observation,
#   but one is more uncertain, that one will receive a larger update than the other.
#   Conversely, an observation with a larger variance will have less impact than an
#   observation with a smaller variance. Working with variances also means that
#   the physical units of the variables are inherently accounted for.
#
# In summary, it is useful to investigate the correlation relations of the ensemble,
# especially for the prior.

# #### Exploratory correlation plot

# The following plots a variety of different correlation fields. Each field may
# be seen as a single column (or row) of a larger ("cross")-covariance matrix,
# which would typically be too large for explicit computation or storage. The
# following solution, though, which computes the correlation fields "on the
# fly", should be viable for relatively large scales.

# +
# Available variable types
prior_fields = {
    "Saturation": lambda time: wsat.past.Prior[:, time],
    "Pre-perm"  : lambda _time: perm.Prior,
    "Perm"      : lambda _time: perm_transf(perm.Prior),
}  # fmt: skip


# Compute correlation field
def corr_comp(N, Field, T, Point, t, x, y):
    xy = model.sub2ind(x, y)
    Point = prior_fields[Point](t)[:, xy]
    Field = prior_fields[Field](T)
    return utils.corr(Field[:N], Point[:N])


# Register controls
corr_comp.controls = dict(
    N=(2, N),
    Field=list(prior_fields),
    T=(0, nTime),
    Point=list(prior_fields),
    t=(0, nTime),
    x=(0, model.Nx - 1),
    y=(0, model.Ny - 1),
)
# -

# plotting.field_console(model, corr_comp, "corr", "Prior", argmax=True, wells=True)

# Use the interative control widgets to investigate the correlation structure.
# Answer the following questions. *NB*: the order matters!
#
# - Set the times as `T = t = 20`
#     and the variable kinds as `Field = Point = "Saturation"`.
#   - Move the point around (`x` and `y` sliders).
#   - Why is the star marker (showing the location of the maximum)
#     on top of the crosshairs?
#     <!-- Answer: Because the correlation the correlation of a variable with itself
#     is 1.00, which is the maximum possible correlation. This is a useful sanity check
#     on our correlation and plotting facilities. Use the zoom functionality if necessary
#     to assert exact superposition.
#     -->
# - Set `Field = "Pre-perm"`.
#   - Move `T` around. Why doesn't anything change?
#   - Set the ensemble size: `N=2`. How does the correlation field look? Why?
#     <!-- Answer: Only 2 colors, because 2 points always lie on a straight line -->
# - Now set `Field = "Saturation"`. Explain the major new and strange appearance.
#   <!-- Answer: Nan's and inf's at corners. Reason: for most realisations,
#   the saturation is (as of yet) constant there.
#   -->
# - Set `N=200`. Move the point to the center again.
#   - Set `T=0` How do the correlation fields look? Why?
#   - Set `t=T=1`. Gradually move `T=2,3,4, etc` (hint: use your arrow keys).
#     Explain the appearance of "fronts".
#   - Move `T=1,2,3, etc` using your arrow keys. Explain the appearance of "fronts".
# - Set `T=20`, `t=40`, and move the point to the location of one of the wells.
#   - Where is the maximum? And minimum? Does this make sense?
#   - Gradually increase `T`. How do the extrema move? Why?.
# - Set `T=40`. Note the location of the maximum. Now switch to `Field = "Per-perm"`.
#   Note that we are now investigate the correlation between
#   the unknowns and the observations.
#   - Where is the maximum now? Does it make sense?
#   - Gradually decrease `t` back down to `10`.
#     Describe and explain the change in the correlation field.
#     <!-- Answer: it weakens, but does not move a lot.
#     It weakens because the early production (saturation) is 100% anyway,
#     thus independent of the permeability fields.
#     -->
#   - Set `t=40` again. Explain the appearance of negative correlations on the opposite
#     side of where the point (`x` and `y`) is located.
#     <!-- Answer: The negative correlations arise because a low permeability on the
#     other side will make the injector pump more water in the direction of the point.
#     -->
# - Set `t=30` and `N=2`, then (use your arrow keys to) gradually increase `N`
#   to `20`. Do the (changes you observe in the) correlation fields inspire confidence?
#   Actually, that's a rhetorical question; the answer is clearly no.
# - Now try flipping between low and high values of `N`.
#   What do you think the tapering radius should be?

# ## Assimilation

# ### Basic ensemble conditioning

# Denote $\mathbf{E}$ the ensemble matrix (whose columns are a sample from the prior),
# $\mathcal{M}(\mathbf{E})$ the observed ensemble,
# and $\mathbf{D}$ be the observation perturbations.
# Let $\mathbf{X}$ and $\mathbf{Y}$ be the ensemble and the observed ensemble, respectively,
# but now with their (ensemble-) mean subtracted.
# Then the ensemble update can be written
#
# $$ \mathbf{E}^a
# = \mathbf{E}
# + \mathbf{X} \mathbf{Y}^T
# \big( \mathbf{Y} \mathbf{Y}^T + (N{-}1) \mathbf{R} \big)^{-1}
# \big\{ \mathbf{y} \mathbf{1}^T - [\mathcal{M}(\mathbf{E}) + \mathbf{D}] \big\} $$


def ens_update0(ens, obs_ens, obs, perturbs, obs_err_cov):
    """Compute the ensemble analysis (conditioning/Bayes) update."""
    X, _ = center(ens)
    Y, _ = center(obs_ens)
    perturbs, _ = center(perturbs, rescale=True)
    obs_cov = obs_err_cov * (len(Y) - 1) + Y.T @ Y
    obs_pert = perturbs @ sqrt(obs_err_cov)  # TODO: sqrtm if R non-diag
    innovations = obs - (obs_ens + obs_pert)
    KG = sla.pinv(obs_cov) @ Y.T @ X
    return ens + innovations @ KG


# Notes:
#  - The formulae used by the code are transposed and reversed compared to the above.
#    [Rationale](https://nansencenter.github.io/DAPPER/dev_guide.html#conventions)
#  - The perturbations are *input arguments* because we will want to re-use the same ones
#    when doing localization. It also enables exact reproducibility (see sanity check below).

# #### Bug check

# It is very easy to introduce bugs.
# Fortunately, most can be eliminated with a few simple tests.
#
# For example, let us generate a case where both the unknown, $\mathbf{x}$,
# and the observation error are (independently) $\mathcal{N}(\mathbf{0}, 2 \mathbf{I})$,
# while the forward model is just the identity.

# Note: the prefix "gg_" stands for Gaussian-Gaussian
gg_ndim = 3
gg_prior = sqrt(2) * rnd.randn(1000, gg_ndim)

# From theory, we know the posterior, $\mathbf{x}|\mathbf{y} \sim \mathcal{N}(\mathbf{y}/2, 1\mathbf{I})$.
# Let us verify that the ensemble update computes this (up to sampling error)

gg_kwargs = dict(
    ens=gg_prior,
    obs_ens=gg_prior,
    obs=10 * np.ones(gg_ndim),
    obs_err_cov=2 * np.eye(gg_ndim),
    perturbs=rnd.randn(*gg_prior.shape),
)
gg_postr = ens_update0(**gg_kwargs)

with np.printoptions(precision=2, suppress=True):
    print("Posterior mean:", np.mean(gg_postr, 0))
    print("Posterior cov:", np.cov(gg_postr.T), sep="\n")

# #### Why smoothing (and not filtering)?
# Before ensemble smoothers were used for history matching, it was though that
# *filtering*, rather than *smoothing*, should be used. As opposed to the (batch)
# ensemble smoothers, filters *sequentially* assimilate the time-series data,
# updating/conditioning both the saturation (i.e. state) fields and the permeability
# (i.e. parameter) fields. Some people might also call this a "sequential" smoother. In
# any case, this is problematic because the ensemble update is approximate, which not
# only causes statistical suboptimality, but also "un-physical" or "non-realisable"
# members -- a problem that gets exasperated by the simulator (manifesting as slow-down
# or crash, often due to convergence problems in the linear solver).  Moreover, the
# approximation (and hence the associated problems) only seem likely to worsen if using
# jointly-updated (rather than re-generated) state fields. This makes the
# parameter-only update of the (batch) smoothers appealing.
# Furthermore, the predominant uncertainty in history matching problems usually originates
# in the prior, rather than model error, reducing the potential for improvement by filtering.
# Finally, it is easier to formulate an iterative smoother than an iterative filter.

# #### Apply

# Our vector of unknowns is the pre-permeability.
# However, further below we will also apply the update to other unknowns
# (future saturation or productions). For brevity, we therefore collect the
# arguments that are common to all of the applications of this update.
#
# *PS: we could also pre-compute the matrices of the update that are common to
# all updates, thus saving time later. The fact that this is a possibility will
# not come as a surprise to readers familiar with state-vector augmentation.*

kwargs0 = dict(
    obs_ens=vect(prod.past.Prior),
    obs=vect(prod.past.Noisy),
    perturbs=rnd.randn(N, nPrd * nTime),
    obs_err_cov=augmented_obs_error_cov,
)

# Thus the update is called as follows

perm.ES = ens_update0(perm.Prior, **kwargs0)

# #### Field plots

# Let's plot the updated ensemble.

# plotting.fields(model, perm.ES, "pperm", "ES (posterior)");  # fmt: skip

# We will see some more diagnostics later.

# ### Localization


def ens_update0_loc(ens, obs_ens, obs, perturbs, obs_err_cov, domains, taper):
    """Perform local analysis/domain updates using `ens_update0`."""

    def local_analysis(ii):
        """Update for domain/batch `ii`."""
        # Get localization mask, coeffs
        oBatch, tapering = taper(ii)
        # Convert [range, slice, epsilon] to inds (for np.ix_)
        oBatch = np.arange(len(obs))[oBatch]
        # Update
        if len(oBatch) == 0:
            # no obs ==> no update
            return ens[:, ii]
        else:
            c = sqrt(tapering)
            return ens_update0(
                ens[:, ii],
                obs_ens[:, oBatch] * c,
                obs[oBatch] * c,
                perturbs[:, oBatch] * c,
                obs_err_cov[np.ix_(oBatch, oBatch)],
            )

    # Run -- could use multiprocessing here (replace `map` by `mp`),
    # but in our case the overhead means that it's not worth it.
    EE = map(local_analysis, domains)

    # Write to ensemble matrix. NB: don't re-use `ens`!
    Ea = np.empty_like(ens)
    for ii, Eii in zip(domains, EE):
        Ea[:, ii] = Eii

    return Ea


# The form of the localization used in the above code is "local/domain analysis".
# Note that it sequentially processing batches (subsets/domains)
# of the vector of unknowns (actually, ideally, we'd iterate over each single element,
# but that is usually computationally inefficient).
#
# The localisation setup (`taper`) must return a mask or list of indices
# that select the observations near the local domain `ii`,
# and the corresponding tapering coefficients.
# For example, consider this setup,
# which makes the update process each local domain entirely independently,
# *assuming an identity forward model, i.e. that `obs := prm + noise`*.


def full_localization(batch_inds):
    return batch_inds, 1


# #### Bug check

# Again, the (localized) method should yield the correct posterior,
# up to some sampling error. However, thanks to `full_localization`,
# this error should be smaller than in our bug check for `ens_update0`.

gg_postr = ens_update0_loc(**gg_kwargs, domains=np.c_[:gg_ndim], taper=full_localization)

with np.printoptions(precision=2, suppress=True):
    print("Posterior mean:", np.mean(gg_postr, 0))
    print("Posterior cov:", np.cov(gg_postr.T), sep="\n")

# #### Sanity check

# Now consider the following setup.


def no_localization(_batch_inds):
    return ..., 1  # ellipsis (...) means "all"


# Hopefully, using this should output the same ensemble (up to *numerical* error)
# as `ens_update0`. Let us verify this:

tmp = ens_update0_loc(perm.Prior, **kwargs0, domains=[...], taper=no_localization)
print("Reproduces global analysis?", np.allclose(tmp, perm.ES))

# *PS: with no localization, it should not matter how the domain is partitioned.
# For example, try `domains=np.arange(model.Nxy).reshape(some_integer, -1)`.*

# #### Configuration for the history matching problem

# Now let us define the local domains for the permeability field.

domains = loc.rectangular_partitioning(model.shape, (2, 3))

# We can illustrate the partitioning by filling each domain by a random color.
# This should produce a patchwork of rectangles.

# +
colors = rnd.choice(len(domains), len(domains), False)
Z = np.zeros(model.shape)
for d, c in zip(domains, colors):
    Z[tuple(model.ind2sub(d))] = c

# fig, ax = plotting.freshfig("Computing domains", figsize=(6, 3))
# ax.imshow(Z, cmap="tab20", aspect=0.5)
# fig.tight_layout()
# plt.show()


# -

# The tapering will be a function of the batch's mean distance to the observations.
# The default `radius` and `sharpness` are the ones we found to be the most
# promising from the above correlation study.


def localization_setup(batch, radius=0.8, sharpness=1):
    dists = distances_to_obs[batch].mean(axis=0)
    obs_coeffs = loc.bump_function(dists / radius, sharpness)
    obs_mask = obs_coeffs > 1e-3
    return obs_mask, obs_coeffs[obs_mask]


# #### Localization tuning

# It is technically challenging to translate/encode **all** of our prior knowledge into
# the computational form of an ensemble.  It is also computationally demanding, because
# a finite ensemble size, $N$, will contain sampling errors.  Thus, in principle, there
# is room to improve the performance of the ensemble methods by "injecting" more prior
# knowledge somehow, as an "auxiliary" technique.  A particularly effective way is
# **localization**, wherein we eliminate correlations (i.e. relationships) that we are
# "pretty sure" are *spurious*: merely due to sampling error, rather than indicative of
# an actual inter-dependence.
#
# Much can be said about the ad-hoc nature of most localization schemes.
# This is out of scope here.
# Furthermore, in particular in petroleum reservoir applications,
# configuring an effective localization setup can be very challenging.
# If successful, however, localization is unreasonably effective,
# allowing the use of much smaller ensemble sizes than one would think.
#
# In our simple case, it is sufficient to use distance-based localization.
# Far-away (remote) correlations will be dampened ("tapered").
# For the shape, we here use the "bump function" rather than the
# conventional (but unnecessarily complicated) "Gaspari-Cohn" piecewise polyomial
# function.  It is illustrated here.

# fig, ax = plotting.freshfig("Tapering ('bump') functions")
# dists = np.linspace(-1, 1, 1001)
# for sharpness in [0.01, 0.1, 1, 10, 100, 1000]:
#     coeffs = loc.bump_function(dists, sharpness)
#     ax.plot(dists, coeffs, label=sharpness)
# ax.legend(title="sharpness")
# ax.set_xlabel("Distance")
# fig.tight_layout()
# plt.show()

# We will also need the distances, which we can pre-compute.
# As seen from `distances_to_obs` below, we will need the
# locations of each observation and each unknown parameter.

xy_obs = model.ind2xy(prod_inds)
xy_prm = model.ind2xy(np.arange(model.Nxy))

# However, as we saw from the correlation dashboard, the localization should be
# time dependent. For example, it is tempting to say that remote-in-time (i.e. late)
# observations should have a larger area of impact,
# since they are integro-spatio-temperal functions (to use a fancy word) of the perm fields.
# We could achieve that by adding a time coordinate to `xy_obs` (setting it to 0 for `xy_prm`).
# However, the correlation dashboard does not really support this "dilation" theory,
# and we should be careful about growing the tapering mask.
# So instead, we simply replicate the same locations for each time instance.

xy_obs = np.tile(xy_obs, nTime)

# Now we compute the distance between the parameters and the (argmax of the
# correlations with the) observations.

distances_to_obs = loc.pairwise_distances(xy_prm.T, xy_obs.T)

# The tapering function is similar to the covariance functions used in
# geostatistics (see Kriging, variograms), and indeed localization can be
# framed as a hybridisation of ensemble covariances with theoretical ones.
# However, the ideal tapering function does not generally equal the theoretical
# covariance function, but must instead be "tuned" for performance in the
# history match. Here we shall content ourselves simply with tuning a "radius"
# parameter.  Neverthless, tuning (wrt. history matching performance) is a
# breathtakingly costly proposition, requiring a great many synthetic
# experiments. This is made all the worse by the fact that it might have to be
# revisited later after some other factors have been tuned, or otherwise
# changed.
#
# Therefore, in lieu of such global tuning, we here undertake a study of the
# direct impact of the localization on the correlation fields. Fortunately, we
# can mostly just re-use the functionality from the above correlation
# dashboard, but now with some different controls; take a moment to study the
# function below, which generates the folowing plotted data.


def corr_wells(N, t, well, localize, radi, sharp):
    t = t - 1
    if not localize:
        N = -1
    C = utils.corr(perm.Prior[:N], prod.past.Prior[:N, t, well])
    if localize:
        dists = distances_to_obs[:, well + nPrd * t]
        c = loc.bump_function(dists / radi, 10**sharp)
        C *= c
        C[c < 1e-3] = np.nan
    return C


corr_wells.controls = dict(
    localize=False,
    radi=(0.1, 5),
    sharp=(-1.0, 1),
    N=(2, N),
    t=(1, nTime),
    well=np.arange(nPrd),
)


# plotting.field_console(model, corr_wells, "corr", "Prior pre-perm to well observation", wells=True)


# - Note that the `N` slider is only active when `localize` is *enabled*.
#   When localization is not enabled, then the full ensemble size is being used.
# - Set `N=20` and toggle `localize` on/off, while you play with different values of `radi`.
#   Try to find a value that makes the `localized` (small-ensemble) fields
#   resemble (as much as possible) the full-size ensemble fields.
# - The suggested value from the author is `0.8` (and sharpness $10^0$, i.e. 1).

# #### Time-dependent localisation
# In the preceding dashboards we could observe that the "locations" (defined as the
# location of the maximum) of the correlations (between a given well observation
# and the permeability field) moved in time. Let us trace these paths computationally.

xy_max_corr = np.zeros((nPrd, nTime, 2))
for i, xy_path in enumerate(xy_max_corr):
    for time in range(6, nTime):
        C = utils.corr(perm.Prior, prod.past.Prior[:, time, i])
        xy_path[time] = model.ind2xy(np.argmax(C))

# In general, minima might be just as relevant as maxima.
# In our case, though, it's a safe bet to focus on the maxima, which also avoids
# the danger of jumping from one case to another in case of weak correlations.
#
# For `time<6`, there is almost zero correlation anywhere,
# so we should not trust `argmax`. Fallback to `time=6`.

xy_max_corr[:, :6] = xy_max_corr[:, [6]]

# Here is a plot of the paths.

# fig, ax = plotting.freshfig("Trajectories of maxima of corr. fields")
# for i, xy_path in enumerate(xy_max_corr):
#     color = dict(color=f"C{1+i}")
#     ax.plot(*xy_path.T, "-o", **color)
#     ax.plot(*xy_path[-1], "s", ms=8, **color)  # start
# model.plt_field(ax, np.zeros(model.shape), "default", colorbar=False);  # fmt: skip

# An intriguing possibility is to co-locate the correlation masks with the path of the correlation maxima,
# rather than centering them on the wells directly. However, this is very experimental, and is disabled by default.

# +
# xy_obs = vect(xy_max_corr.T)
# distances_to_obs = loc.pairwise_distances(xy_prm.T, xy_obs.T)
# -

# #### Apply

perm.LES = ens_update0_loc(perm.Prior, **kwargs0, domains=domains, taper=localization_setup)

# Again, we plot some updated/posterior fields

# plotting.fields(model, perm.LES, "pperm", "LES (posterior)");  # fmt: skip

# ### Iterative smoother

# #### Why iterate?
# Due to non-linearity of the forward model, the likelihood is non-Gaussian, and
# ensemble methods do not compute the true posterior (even with infinite `N`).  Still,
# after the update, it may be expected that the estimate of the sensitivity (of the
# model to the observations) has improved. Thus, it makes sense to retry the update
# (starting from the prior again, so as not to over-condition/use the data), but this
# time with the improved sensitivity estimate.  This cycle can then be repeated
# indefinitely.
#
# Caution: the meaning of "improvement" of the sensitivity estimate is not well defined.
# It is known that ensemble sensitivities estimate the *average* sensitivities
# ([[Raa19]](#Raa19)); however, it does not seem possible to prove (with generality)
# that the average defined by the (iteratively approximated) posterior is better suited
# than that of the prior, as neither one will yield the correct posterior.
# Nevertheless, when accompanied by sufficient hand-waving, most people will feel
# convinced by the above argument, or something similar.
#
# Another perspective is that the iterations *might* manage to find the
# mode of the posterior, i.e. perform maximum-a-posteriori (MAP) estimation.
# This perspective comes from weather forecasting and their "variational"
# methods, as well as classical (extended, iterative) Kalman filtering.
# However, this perspective is more of a first-order approximation
# to the fully Bayesian uncertainty quantification approximated by ensemble methods.
#
# In any case, empricial evidence leave little room to doubt that iterations
# yield improved estiamtion accuracy, albeit a the cost of (linearly) more
# computational effort.


# fmt: off
def IES_analysis(w, T, Y, dy):
    """Compute the ensemble analysis."""
    N         = len(w)
    Y0        = sla.pinv(T) @ Y                          # "De-condition"
    nExs      = Y0.shape[0] - Y0.shape[1]                # nEns - len(y), i.e. "Excess N"
    V, s, _UT = sla.svd(Y0, full_matrices = (nExs > 0))  # Decompose
    cow1s     = N - 1 + np.pad(s**2, (0, max(0, nExs)))  # Postr. cov_w^{-1} spectrum
    cowp      = lambda p: (V * cow1s**p) @ V.T           # Postr. cov_w^{-p}
    grad      = Y0 @ dy - w * (N - 1)                    # Cost function gradient
    dw        = grad @ cowp(-1.0)                        # Gauss-Newton step
    T         = cowp(-0.5) * sqrt(N - 1)                 # Transform matrix
    return dw, T
# fmt: on


def IES(ens, obs, obs_err_cov, stepsize=1, nIter=10, wtol=1e-4):
    """Iterative ensemble smoother."""
    E = ens
    y = obs
    N = len(E)
    N1 = N - 1
    Rm12T = np.diag(sqrt(1 / np.diag(obs_err_cov)))  # TODO?

    # Init
    stat = Dict(dw=[], rmse=[], stepsize=[], obj=Dict(lklhd=[], prior=[], postr=[]))

    # Init ensemble decomposition.
    X0, x0 = center(E)  # Decompose ensemble.
    w = np.zeros(N)  # Control vector for the mean state.
    T = np.eye(N)  # Anomalies transform matrix.

    for itr in utils.progbar(range(nIter), desc="Iter.ES"):
        # Compute rmse (vs. supposedly unknown Truth)
        # err = E.mean(0) - perm.Truth
        # stat.rmse += [np.sqrt(np.mean(err * err))]  # == norm / sqrt(len)

        # Forecast.
        _, Eo = forward_model(E, leave=False)
        Eo = vect(Eo)

        # Prepare analysis.
        Y, xo = center(Eo)  # Get anomalies, mean.
        dy = (y - xo) @ Rm12T  # Transform obs space.
        Y = Y @ Rm12T  # Transform obs space.

        # Diagnostics
        stat.obj.prior += [w @ w * N1]
        stat.obj.lklhd += [dy @ dy]
        stat.obj.postr += [stat.obj.prior[-1] + stat.obj.lklhd[-1]]

        reject_step = itr > 0 and stat.obj.postr[itr] > np.min(stat.obj.postr)
        if reject_step:
            # Restore prev. ensemble, lower stepsize
            stepsize /= 10
            w, T = old  # noqa: F821
        else:
            # Store current ensemble, boost stepsize
            old = w, T
            stepsize *= 2
            stepsize = min(1, stepsize)

            dw, T = IES_analysis(w, T, Y, dy)

        stat.dw += [dw @ dw / N]
        stat.stepsize += [stepsize]

        # Step
        w = w + stepsize * dw
        E = x0 + (w + T) @ X0

        if stepsize * np.sqrt(dw @ dw / N) < wtol:
            break

    # The last step must be discarded,
    # because it cannot be validated without re-running the model.
    w, T = old
    E = x0 + (w + T) @ X0

    return E, stat


# #### Bug check

_tmp = forward_model
forward_model = lambda x, **kwargs: (None, np.expand_dims(x, -2))
gg_postr = IES(gg_kwargs["ens"], gg_kwargs["obs"], gg_kwargs["obs_err_cov"])[0]
forward_model = _tmp

with np.printoptions(precision=2, suppress=True):
    print("Posterior mean:", np.mean(gg_postr, 0))
    print("Posterior cov:", np.cov(gg_postr.T), sep="\n")


# #### Compute

kwargsI = dict(
    obs=vect(prod.past.Noisy),
    obs_err_cov=augmented_obs_error_cov,
)

perm.IES, diagnostics = IES(perm.Prior, **kwargsI, stepsize=1)

# #### Field plots
# Let's plot the updated, initial ensemble.

# plotting.fields(model, perm.IES, "pperm", "IES (posterior)");  # fmt: skip

# The following plots the cost function(s) together with the error compared to the true
# (pre-)perm field as a function of the iteration number. Note that the relationship
# between the (total, i.e. posterior) cost function  and the RMSE is not necessarily
# monotonic. Re-running the experiments with a different seed is instructive. It may be
# observed that the iterations are not always very successful.

# fig, ax = plotting.freshfig("IES Objective function")
# ls = dict(postr="-", prior=":", lklhd="--")
# for name, J in diagnostics.obj.items():
#     ax.plot(np.sqrt(J), color="b", ls=ls[name], label=name)
# ax.set_xlabel("iteration")
# ax.set_ylabel("RMS mismatch", color="b")
# ax.tick_params(axis="y", labelcolor="b")
# ax.legend()
# ax2 = ax.twinx()  # axis for rmse
# ax2.set_ylabel("RMS error", color="r")
# ax2.plot(diagnostics.rmse, color="r")
# ax2.tick_params(axis="y", labelcolor="r")
# fig.tight_layout()
# plt.show()

# ## Diagnostics

# In terms of root-mean-square error (RMSE), the ES is expected to improve on the prior.
# The "expectation" wording indicates that this is true on average, but not always. To
# be specific, it means that it is guaranteed to hold true if the RMSE is calculated for
# infinitely many experiments (each time simulating a new synthetic truth and
# observations from the prior). The reason for this is that the ES uses the Kalman
# update, which is the BLUE (best linear unbiased estimate), and "best" means that the
# variance must get reduced. However, note that this requires the ensemble to be
# infinitely big, which it most certainly is not in our case. Therefore, we do not need
# to be very unlucky to observe that the RMSE has actually increased. Despite this, as
# we will see later, the data match might yield a different conclusions concerning the
# utility of the update.

# ### Wrt. True field

# #### RMS summary
# RMS stands for "root-mean-square(d)" and is a summary measure for deviations.
# With ensemble methods, it is (typically, and in this case study) applied
# to the deviation from the **ensemble mean**, whence the trailing `M` in `print_RMSMs` below.

print(f"Accuracy wrt. unknown {emph('parameter')} field\n")
utils.print_RMSMs(perm, ref="Truth")

# #### Field plots
# Let's plot mean fields.
#
# NB: Caution! Mean fields are liable to smoother than the truth. This is a phenomenon
# familiar from geostatistics (e.g. Kriging). As such, their importance must not be
# overstated (they're just one estimator out of many). Instead, whenever a decision is
# to be made, all of the members should be included in the decision-making process. This
# does not mean that you must eyeball each field, but that decision analyses should be
# based on expected values with respect to ensembles.

perm_means = Dict({k: perm[k].mean(axis=0) for k in perm})

plotting.fields(model, perm_means, "pperm", "Means");  # fmt: skip

# ### Data mismatch (past production)
# In synthetic experiments such as this one, is is instructive to computing the "error":
# the difference/mismatch of the (supposedly) unknown parameters and the truth.  Of
# course, in real life, the truth is not known.  Moreover, at the end of the day, we
# mainly care about production rates and saturations.

# #### Re-run
# Therefore, let us now compute the "residual" (i.e. the mismatch between
# predicted and true *observations*), which we get from the predicted
# production "profiles".

for methd in perm:
    if methd not in prod.past:
        (wsat.past[methd],
         prod.past[methd]) = forward_model(perm[methd], desc=methd)  # fmt: skip

# The ES can be applied to any un-conditioned ensemble (not just the permeabilities).
# A particularly interesting case is applying it to the prior's production predictions.
# This provides another posterior approximation of the production history
# -- one which doesn't require running the model again
# (in contrast to what we did for `prod.past.(I)ES` immediately above).
# The approach is sometimes called data-space inversion.
# Since it requires 0 iterations, let's call this "ES0". Let us try that as well.

prod.past.ES0 = vect(ens_update0(vect(prod.past.Prior), **kwargs0), undo=True)

# #### Production plots

plotting.productions(prod.past, "Past");  # fmt: skip

# ##### Comment on prior
# Note that the prior "surrounds" the data. This the likely situation in our synthetic
# case, where the truth was generated by the same random draw process as the ensemble.
#
# In practice, this is often not the case. If so, you might want to go back to your
# geologists and tell them that something is amiss. You should then produce a revised
# prior with better properties.
#
# Note: the above instructions sound like statistical heresy. We are using the data
# twice over (on the prior, and later to update/condition the prior). However, this is
# justified to the extent that prior information is difficult to quantify and encode.
# Too much prior adaptation, however, and you risk overfitting! Indeed, it is a delicate
# matter. It is likely best resolved by only revising coarse features of the prior,
# and increasing its uncertainty rather than trying to adjust its mean (bias).

# ##### Comment on posterior
# If the assumptions (statistical indistinguishability, Gaussianity) are not too far
# off, then the ensemble posteriors (`ES`, `LES`, `IES`, `ES0`)
# should also surround the data, but with a tighter fit.

# #### RMS summary

print(f"Accuracy wrt. {emph('past')} production (i.e. actual, noisy, obs. data)\n")
utils.print_RMSMs(prod.past, ref="Noisy")

# Note that, here, the "err" is comptuted vs. the observations,
# not the (supposedly unknown) truth. In any case,
# the `rms err` for any of the (approximate) posteriors should be lower than
# the `rms err` of the `Prior`, although for very small $N$ this may (spuriously) fail.
# Moreover, in the linear-Gaussian, infinite-$N$ case, the `rms err` of the posteriors
# should also be lower than that of the `Truth`
# (whose `rms err` approximates the std. dev. of the obs. noise).
# Evidently these assumptions are far from valid,
# since none of the approximate posteriors (i.e. methods) achieve this
# low magnitude of error.
#
# Note that the error of `ES0` is very low. As we shall see, however,
# this "method" is very poor at prediction (in this nonlinear case).

# ## Prediction

# We now prediction the future by forecasting from the current (present-time) ensembles.
#
# Note that we must use the current saturation in the "restart" for the predictive
# simulations. Since the estimates of the current saturation depend on the assumed
# permeability field, these estimates are also "posterior", and depend on the
# conditioning method used. For convenience, we first extract the slice of the current
# saturation fields (which is really the only one we make use of among those of the
# past), and plot the mean fields.

wsat.curnt = Dict({k: v[..., -1, :] for k, v in wsat.past.items()})
wsat_means = Dict({k: np.atleast_2d(v).mean(axis=0) for k, v in wsat.curnt.items()})
plotting.fields(model, wsat_means, "oil", "Means");  # fmt: skip

# ### Run
# Now we predict.

print("Future/prediction")

(wsat.futr.Truth,
 prod.futr.Truth) = comp1(perm.Truth, wsat.curnt.Truth)  # fmt: skip

for methd in perm:
    if methd not in prod.futr:
        (wsat.futr[methd],
         prod.futr[methd]) = forward_model(perm[methd], wsat.curnt[methd], desc=methd)  # fmt: skip

# Again, data-space inversion requires no new simulations:

prod.futr.ES0 = vect(ens_update0(vect(prod.futr.Prior), **kwargs0), undo=True)

# ### Production plots

plotting.productions(prod.futr, "Future");  # fmt: skip

# ### RMS summary

print(f"Accuracy vs. (supposedly unknown) {emph('future')} production\n")
utils.print_RMSMs(prod.futr, ref="Truth")

# ## Final comments

# It is instructive to run this notebook/script again, but with a different random seed.
# This will yield a different truth, and noisy production data, and so a new
# case/problem, which may be more, or less, difficult.
#
# Another alternative is to only re-run the notebook cells starting from where the prior
# was sampled. Thus, the truth and observations will not change, yet because the prior
# sample will change, the results will change. If this change is significant (which can
# only be asserted by re-running the experiments several times), then you cannot have
# much confidence in your result. In order to fix this, you must increase the ensemble
# size (to reduce sampling error), or try to tune parameters such as the
# localization radius (or more generally, improve your localization implementation).

# Either way, re-running the synthetic experiments and checking that your setup and tuning
# produces resonably improved results will give you confidence in their generalizability;
# as such it it similar in its aim to statistical cross-validation.
# For this reason, synthetic experiments should also be applied in real applications!
# The fact that the real truth is unknown does not prevent you from testing your setup
# with a synthetic truth, sampled from the prior.

# ## References

# <a id="Jaz70">[Jaz70]</a>: Jazwinski, A. H. 1970. *Stochastic Processes and Filtering Theory*. Vol. 63. **Academic Press**.
#
# <a id="Raa19">[Raa19]</a>: Raanes, Patrick Nima, Andreas Størksen Stordal, and Geir Evensen. 2019. *Revising the Stochastic Iterative Ensemble Smoother.* **Nonlinear Processes in Geophysics** 26 (3): 325–38.  https://doi.org/10.5194/npg-26-325-2019.
