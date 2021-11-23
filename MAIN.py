# -*- coding: utf-8 -*-
# # Tutorial on ensemble history matching and optimisation
#
# Copyright Patrick N. Raanes, NORCE, 2020.
#
# This is a self-contained tutorial on history matching (HM) using ensemble methods.
# Please do not hesitate to file issues on
# [GitHub](https://github.com/patricknraanes/HistoryMatching),
# or submit pull requests.

# ## Python in Jupyter

# **Jupyter notebooks** combine **cells/blocks** of code (Python) and text (markdown).
#
# For example, try to *edit* the cell below to insert your name, and then *run* it.

name = "Batman"
print("Hello world! I'm " + name)

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
# you should skip/delete this cell.

github = "https://raw.githubusercontent.com/patricknraanes/HistoryMatching"
# !wget -qO- {github}/master/colab_bootstrap.sh | bash -s

# There is a huge amount of libraries available in **Python**,
# including the popular `numpy (np)` and `matplotlib/pyplot (plt)` packages.
# Try them out by running in the next few cells following,
# which illustrates some algebra using syntax reminiscent of Matlab.

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
from matplotlib.ticker import LogLocator
from mpl_tools.place import freshfig
from numpy import sqrt
from struct_tools import DotDict as Dict
from tqdm.auto import tqdm as progbar

# ## Problem case (simulator, truth, obs)

# For exact reproducibility of our problem/case, we set the random generator seed.

seed = rnd.seed(4)  # very easy
# seed = rnd.seed(5)  # hard
# seed = rnd.seed(6)  # very easy
# seed = rnd.seed(7)  # easy

# Our reservoir simulator takes up about 100 lines of python code. This may seem
# outrageously simple, but serves the purpose of *illustrating* the main features of
# the history matching process. Indeed, we do not detail the simulator code here, but
# simply import it from the accompanying python modules, together with the associated
# plot functionality, the (geostatistical) random field generator, and some linear
# algebra. Hence our focus and code will be of aspects directly related to the history
# matching and optimisation process.

import simulator
import simulator.plotting as plots
from tools import geostat, misc
from tools.misc import center, insert_batches
import tools.localization as loc

# In short, the model is a 2D, two-phase, immiscible, incompressible simulator using
# two-point flux approximation (TPFA) discretisation. It was translated from the Matlab
# code here http://folk.ntnu.no/andreas/papers/ResSimMatlab.pdf

# +
model = simulator.ResSim(Nx=20, Ny=20, Lx=2, Ly=1)

# Also init plots module
plots.model = model
plots.coord_type = "absolute"
# -

# The following declares some data containers to help us keep organised.
# The names have all been shortened to 4 characters, but this is just
# to obtain more convenient code alignment and readability.

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

# Technical note: This data hierarchy is convienient in *this* notebook/script,
# especially for plotting purposes. For example, we can with ease refer to
# `wsat.past.Truth` and `wsat.past.Prior`. The former will be a numpy array of shape
# `(nTime, M)` where `M = model.M`, and the latter will have shape `(N, nTime, M)` where
# `N` is the size of the ensemble. However, in other implementations, different choices
# for the data structure may be more convenient, e.g. where the different components of
# the unknowns are merely concatenated along the last axis, rather than being kept in
# separate dicts.

# #### The unknown: permeability
# We will estimate the log permeability field.  We parameterize the permeability
# parameters via some transform, which becomes part of the forward model. We term the
# parameterized permeability fields "pre-permeability". *If* we use the exponential,
# then we will we working with log-permeabilities. At any rate, the transform should be
# chosen so that the parameterized permeabilities are suited for ensemble methods, i.e.
# are distributed as a Gaussian.  But this consideration must be weighted against the
# fact that that nonlinearity (which is also a difficulty for ensemble methods) in the
# transform might add to the nonlinearity of the total/composite forward model.  In any
# case, since this is a synthetic case, we can freely choose *both* the distribution of
# the parameterized permeabilities, *and* the transform.  Here we use Gaussian fields,
# and a "perturbed" exponential function (to render the problem a little more complex).

# +
def sample_prior_perm(N):
    lperms = geostat.gaussian_fields(model.mesh(), N, r=0.8)
    return lperms

# Also configure plot parameters suitable for pre-perm
plots.styles["pperm"]["levels"] = np.linspace(-4, 4, 21)
plots.styles["pperm"]["ticks"] = np.arange(-4, 4+1)


# -

def perm_transf(x):
    return .1 + np.exp(5*x)
    # return 1000*np.exp(3*x)

# For any type of parameter, one typically has to write a "setter" function that takes
# the vector of parameter parameter values, and applies it to the specific model
# implementation. We could merge this functionality with `perm_transf` (and indeed the
# "setter" function is also part of the composite forward model) but it is convenient to
# separate these implementation specifics from the mathematics going on in
# `perm_transf`.

def set_perm(model, log_perm_array):
    """Set perm. in model code. Duplicates the perm. values in x- and y- dir."""
    p = perm_transf(log_perm_array)
    p = p.reshape(model.shape)
    model.Gridded.K = np.stack([p, p])

# Now we are in position to sample the permeability of the (synthetic) truth.

perm.Truth = sample_prior_perm(1)
set_perm(model, perm.Truth)

# #### Wells
# In this model, wells are represented simply by point **sources** and **sinks**. This
# is of course incredibly basic and not realistic, but works for our purposes. So all we
# need to specify is their placement and flux (which we will not vary in time). The code
# below puts wells on a grid. Try `print(grid2)` to see how to easily specify another
# well configuration.
#
# Since the **boundary conditions** are Dirichlet, specifying *zero flux*, and the fluid
# is incompressible, the total of the source terms must equal that of the sinks. This is
# ensured by the `config_wells` function used below.

grid1 = [.1, .9]
grid2 = np.dstack(np.meshgrid(grid1, grid1)).reshape((-1, 2))
rates = np.ones((len(grid2), 1))  # ==> all wells use the same (constant) rate
model.config_wells(
    # Each row in `inj` and `prod` should be a tuple: (x, y, rate),
    # where x, y ∈ (0, 1) and rate > 0.
    inj  = [[0.50, 0.50, 1.00]],
    prod = np.hstack((grid2, rates)),
);

# #### Plot
# Let's take a moment to visualize the (true) model permeability field,
# and the well locations.

fig, ax = freshfig("True perm. field", figsize=(1.5, 1), rel=1)
# plots.field(ax, perm.Truth, "pperm")
plots.field(ax, perm_transf(perm.Truth),
            locator=LogLocator(), wells=True, colorbar=True)
fig.tight_layout()


# #### Observation operator
# The data will consist in the water saturation of at the well locations, i.e. of the
# production. I.e. there is no well model. It should be pointed out, however, that
# ensemble methods technically support observation models of any complexity, though your
# accuracy mileage may vary (again, depending on the incurred nonlinearity and
# non-Gaussianity). Furthermore, it is also no problem to include time-dependence in the
# observation model.

obs_inds = [model.xy2ind(x, y) for (x, y, _) in model.producers]
def obs_model(water_sat):
    return water_sat[obs_inds]

# #### Simulation
# The following generates the synthetic truth evolution and data.

T = 1
dt = 0.025
nTime = round(T/dt)
wsat.init.Truth = np.zeros(model.M)

(wsat.past.Truth,
 prod.past.Truth) = misc.repeat(model.step, nTime, wsat.init.Truth, dt, obs_model)

# #### Animation
# Run the code cells below to get an animation of the oil saturation evolution.
# Injection/production wells are marked with triangles pointing down/up.
# The (untransformed) pre-perm field is plotted, rather than the actual permeability.

# %%capture
animation = plots.dashboard("Truth", perm, wsat.past, prod.past);

# Note: can take up to a minute to appear
animation

# #### Noisy obs
# In reality, observations are never perfect. To emulate this, we corrupt the
# observations by adding a bit of noise.

prod.past.Noisy = prod.past.Truth.copy()
nProd = len(model.producers)  # num. of obs (each time)
R = 1e-3 * np.eye(nProd)
for iT in range(nTime):
    prod.past.Noisy[iT] += sqrt(R) @ rnd.randn(nProd)


# Plot of observations (and their noise):

fig, ax = freshfig("Observations", figsize=(2, .7), rel=True)
plots.production1(ax, prod.past.Truth, prod.past.Noisy);

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

# Note that field (before transformation) is Gaussian with (expected) mean 0 and variance 1.
print("Prior mean:", np.mean(perm.Prior))
print("Prior var.:", np.var(perm.Prior))

# #### Histogram
# Let us inspect the parameter values in the form of their histogram.
# Note that the histogram of the truth is simply counting the values of a single field,
# whereas the histogram of the ensemble counts the values of `N` fields.

fig, ax = freshfig("Perm.", figsize=(1.5, .7), rel=1)
bins = np.linspace(*plots.styles["pperm"]["levels"][[0, -1]], 32)
for label, perm_field in perm.items():
    x = perm_field.ravel()
    ax.hist(perm_transf(x),
            perm_transf(bins),
            # Divide counts by N to emulate `density=1` for log-scale.
            weights=(np.ones_like(x)/N if label != "Truth" else None),
            label=label, alpha=0.3)
ax.set(xscale="log", xlabel="Permeability", ylabel="Count")
ax.legend()
fig.tight_layout()

# Since the x-scale is logarithmic, the prior's histogram should look Gaussian if
# `perm_transf` is purely exponential. By contrast, the historgram of the truth is from
# a single (spatially extensive) realisation, and therefore will contain significant
# sampling "error".

# #### Field plots
# Below we can see some (pre-perm) realizations (members) from the ensemble.

plots.fields(perm.Prior, "pperm", "Prior");

# #### Variance/Spectrum
# In practice, of course, we would not be using an explicit `Cov` matrix when generating
# the prior ensemble, because it would be too large.  However, since this synthetic case
# in being made that way, let's inspect its spectrum.

U, svals, VT = sla.svd(perm.Prior)
plots.spectrum(svals, "Prior cov.");

# With our limited ensemble size, we see no clear cutoff index. In other words, we are
# not so fortunate that the prior is implicitly restricted to some subspace that is of
# lower rank than our ensemble. This is a very realistic situation, and indicates that
# localisation (implemented further below) will be very beneficial.

# ## Forward model

# In order to (begin to attempt to) solve the *inverse problem*,
# we first have to be able to solve the *forward problem*.
# Indeed, ensemble methods obtain observation-parameter sensitivities
# from the covariances of the ensemble run through the ("forward") model.
# This is a composite function. In our simple case, it only consists of two steps:
#
# - The main work consists of running the reservoir simulator
#   for each realisation in the ensemble.
# - However, the simulator only inputs/outputs *state* variables,
#   so we also have to take the necessary steps to set the *parameter* values.
#
# This all has to be stitched together; this is not usually a pleasant task, though some
# tools like [ERT](https://github.com/equinor/ert) have made it a little easier.

# A huge technical advantage of ensembel methods is that they are "embarrasingly
# parallelizable", because each member run is complete independent (requires no
# communication) from the others.  We take advantage of this through multiprocessing
# which, in Python, requires very little code overhead.

# Set (int) number of CPU cores to use. Set to False when debugging.
multiprocess = False

def forward_model(nTime, *args, desc=""):
    """Create the (composite) forward model, i.e. forecast. Supports ensemble (2D array) input."""

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
    # In this code we output full time series, but really we need only emit
    # - The state at the final time, for restarts (predictions).
    # - The observations (for the assimilation update).
    # - The variables used for production optimisation
    #   (in this case the same as the obs, namely the production).
    saturation, production = zip(*Ef)

    return np.array(saturation), np.array(production)

# Note that the `args` of `forward_model` should contain **not only** permeability
# fields, but **also** initial water saturations. It also outputs saturations.  Why did
# we make it so?  Because further down we'll be "restarting" (running) the simulator
# from a later point in time (to generate future predictions) at which point the
# saturation fields will depend on the assumed permeability field, and hence vary from
# realisation to realisation.  Therefore this state (i.e. time-dependent, prognostic)
# variable must be part of the input and output of the forward model.
#
# On the other hand, in this case study we assume that the time-0 saturations are not
# uncertain (unknown). Rather than coding a special case in `forward_model` for time-0,
# we can express this 100% knowledge by setting each saturation field equal to the
# *true* time-0 saturation (a constant field of 0).

wsat.init.Prior = np.tile(wsat.init.Truth, (N, 1))

# Now that we have the forward model, we can make prior estimates of the saturation
# evolution and production.  This is interesting in and of itself and, as we'll see
# later, is part of the assimilation process.  Let's run the forward model on the prior.

(wsat.past.Prior,
 prod.past.Prior) = forward_model(nTime, wsat.init.Prior, perm.Prior)

# ## Localisation (*optional*)

# If you choose not to run this section, then you must use a fairly large ensemble size
# in order to obtain results of any value.
#
# Localisation invervenes to fix-up the estimated correlations before they are used. It
# is a method of injecting prior information (distant points are likely not strongly
# codependent) that is not *encoded* in the ensemble (usually due to their finite size).
# Defining an effective localisation mask or tapering function can be a difficult task.

# ### Correlation plots
# The conditioning "update" of ensemble methods is often formulated in terms of a
# "**Kalman gain**" matrix, derived so as to achieve a variety of optimality properties
# (see e.g. [[Jaz70]](#Jaz70)):
# - in the linear-Gaussian case, to compute the correct posterior moments;
# - in the linear (not-necessarily-Gaussian) case, to compute the BLUE,
#   i.e. to achieve orthogonality of the posterior error and innovation;
# - in the non-linear, non-Gaussian case, the ensemble version can be derived as
#   linear regression (with some tweaks) from the perturbed obs. to the unknowns.
#
# Another way to look at it is to ask "what does it do?"
# Heuristically, this may be answered as follows:
#
# - It uses correlation coefficients to establish relationships between
#   observations and unknowns. For example, if there is no correlation, there will
#   be no update (even for iterative methods).
# - It takes into account the "intermingling" of correlations. For example, two
#   measurements/observations that are highly correlated (when including both prior and
#   observation errors) will barely contribute more than either one alone.
# - It takes into account the variables' variance (hence why it works with covariances,
#   and not just correlations), and thereby the their relative uncertainties.
#   Thus, if two variables have equal correlation with an observation,
#   but one is more uncertain, that one will receive a larger update than the other.
#   Also, an observation with a larger variance will have less impact than an
#   observation with a smaller variance. Working with variances also means that
#   the physical units of the variables are inherently accounted for.
#
# In summary, it is useful to investigate the correlation relations of the ensemble,
# especially for the prior.

# #### Auto-correlation for `wsat`
# First, as a sanity check, it is useful to plot the correlation of the saturation field
# at some given time vs. the production at the same time. The correlation should be
# maximal (1.00) at the location of the well in question. Let us verify this: zoom-in
# several times (not available on Colab), centering on the green star, to verify that it
# lies on top of the well of that panel.  The green stars mark the location of the
# maximum of the correlation field.

Field = wsat.past.Prior[:, -1]
Obsvs = prod.past.Prior[:, -1].T
corrs = [misc.corr(Field, obs) for obs in Obsvs]

plots.fields(corrs, "corr", "Saturation vs. obs", argmax=True, wells=True);

# #### Interactive correlation plot
# The following plots a variety of different correlation fields. Each field may
# be seen as a single column (or row) of a larger ("cross")-covariance matrix,
# which would typically be too large for explicit computation or storage. The
# following solution, though, which computes the correlation fields "on the
# fly", should be viable for relatively large scales.

# +
# Compute correlation field
def corr_comp(Field, T, Point, t, x, y):
    Field = prior_fields[Field]
    Point = prior_fields[Point]
    if Field.ndim > 2: Field = Field[:, T]  # noqa
    if Point.ndim > 2: Point = Point[:, t]  # noqa
    Point = Point[:, model.sub2ind(x, y)]
    return misc.corr(Field, Point)

# Available variable types
prior_fields = {
    "Saturation": wsat.past.Prior,
    "Pre-perm": perm.Prior,
}

# Register controls
corr_comp.controls = dict(
    Field = list(prior_fields),
    T = (0, nTime),
    Point = list(prior_fields),
    t = (0, nTime),
    x = (0, model.Nx-1),
    y = (0, model.Ny-1),
)
# -

plots.field_interact(corr_comp, "corr", "Field(T) vs. Point(t, x, y)", argmax=True)

# Use the interative control widgets to investigate the correlation structure.
# Answer and discuss the following questions:
#
# - For each combination of `Field` and `Point`:
#   - Set `T` or `t` to 0. How do the correlation fields look? Why?
#   - Set `T` or `t` to 1. How do the correlation fields look? Why?
# - Set `T = t = 20` and `Field = Point = Saturation`. Why is the green marker
#   (showing the location of the maximum) on top of the crosshairs?
#   Does this hold when `Field != Point` (hint: try moving `x` and `y`)?
# - Set `Field = Point = Pre-perm`, and put the point somewhere near the center.
#   Why is the correlation field so regular (almost perfectly circular or elliptic)?
#   Also note, as you can tell from `corr_comp`,
#   that time plays no role for the perm field.
# - Set `Field = Point = Saturation`, set the slider for `T` to the middle, `t` large,
#   and put the `Point` near a corner (e.g. `x = y = 2`).
#   - Where is the maximum? And minimum? Does this make sense?
#   - Gradually increase `T`. How do the extrema move? Why?.
# - TODO: Add more remarks/questions

# ### Tapering

# #### Plot of localized domains

# #### Plot of localized correlations

# ## Assimilation

# ### Ensemble update

class pre_compute_ens_update:
    """Compute the ensemble gain * innovation, using random obs. perturbations.

    In other words, prepare the ensemble update/conditioning (Bayes' rule).
    This pre-computed "X5" matrix can then be applied to *any* ensemble,
    similar to how the state/parm vector of unknowns can be augmented by anything.

    NB: obs_err_cov is treated as diagonal. A non-diagonal implementation requires using
    `sla.sqrtm` or equivalent EVD manips.

    NB: some of these formulae are transposed and reversed compared to EnKF literature
    convention. The reason is that we stack the members as rows (instead of columns).
    [Rationale](https://nansencenter.github.io/DAPPER/dev_guide.html#conventions)
    """

    def __init__(self, obs_ens, observations, obs_err_cov):
        """Prepare the update."""
        Y, _        = center(obs_ens)
        obs_cov     = obs_err_cov*(len(Y)-1) + Y.T@Y
        obs_pert    = rnd.randn(*Y.shape) @ sqrt(obs_err_cov)
        # obs_pert  = center(obs_pert, rescale=True)
        innovations = observations - (obs_ens + obs_pert)

        # (pre-) Kalman gain * Innovations.Also called the X5 matrix by Evensen'2003.
        self.KGdY = innovations @ sla.pinv2(obs_cov) @ Y.T

    def __call__(self, E):
        """Do the update."""
        return E + self.KGdY @ center(E)[0]

# ### Bug check

# It is very easy to introduce bugs in the code.
# Fortunately, most can be eliminated with a few simple tests.
#
# For example, let us generate a case where both $x$
# and the observation error are (independently) $\mathcal{N}(0, 2)$,
# while the forward model is just the identity

# Note: the prefix "gg_" stands for Gaussian-Gaussian
gg_ndim = 3
gg_prior = sqrt(2) * rnd.randn(1000, gg_ndim)
gg_postr = pre_compute_ens_update(
    obs_ens      = gg_prior,
    observations = 10*np.ones(gg_ndim),
    obs_err_cov  = 2*np.eye(gg_ndim),
)(gg_prior)

# From theory, we know that $x|y \sim \mathcal{N}(y/2, 1)$.
# Let us verify that the method reproduces this (up to sampling error)

with np.printoptions(precision=1):
    print(np.mean(gg_postr, 0))
    print(np.cov(gg_postr.T))


# ### Ensemble smoother

# #### Why not filtering?
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
# jointly-updated (rather than re-generated) state fields.  This makes the
# parameter-only update of the (batch) smoothers appealing.

# We have organised our simulated ensemble data in 3D arrays, with time along the middle
# dimension (the 1st axis). Ensemble methods have no notion of 3D arrays, so we need to
# be able to flatten the time dimension, and to undo this.

def t_ravel(x, undo=False):
    """Ravel/flatten the last two axes, or undo this operation."""
    if undo:
        *N, ab = x.shape
        return x.reshape(N + [nTime, ab//nTime])
    else:
        *N, a, b = x.shape
        return x.reshape(N + [a*b])

# #### Compute

# Pre-compute
ens_update0 = pre_compute_ens_update(
    obs_ens      = t_ravel(prod.past.Prior),
    observations = t_ravel(prod.past.Noisy),
    obs_err_cov  = sla.block_diag(*[R]*nTime),
)
# Apply
perm.ES = ens_update0(perm.Prior)

# #### Field plots
# Let's plot the updated, initial ensemble.

plots.fields(perm.ES, "pperm", "ES (posterior)");

# We will see some more diagnostics later.

# ### With localisation

def enAnalysis(E, Eo, y, R):
    return pre_compute_ens_update(obs_ens=Eo, observations=y, obs_err_cov=R)(E)


def localized_ens_update0(E, Eo, R, y, domains, obs_taperer, mp=map):
    """Perform local analysis update for the LETKF."""
    def local_analysis(ii):
        """Perform analysis, for state index batch `ii`."""
        # Locate local domain
        oBatch, tapering = obs_taperer(ii)
        Eii = E[:, ii]

        # No update
        if not oBatch.any():
            return Eii

        # Localize
        Yl  = Y[:, oBatch]
        dyl = dy[oBatch]
        tpr = sqrt(tapering)

        # Since R^{-1/2} was already applied (necesry for effective_N), now use R=Id.
        # TODO 4: the cost of re-init this R might not always be insignificant.
        R = np.eye(len(dyl))

        # Update
        return enAnalysis(Eii, Yl*tpr, dyl*tpr, R)

    # Prepare analysis
    Y, xo = center(Eo)

    # TODO: leave to EnKF_analysis
    # Transform obs space
    Y  = Y        @ np.diag(1/sqrt(np.diag(R)))
    dy = (y - xo) @ np.diag(1/sqrt(np.diag(R)))

    # Run
    EE = mp(local_analysis, domains)
    return insert_batches(np.zeros_like(E), domains, EE)


# #### Bug check

gg_postr = localized_ens_update0(
    E  = gg_prior,
    Eo = gg_prior,
    R  = 2*np.eye(gg_ndim),
    y  = 10*np.ones(gg_ndim),
    # Localize simply by processing each dim. entirely seperately:
    domains=np.arange(gg_ndim),
    obs_taperer=(lambda i: (i == np.arange(gg_ndim), 1)),
)

with np.printoptions(precision=1):
    print(np.mean(gg_postr, 0))
    print(np.cov(gg_postr.T))


# ## Localize point obs of an N-D, homogeneous, rectangular domain.
# TODO: localise differently in time?
# TODO: localisation adds prior knowledge, because we know htat ensemble does
#       not "encode" all of our prior knowledge (especially due to sampling error)

# Define local domains
domains = loc.rectangular_partitioning(model.shape, (5, 7))

# Illustration: fill each domain by random color. Should produce rectangle patchwork!
colors = rnd.choice(len(domains), len(domains), False)
Z = np.zeros(model.shape)
for d, c in zip(domains, colors):
    Z[tuple(model.ind2sub(d))] = c
plt.imshow(Z, cmap="tab20")

distances = loc.pairwise_distances(model.ind2sub(np.arange(model.M)).T,
                                   model.ind2sub(obs_inds*nTime).T)

def obs_taperer(batch):
    dists = distances[batch].mean(axis=0)  # obs - batch(mean location)
    coeffs = loc.dist2coeff(dists, radius=10, tag="GC")
    non0 = coeffs > 1e-3
    return non0, coeffs[non0]

### fawef
perm.LES = localized_ens_update0(perm.Prior, t_ravel(prod.past.Prior),
                                 sla.block_diag(*[R]*nTime), t_ravel(prod.past.Noisy),
                                 domains, obs_taperer)

# ### Iterative ensemble smoother

# #### Why iterate?
# Because of non-linearity of the forward model, the likelihood is non-Gaussian, and
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
        stat.rmse += [misc.RMSM(E, perm.Truth).rmse]

        # Forecast.
        _, Eo = forward_model(nTime, wsat.init.Prior, E, desc=f"Iter #{itr}")
        Eo = t_ravel(Eo)

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

# #### Compute

perm.IES, diagnostics = IES(
    ensemble     = perm.Prior,
    observations = t_ravel(prod.past.Noisy),
    obs_err_cov  = sla.block_diag(*[R]*nTime),
    stepsize     = 1,
)

# #### Field plots
# Let's plot the updated, initial ensemble.

plots.fields(perm.IES, "pperm", "IES (posterior)");

# The following plots the cost function(s) together with the error compared to the true
# (pre-)perm field as a function of the iteration number. Note that the relationship
# between the (total, i.e. posterior) cost function  and the RMSE is not necessarily
# monotonic. Re-running the experiments with a different seed is instructive. It may be
# observed that the iterations are not always very successful.

fig, ax = freshfig("IES Objective function")
ls = dict(postr="-", prior=":", lklhd="--")
for name, J in diagnostics.obj.items():
    ax.plot(np.sqrt(J), color="b", ls=ls[name], label=name)
ax.set_xlabel("iteration")
ax.set_ylabel("RMS mismatch", color="b")
ax.tick_params(axis='y', labelcolor="b")
ax.legend()
ax2 = ax.twinx()  # axis for rmse
ax2.set_ylabel('RMS error', color="r")
ax2.plot(diagnostics.rmse, color="r")
ax2.tick_params(axis='y', labelcolor="r")

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

# ### Means vs. True field

# #### RMS summary
# RMS stands for "root-mean-square(d)" and is a summary measure for deviations.
# With ensemble methods, it is (typically, and in this case study) applied
# to the deviation from the **ensemble mean**, whence the trailing `M` in `RMSM` below.

print("Stats vs. true field\n")
misc.RMSMs(perm, ref="Truth")

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

plots.fields(perm_means, "pperm", "Means");

# ### Means vs. Data mismatch (past production)
# In synthetic experiments such as this one, is is instructive to computing the "error":
# the difference/mismatch of the (supposedly) unknown parameters and the truth.  Of
# course, in real life, the truth is not known.  Moreover, at the end of the day, we
# mainly care about production rates and saturations.

# #### Re-run
# Therefore, let us now compute the "residual" (i.e. the mismatch between
# predicted and true *observations*), which we get from the predicted
# production "profiles".

(wsat.past.ES,
 prod.past.ES) = forward_model(nTime, wsat.init.Prior, perm.ES)

(wsat.past.IES,
 prod.past.IES) = forward_model(nTime, wsat.init.Prior, perm.IES)

# It is Bayesian(ally) consistent to apply the pre-computed ES gain to any
# un-conditioned ensemble, e.g. that of the prior's production predictions. This can be
# seen (by those familiar with that trick) by state augmentation. This provides another
# posterior approximation of the production history -- one which doesn't require running
# the model again (in contrast to what we did for `prod.past.(I)ES` immediately above).
# Since it requires 0 iterations, let's call this "ES0". Let us try that as well.

prod.past.ES0 = t_ravel(ens_update0(t_ravel(prod.past.Prior)), undo=True)

# #### Production plots

plots.productions(prod.past, "Past");

# #### RMS summary

print("Stats vs. past production (i.e. NOISY observations)\n")
misc.RMSMs(prod.past, ref="Noisy")

# The RMSE obtained from the (given method of approximate computation of the) posterior
# should pass two criteria.
# - It should be lower than the `rmse` of the (noisy) observations.
#   Aside: here, this is represented by the `rmse` of the `Truth`,
#   since we've set `Noisy` as the reference
#   (for realism, since in practice the Truth is unknown).
# - It should be lower than the `rmse` of the `Prior`.
#   Note that this may occur even if the updated `perm` did not achieve a lower `rmse`;
#   A related phenomenon is that the `rmse` of `ES0` is likely to be very low.
#   However, in both cases, as we shall see,
#   the `rmse` of the `perm` field is a much better indicator of
#   **predictive** skill of the production
#   (as well as the saturation fields as a whole).

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
# off, then the ensemble posteriors (ES, EnKS, ES0) should also surround the data, but
# with a tighter fit.

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
plots.fields(wsat_means, "oil", "Means");

# #### Run
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

prod.futr.ES0 = t_ravel(ens_update0(t_ravel(prod.futr.Prior)), undo=True)

# #### Production plots

plots.productions(prod.futr, "Future");

# #### RMS summary

print("Stats vs. (supposedly unknown) future production\n")
misc.RMSMs(prod.futr, ref="Truth")


# ## Robust optimisation

# NB: This section is very unfinished, and should not be seen as a reference.

# This section uses EnOpt to optimise the controls: the relative rates of production of
# the wells (again, for simplicity, these will be constant in time).

# Ojective function definition: total oil from production wells. This objective function
# takes an ensemble (`*E`) of unknowns (`wsat, perm`) and controls (`rates`) and outputs
# the corresponding ensemble of total oil productions.

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


# ## Final comments

# It is instructive to run this notebook/script again, but with a different random seed.
# This will yield a different truth, and noisy production data, and so a new
# case/problem, which may be more, or less, difficult.
#
# Another alternative is to only re-run the notebook cells starting from where the prior
# was sampled. Thus, the truth and observations will not change, yet because the prior
# sample will change, the results will change. If this change is significant (which can
# only be asserted by re-running the experiments several times), then you can not have
# much confidence in your result. In order to fix this, you must increase the ensemble
# size (to reduce sampling error), or play with the tuning parameters such as the
# localisation radius (or more generally, improve your localisation implementation).
#
# Such re-running of the synthetic experiments is similar in aim to statistical
# cross-validation. Note that it may (and should) also be applied in real applications!
# Of couse, then the truth is unknown. But even though the truth is unknown, a synthetic
# truth can be sampled from the prior uncertainty. And at the very least, the history
# matching and optimisation methods should yield improved performance (reduced errors
# and increased NPV) in the synthetic case.

# ## References

# <a id="Jaz70">[Jaz70]</a>: Jazwinski, A. H. 1970. *Stochastic Processes and Filtering Theory*. Vol. 63. Academic Press.
#
# <a id="Raa19">[Raa19]</a>: Raanes, Patrick Nima, Andreas Størksen Stordal, and Geir Evensen. 2019. “Revising the Stochastic Iterative Ensemble Smoother.” *Nonlinear Processes in Geophysics* 26 (3): 325–38.  https://doi.org/10.5194/npg-26-325-2019.
