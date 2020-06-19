# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # History matching tutorial
# This notebook presents a tutorial on history matching (HM) using ensemble methods.
# It is a work in progress.

# Import tools, model, and ensemble-generator
from common import *
import model
from res_gen import gen_ens

try:
    __IPYTHON__
    from IPython import get_ipython
    is_notebook_or_qt = 'zmq' in str(type(get_ipython())).lower()
except (NameError,ImportError):
    is_notebook_or_qt = False

if is_notebook_or_qt:
    mpl.rcParams.update({'font.size': 13})
    mpl.rcParams["figure.figsize"] = [8,6]
else:
    fig_placement_load()

def validate_ens(E):
    # assert E.max() <= 1 + 1e-10
    # assert E.min() >= 0 - 1e-10
    if (E.max() <= 1 + 1e-10) or (E.min() >= 0 - 1e-10):
        print("Warning: clipping ensemble.")
        E.clip(0,1,out=E)
    return E

# ## Model and case
# The reservoir model, which takes $\approx 100$ lines of python code, is a 2D, two-phase, immiscible, incompressible simulator using TPFA,
# grabbed from http://folk.ntnu.no/andreas/papers/ResSimMatlab.pdf
#
# The model supports inhomogeneous permeabilities and porosities, but to keep things as simple as possible we will only be estimating the water saturation.
# Specifically, we will focus on the initial saturation, as is commonplace in ensemble HM.
# The data will be production saturations.

# Set well locations and relative throughput
np.random.seed(1)
# injectors = rand((2,3))
# producers = rand((4,3))
injectors = [ [0.1, 0.0, 1.0], [0.9, 0.0, 1.0] ]
producers = [ [0.1, 0.7, 1.0], [0.9, 1.0, 1.0] , [.5,.2,1]]
model.injectors, model.producers, model.Q = init_Q(injectors, producers)

# ## Initial ensemble
# We generate Gaussian fields with a given variogram to provide an initial ensemble, as well as the (statistically indistinguishable) truth. Below we plot some of the realizations.

# +
## Gen surfaces of water saturation
N = 40
sill = 0.7
E0, Cov = gen_ens(N+1,model.grid,sill)

# Pick first one as "truth"
x0, E0 = E0[0], E0[1:]
print("Error Initially: %.4f"%norm(x0-E0.mean(0)))


# Plot
vm = (.2,1)
fig, axs = freshfig(23,nrows=3,ncols=4,sharex=True,sharey=True)
plot_realizations(axs,E0,"Initial",vm)
# -


# ## Simulation of the synthetic truth
# Plotted below is the oil saturation before/after production.
# Injection wells (numbered) have blue triangle markers, production have orange.
# The obs. locations used for Kriging are marked with white dots.

# +
dt = 0.025
nT = 28
tt = dt*(1+arange(nT))
saturation,production = model.simulate(nT,x0,dt,dt_plot=None)
xx = saturation # "x" for unknown-state

# Plot
fig, axs = freshfig(19,figsize=(12,6),ncols=2,sharey=True)
chxx = plot_field(axs[0], 1-x0      , vm); axs[0].set_title("Truth")
chEr = plot_field(axs[1], 1-xx[-1]  , vm); axs[1].set_title("Truth at t=end")
plot_wells(axs[0], model.injectors)
plot_wells(axs[0], model.producers, False)
fig_colorbar(fig, chxx)
# -

# ## Noisy obs
# The observations are corrupted with a little bit of noise.

# +
p = len(model.producers)
R = 0.01**2 * np.eye(p)
RR = sp.linalg.block_diag(*[R]*nT)
yy = np.copy(production)
for iT in range(nT):
    yy[iT] += R @ randn(p)

fig, ax_prod = freshfig(2)
hh_y = plot_prod(ax_prod,production,dt,nT,obs=yy)
# -


# Perhaps the ensemble spread is too large for history matching methods to be
# effective (because they produce too nonlinear behaviours). In that case, we
# might adjust our test case by reducing the initial (prior) ensemble spread,
# also adjusting its mean towards the truth. A less artifical means is Kriging
# (geostatistics), illustrated below. However, with the default parameters,
# this adjustment is not necessary, but is left for completeness.

# +
## Initial Kriging/ES
# kriging_inds = linspace(0, M-1, 10).astype(int)
# kriging_obs = x0[kriging_inds]
# Cxy = Cov[:,kriging_inds]
# Cyy = Cov[kriging_inds][:,kriging_inds]
# Reg = Cxy @ nla.pinv(Cyy)
# Kriged = x0.mean() + Reg @ (kriging_obs-x0.mean())

# print("Error for Krig.: %.4f"%norm(x0-Kriged))
# TODO: use Kriged (ie. best) covariance to generate spread

# Eb = Kriged + 0.4*center(E0)
# fig, axs = freshfig(24,nrows=3,ncols=4,sharex=True,sharey=True)
# axs[0,0].plot(*array([ind2xy(j) for j in kriging_inds]).T, 'w.',ms=10)
# plot_realizations(axs,Eb,"Krig/Prior",vm)


# TODO
# Eb = E0 + (Reg @ (kriging_obs-E0[:,kriging_inds]).T).T
Eb = E0.copy()

Eb = validate_ens(Eb)
# -


# In practice, of course, we would not be using an explicit `Cov` matrix for this, because it would be too large.
# However, since this synthetic case in being made that way, let's go ahead an inspect what kind of spectrum it has.

## Inspect eigenvalue specturm
eigs = nla.eigvalsh(Cov)[::-1]
fig, ax = freshfig(21)
#ax.loglog(eigs)
ax.semilogx(eigs)
ax.grid(True,"minor",axis="x")
ax.grid(True,"major",axis="y")
ax.set(xlabel="eigenvalue #",ylabel="var.",title="Spectrum of initial, true cov");

# It appears that the spectrum tails off around $N=30$, so maybe this ensemble size will suffice. However, try plotting the above using `loglog` instead of `semilogx`, and you might not be so convinced. Nevertheless, as we shall see, it does seem to yield tolerable results, even without localization.

# ## Assimilate w/ ES
# First, we assimilate using the batch method: ensemble smoother (ES).

# +
# Forecast
prior_production = np.zeros((nT,N,p))
for n,xn in enumerate(Eb):
    _,prior_production[:,n,:] = model.simulate(nT,xn,dt,dt_plot=None)

# ## Plot prior production
for iw, Ew in enumerate(1-np.moveaxis(array(prior_production),2,0)):
    ax_prod.plot(tt, Ew, color=hh_y[iw].get_color(), alpha=0.1)

# Analysis
Eo = prior_production.swapaxes(0,1).reshape((N,nT*p))
Y  = center(Eo)
X  = center(Eb)
D  = randn((N, p*nT)) @ sqrt(RR)

XY = X.T @ Y
CY = Y.T @ Y + RR*(N-1)
KG_ES = XY @ nla.pinv(CY)
ES = Eb + (yy.ravel() - (Eo+D)) @ KG_ES.T

ES = validate_ens(ES)
# -

# ## Assimilate w/ EnKS
# Next, we test using the EnKS.

# +
EnKS = Eb.copy()
E    = Eb.copy()
EnKS_production = np.zeros((nT,N,p))

for iT in range(nT):
    # Forecast
    for n,xn in enumerate(E):
        E[n],EnKS_production[iT,n] = model.simulate(1,xn,dt,dt_plot=None)

    # Obs ens
    Eo = EnKS_production[iT]
    Y  = center(Eo)
    D  = randn((N, p)) @ sqrt(R)
    CY = Y.T @ Y + R*(N-1)
    Ci = nla.pinv(CY)

    # Analysis filter
    X = center(E)
    XY = X.T @ Y
    KG_EnKS = XY @ Ci
    E = E + (yy[iT] - (Eo+D)) @ KG_EnKS.T

    # Analysis smoother
    XK = center(EnKS)
    XY = XK.T @ Y
    KG_EnKS = XY @ Ci
    EnKS = EnKS + (yy[iT] - (Eo+D)) @ KG_EnKS.T

print("Error for prior: %.4f"%norm(x0-Eb.mean(axis=0)))
print("Error for ES   : %.4f"%norm(x0-ES .mean(axis=0)))
print("Error for EnKS : %.4f"%norm(x0-EnKS.mean(axis=0)))
# -

# ## Compare ensemble mean fields
# Plots of petroleum saturation fields.

fig, axs = freshfig(25,figsize=(8,8),nrows=2,ncols=2,sharey=True,sharex=True)
chxx = plot_field(axs[0,0], 1-x0               , vm); axs[0,0].set_title("Truth")
chE0 = plot_field(axs[0,1], 1-Eb  .mean(axis=0), vm); axs[0,1].set_title("Prior mean")
chEa = plot_field(axs[1,0], 1-ES  .mean(axis=0), vm); axs[1,0].set_title("ES")
chEr = plot_field(axs[1,1], 1-EnKS.mean(axis=0), vm); axs[1,1].set_title("EnKS")
fig_colorbar(fig, chxx)

# ## Correlation fields
# Plot of correlation fields (of the saturation at t=0) vs. a specific point

fig, axs = freshfig(22, figsize=(8,8), nrows=2, ncols=2, sharex=True, sharey=True)
i_well = 2
xy = model.producers[i_well,:2]
z = plot_corr_field_vs(axs[0,0],E0   ,xy,"Initial")
# z = plot_corr_field_vs(axs[0,1],Eb   ,xy,"Kriged")
z = plot_corr_field_vs(axs[1,0],ES   ,xy,"ES")
z = plot_corr_field_vs(axs[1,1],EnKS ,xy,"EnKS")
fig_colorbar(fig, z)

# ## Kalman gains
# Plot of Kalman gain fields vs. specific observations.

# +
fig, axs = freshfig(33, figsize=(8,8), nrows=2, ncols=2, sharex=True, sharey=True)
def pkg(ax, z):
    a, b = KG_EnKS.min(), KG_EnKS.max()
    return plot_field(ax, z, cmap=mpl.cm.PiYG_r, vmin=a, vmax=b)
i_last = i_well + (nT-1)*p
collections = pkg(axs[0,0], KG_ES  .T[i_well])
collections = pkg(axs[0,1], KG_ES  .T[i_last])
collections = pkg(axs[1,1], KG_EnKS.T[i_well])
# Turn off EnKS/initial axis
for s in axs[1,0].spines.values(): s.set_color("w")
axs[1,0].tick_params(colors="w")

axs[0,0].set_title("t=1")
axs[0,1].set_title("t=end")
axs[0,0].set_ylabel("ES")
axs[1,0].set_ylabel("EnKS")
for ax in axs.ravel():
    ax.plot(*model.producers[i_well,:2], '*k',ms=10)
fig.suptitle(f"KG for a given well obs.\n"
             "Note how the impact is displaced in time.")
fig_colorbar(fig, collections)

# -

# ## Posterior realizations

fig, axs = freshfig(27,nrows=3,ncols=4,sharex=True,sharey=True)
plot_realizations(axs,ES,"ES",vm)

fig, axs = freshfig(28,nrows=3,ncols=4,sharex=True,sharey=True)
plot_realizations(axs,EnKS,"EnKS",vm)

# ## Production plots for EnKS

# +
fig, ax = freshfig(35)
hh_y = plot_prod(ax,production,dt,nT,obs=yy)

for iw, Ew in enumerate(1-np.moveaxis(array(EnKS_production),2,0)):
    ax.plot(tt, Ew, color=hh_y[iw].get_color(), alpha=0.1)
    
## Forecast production from filter analysis
Ef = E.copy()
prodf = []

for iT in range(50):
    # Forecast
    Eo = np.zeros((N,p))
    for n,xn in enumerate(Ef):
        Ef[n],Eo[n] = model.simulate(1,xn,dt,dt_plot=None)
    prodf.append(Eo)

ttf = tt[-1] + dt*(1+arange(iT+1))
for iw, Ew in enumerate(1-np.moveaxis(array(prodf),2,0)):
    ax.plot(ttf, Ew, color=hh_y[iw].get_color(), alpha=0.1)

ax.axvspan(ttf[0],ttf[-1], alpha=.1, color="b")
