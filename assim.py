"""DA/HM example"""

from common import *
import model
from res_gen import gen_ens

np.random.seed(1)
model.injectors, model.producers, model.Q = init_Q(rand((3,5)).T, rand((3,10)).T)

fig_placement_load()

# TODO:
# Model class
# EnOpt
# Tests
# Clean
# Assimilate w/ iES
# Localize
# Repeat for many experiments

# Ensure uniformity of colorbars

## Gen surfaces of S0
N = 40
sill = 0.7
E0, Cov = gen_ens(N+1,model.grid,sill)
x0, E0 = E0[0], E0[1:]
vm = (1-x0.max(),1)
if True:
    fig, axs = freshfig(23,nrows=3,ncols=4,sharex=True,sharey=True)
    plot_realizations(axs,E0,"Initial",vm)


## Inspect eigenvalue specturm
eigs = nla.eigvalsh(Cov)[::-1]
fig, ax = freshfig(21)
# ax.loglog(eigs)
ax.semilogx(eigs)
ax.grid(True,"minor",axis="x")
ax.grid(True,"major",axis="y")
ax.set(xlabel="eigenvalue #",ylabel="var.",title="Spectrum of initial, true cov")


## Initial Kriging/ES
inds_krigin = linspace(0, M-1, 10).astype(int)
yy = x0[inds_krigin]
Cxy = Cov[:,inds_krigin]
Cyy = Cov[inds_krigin][:,inds_krigin]
Reg = Cxy @ nla.pinv(Cyy)
Kriged = x0.mean() + Reg @ (yy-x0.mean())

print("Error for Krig.: %.4f"%norm(x0-Kriged))
# TODO: use Kriged (ie. best) covariance to generate spread
Eb = Kriged + 0.4*center(E0)
if True:
    fig, axs = freshfig(24,nrows=3,ncols=4,sharex=True,sharey=True)
    plot_realizations(axs,Eb,"Krig/Prior",vm)


## Simulate truth
dt = 0.025
nT = 28
saturation,production = model.simulate(nT,x0,dt,dt_plot=None)
xx = saturation


## Noisy obs
p = len(model.producers)
R = 0.01**2 * np.eye(p)
RR = sp.linalg.block_diag(*[R]*nT)
yy = np.copy(production)
for iT in range(nT):
    yy[iT] += R @ randn(p)

if True:
    fig, ax = freshfig(2)
    hh_y = plot_prod(ax,production,dt,nT,obs=yy)



## Assimilate w/ ES

# Forecast
Eo = np.zeros((N,nT*p))
for n,xn in enumerate(Eb):
    saturation,production = model.simulate(nT,xn,dt,dt_plot=None)
    Eo[n] = production.ravel()

# Analysis
Y  = center(Eo)
X  = center(Eb)
D  = randn((N, p*nT)) @ sqrt(RR)

XY = X.T @ Y
CY = Y.T @ Y + RR*(N-1)
KG_ES = XY @ nla.pinv(CY)
ES = Eb + (yy.ravel() - (Eo+D)) @ KG_ES.T

print("Error for prior: %.4f"%norm(x0-Eb.mean(axis=0)))
print("Error for ES   : %.4f"%norm(x0-ES .mean(axis=0)))


## Assimilate w/ EnKS
EnKS = Eb.copy()
E    = Eb.copy()

EnKS_production = []
for iT in range(nT):
    # Forecast
    Eo = np.zeros((N,p))
    for n,xn in enumerate(E):
        E[n],Eo[n] = model.simulate(1,xn,dt,dt_plot=None)
    EnKS_production.append(Eo)

    # Obs ens
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

##
print("Error for EnKS : %.4f"%norm(x0-EnKS.mean(axis=0)))
if True:
    fig, axs = freshfig(25,figsize=(8,8),nrows=2,ncols=2,sharey=True,sharex=True)
    chxx = plot_field(axs[0,0], 1-x0               , vm); axs[0,0].set_title("Truth")
    chE0 = plot_field(axs[0,1], 1-Eb  .mean(axis=0), vm); axs[0,1].set_title("Prior mean")
    chEa = plot_field(axs[1,0], 1-ES  .mean(axis=0), vm); axs[1,0].set_title("ES")
    # chEr = plot_field(axs[1,1], 1-EnKS.mean(axis=0), vm); axs[1,1].set_title("EnKS")
    chEr = plot_field(axs[1,1], 1-xx[-1]           , vm); axs[1,1].set_title("Truth t=end")
    plot_wells(axs[0,0], model.injectors)
    plot_wells(axs[0,0], model.producers, False)
    axs[0,0].plot(*array([ind2xy(j) for j in inds_krigin]).T, 'w.',ms=3)
    fig_colorbar(fig, chxx)

## Correlations
if True:
    fig, axs = freshfig(22, figsize=(8,8), nrows=2, ncols=2, sharex=True, sharey=True)
    xy = model.producers[4,:2]
    z = plot_corr_field_vs(axs[0,0],E0   ,xy,"Initial")
    z = plot_corr_field_vs(axs[0,1],Eb   ,xy,"Kriged")
    z = plot_corr_field_vs(axs[1,0],ES   ,xy,"ES")
    z = plot_corr_field_vs(axs[1,1],EnKS ,xy,"EnKS")
    fig_colorbar(fig, z)

## Kalman gains
if True:
    fig, axs = freshfig(33, figsize=(8,8), nrows=2, ncols=2, sharex=True, sharey=True)
    def pkg(ax, z):
        a, b = KG_EnKS.min(), KG_EnKS.max()
        return plot_field(ax, z, cmap=mpl.cm.PiYG_r, vmin=a, vmax=b)
    i_well = 4
    i_last = i_well + (nT-1)*p
    collections = pkg(axs[0,0], KG_ES  .T[i_well])
    collections = pkg(axs[0,1], KG_ES  .T[i_last])
    collections = pkg(axs[1,1], KG_EnKS.T[i_well])
    # Turn off EnKS/initial axis
    for s in axs[1,0].spines.values(): s.set_color("w")
    axs[1,0].tick_params(colors="w")

    axs[0,0].set_title("Initial")
    axs[0,1].set_title("Final")
    axs[0,0].set_ylabel("ES")
    axs[1,0].set_ylabel("EnKS")
    fig.suptitle(f"KG for a given well obs.\n"
                 "Note how the impact is displaced in time.")
    fig_colorbar(fig, collections)
    i = xy2i(*model.producers[i_well,:2])
    for ax in axs.ravel():
        ax.plot(*xy, '*k',ms=4)

if True:
    fig, axs = freshfig(25,nrows=3,ncols=4,sharex=True,sharey=True)
    plot_realizations(axs,ES,"ES",vm)
    fig, axs = freshfig(26,nrows=3,ncols=4,sharex=True,sharey=True)
    plot_realizations(axs,EnKS,"EnKS",vm)


## EnKS production plot
if True:
    fig, ax = freshfig(35)
    tt = dt*(1+arange(nT))
    for iw, Ew in enumerate(1-np.moveaxis(array(EnKS_production),2,0)):
        ax.plot(tt, Ew, color=hh_y[iw].get_color(), alpha=0.2)

## Forecast production from filter analysis
if True:
    Ef = E.copy()
    prodf = []

    for iT in range(nT):
        # Forecast
        Eo = np.zeros((N,p))
        for n,xn in enumerate(Ef):
            Ef[n],Eo[n] = model.simulate(1,xn,dt,dt_plot=None)
        prodf.append(Eo)

    ttf = tt[-1] + dt*(1+arange(nT))
    for iw, Ew in enumerate(1-np.moveaxis(array(prodf),2,0)):
        ax.plot(ttf, Ew, color=hh_y[iw].get_color(), alpha=0.2)

    ax.axvspan(ttf[0],ttf[-1], alpha=.1, color="b")
