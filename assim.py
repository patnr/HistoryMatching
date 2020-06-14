"""DA/HM example"""

from common import *
from simulation import *
from res_gen import gen_ens
np.random.seed(9)

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
p01 = lambda ax, z: plot_field(ax, z, cmap=mpl.cm.viridis,
                                   vmin=1-sill,vmax=1)
pkg = lambda ax, z, kg: plot_field(ax, z, cmap=mpl.cm.PiYG_r,
                                   vmin=kg.min(), vmax=kg.max())

def plot_realizations(fignum,E,title=""):
    fig, axs = freshfig(fignum,nrows=3,ncols=4,sharex=True,sharey=True)
    fig.suptitle(f"Some realizations -- {title}")
    for i, (ax, S) in enumerate(zip(axs.ravel(),E)):
        ax.text(0,.85*Ny,str(i),c="w",size=12)
        collections = p01(ax, 1-S)
    fig_colorbar(fig, collections)
    plt.pause(.01)


## Gen surfaces of S0
N = 40
sill = 0.7
E0, Cov = gen_ens(N+1,grid,sill)
xx, E0 = E0[0], E0[1:]
if True:
    plot_realizations(23,E0,"Initial")

##
eigs = nla.eigvalsh(Cov)[::-1]
fig, ax = freshfig(21)
# ax.loglog(eigs)
ax.semilogx(eigs)
ax.grid(True,"minor",axis="x")
ax.grid(True,"major",axis="y")
ax.set(xlabel="eigenvalue #",ylabel="var.",title="Spectrum of initial, true cov")


## Initial Kriging/ES
jj = linspace(0, M-1, 10).astype(int)
yy = xx[jj]
Cxy = Cov[:,jj]
Cyy = Cov[jj][:,jj]
Reg = Cxy @ nla.pinv(Cyy)
Kriged = xx.mean() + Reg @ (yy-xx.mean())

print("Error for Krig.: %.4f"%norm(xx-Kriged))
# TODO: use Kriged (ie. best) covariance to generate spread
EK = Kriged + 0.4*(E0-E0.mean(axis=0))
if True:
    plot_realizations(24,EK,"Krig/Prior")


## Simulate truth
dt = 0.025
nT = 28
saturation,production = simulate(nT,xx,dt,dt_plot=None)
p = len(producers)

## Noisy obs
R = 0.01**2 * np.eye(p)
RR = sp.linalg.block_diag(*[R]*nT)
yy = np.copy(production)
for iT in range(nT):
    yy[iT] += R @ randn(p)

if True:
    hh = plot_prod(production,dt,nT,obs=yy)



## Assimilate w/ ES

# Forecast
Eo = np.zeros((N,nT*p))
for n,xn in enumerate(EK):
    saturation,production = simulate(nT,xn,dt,dt_plot=None)
    Eo[n] = production.ravel()

# Analysis
Y  = Eo - Eo.mean(axis=0)
X  = EK - EK.mean(axis=0)
D  = randn((N, p*nT)) @ sqrt(RR)

XY = X.T @ Y
CY = Y.T @ Y + RR*(N-1)
KS = XY @ nla.pinv(CY)

ES = EK + (yy.ravel() - (Eo+D)) @ KS.T

print("Error for prior: %.4f"%norm(xx-EK.mean(axis=0)))
print("Error for ES   : %.4f"%norm(xx-ES.mean(axis=0)))

## Assimilate w/ EnKS
EF = EK.copy()
E  = EK.copy()

EnKS_production = []
for iT in range(nT):
    # Forecast
    Eo = np.zeros((N,p))
    for n,xn in enumerate(E):
        E[n],Eo[n] = simulate(1,xn,dt,dt_plot=None)
    EnKS_production.append(Eo)

    # Obs ens
    Y  = Eo - Eo.mean(axis=0)
    D  = randn((N, p)) @ sqrt(R)
    CY = Y.T @ Y + R*(N-1)
    Ci = nla.pinv(CY)

    # Analysis filter
    X  = E - E.mean(axis=0)
    XY = X.T @ Y
    KF = XY @ Ci
    E  = E + (yy[iT] - (Eo+D)) @ KF.T

    # Analysis smoother
    XK = EF - EF.mean(axis=0)
    XY = XK.T @ Y
    KF = XY @ Ci
    EF = EF + (yy[iT] - (Eo+D)) @ KF.T

##
print("Error for EnKS : %.4f"%norm(xx-EF.mean(axis=0)))
if True:
    fig, axs = freshfig(25,figsize=(8,8),nrows=2,ncols=2,sharey=True,sharex=True)
    chxx = p01(axs[0,0], 1 - xx             );  axs[0,0].set_title("Truth")
    chE0 = p01(axs[0,1], 1 - EK.mean(axis=0));  axs[0,1].set_title("Prior mean")
    chEa = p01(axs[1,0], 1 - ES.mean(axis=0));  axs[1,0].set_title("ES")
    chEr = p01(axs[1,1], 1 - EF.mean(axis=0));  axs[1,1].set_title("EnKS")
    plot_wells(axs[0,0], injectors)
    plot_wells(axs[0,0], producers, False)
    axs[0,0].plot(*array([ind2xy(j) for j in jj]).T, 'w.',ms=3)
    fig_colorbar(fig, chxx)

## Correlations
if True:
    fig, axs = freshfig(22, figsize=(8,8), nrows=2, ncols=2, sharex=True, sharey=True)
    xy = producers[4,:2]
    z = plot_corr_field_vs(axs[0,0],E0,xy,"Initial")
    z = plot_corr_field_vs(axs[0,1],EK,xy,"Kriged")
    z = plot_corr_field_vs(axs[1,0],ES,xy,"ES")
    z = plot_corr_field_vs(axs[1,1],EF,xy,"EnKS")
    fig_colorbar(fig, z)

## Kalman gains
if True:
    fig, axs = freshfig(33, figsize=(8,8), nrows=2, ncols=2, sharex=True, sharey=True)
    i_well = 4
    i_last = i_well + (nT-1)*p
    collections = pkg(axs[0,0], KS.T[i_well], KF)
    collections = pkg(axs[0,1], KS.T[i_last], KF)
    collections = pkg(axs[1,1], KF.T[i_well], KF)
    # Turn off EnKS/initial
    for s in axs[1,0].spines.values(): s.set_color("w")
    axs[1,0].tick_params(colors="w")

    axs[0,0].set_title("Initial")
    axs[0,1].set_title("Final")
    axs[0,0].set_ylabel("ES")
    axs[1,0].set_ylabel("EnKS")
    fig.suptitle(f"KG for a given well obs.\n"
                 "Note how the impact is displaced in time.")
    fig_colorbar(fig, collections)
    i = xy2i(*producers[i_well,:2])
    for ax in axs.ravel():
        ax.plot(*xy, '*k',ms=4)

if True:
    plot_realizations(26,ES,"ES")
    plot_realizations(27,EF,"EnKS")


## EnKS production plot
if True:
    fig, ax = freshfig(35)
    tt = dt*(1+arange(nT))
    for iw, Ew in enumerate(1-np.moveaxis(array(EnKS_production),2,0)):
        ax.plot(tt, Ew, color=hh[iw].get_color(), alpha=0.2)

## Forecast production
E2 = E.copy()

prod2 = []
for iT in range(nT):
    # Forecast
    Eo = np.zeros((N,p))
    for n,xn in enumerate(E2):
        E2[n],Eo[n] = simulate(1,xn,dt,dt_plot=None)
    prod2.append(Eo)

if True:
    tt2 = tt[-1] + dt*(1+arange(nT))
    for iw, Ew in enumerate(1-np.moveaxis(array(prod2),2,0)):
        ax.plot(tt2, Ew, color=hh[iw].get_color(), alpha=0.2)

    ax.axvspan(tt2[0],tt2[-1], alpha=.1, color="b")
