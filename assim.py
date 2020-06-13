"""DA/HM example"""

from common import *
from simulation import *
from res_gen import gen_ens
np.random.seed(9)

## Contents:
## ----------
## Gen surfaces of S0
## Investigate covariances, from one well perhaps?
## Pick truth
## Simulate truth/obs
## Initial Kriging/ES
## Assimilate w/ ES
## Compare data mismatch and state-error
## Assimilate w/ iES
## Localize
## Repeat for many experiments



## Gen surfaces of S0
N = 30
sill = 0.7
E = gen_ens(N+1,grid,sill)
xx, E = E[0], E[1:]

if False:
    fig, axs = freshfig(21,nrows=3,ncols=4,sharex=True,sharey=True)
    fig.suptitle("Some realizations")
    for i, (ax, S) in enumerate(zip(axs.ravel(),E)):
        ax.text(0,.85*Ny,str(i),c="w",size=12)
        collections = plot_field(ax, 1-S, vmin=1-sill)
    fig_colorbar(fig,collections)


## Investigate covariances
X = E - E.mean()
xy = (.2, .4)
i = xy2i(*xy)
covs = X[:,i] @ X / (N-1)
# CovMat = X.T @ X / (N-1)
# CovMat = np.cov(E.T)
# vv = diag(CovMat)
vv = np.sum(X*X,0) / (N-1)
# CorrMat = CovMat/sqrt(vv)/sqrt(vv[:,None])
# corrs = CorrMat[i]
corrs = covs/sqrt(vv[i])/sqrt(vv)

if True:
    fig, ax = freshfig(22)
    collections = plot_field(ax, corrs, cmap=mpl.cm.bwr,vmin=-1)
    ax.set(title=f"Correlations vs. {xy}")
    fig.colorbar(collections)


## Simulate truth
dt = 0.025
nT = 28
saturation,production = simulate(nT,xx,dt,dt_plot=None)
p = len(producers)

## Initial Kriging/ES
## Forecast
Eo = np.zeros((N,nT*p))
for n,xn in enumerate(E):
    saturation,production = simulate(nT,xn,dt,dt_plot=None)
    Eo[n] = production.ravel()


## Noisy obs
R = 0.1**2 * np.eye(p)
RR = sp.linalg.block_diag(*[R]*nT)
yy = np.copy(production)
for iT in range(nT):
    yy[iT] += R @ randn(p)

if False:
    prod_plot(production,dt,nT,obs=yy)


## Assimilate w/ ES
yy = yy.ravel()
Y = Eo - Eo.mean()

def mrdiv(b,A):
    return nla.solve(A.T,b.T).T

XY = X.T @ Y
CEo = Y.T @ Y + RR*(N-1)
KG = mrdiv(XY, CEo)

D = randn((N, p*nT)) @ sqrt(RR)

Ea = E + (yy - (Eo+D)) @ KG.T

if True:
    fig, axs = freshfig(23,figsize=(8,8),nrows=2,ncols=2,sharey=True,sharex=True)
    chxx = plot_field(axs[0,0], 1 -xx);               axs[0,0].set_title("Truth")
    chE0 = plot_field(axs[0,1], 1 - E.mean(axis=0));  axs[0,1].set_title("Prior mean")
    chEa = plot_field(axs[1,0], 1 -Ea.mean(axis=0));  axs[1,0].set_title("Post. mean")
    chEr = plot_field(axs[1,1], xx-Ea.mean(axis=0));  axs[1,1].set_title("Difference")
    fig_colorbar(fig,chxx)

norm = lambda xx: np.sqrt(np.sum(xx@xx)/xx.size)
print("Prior err:", norm(xx-E .mean(axis=0)))
print("Post. err:", norm(xx-Ea.mean(axis=0)))

## Compare data mismatch and state-error
## Assimilate w/ EnKF
## Assimilate w/ iES
## Localize
## Repeat for many experiments
