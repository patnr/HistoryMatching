import IPython.display as ipy_disp
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from mpl_tools.misc import axprops, fig_colorbar, freshfig, is_notebook_or_qt

import model
from tools import center


def display(animation):
    if is_notebook_or_qt:
        ipy_disp.display(ipy_disp.HTML(animation.to_jshtml()))
    else:
        plt.show(block=False)


# These do not get updated with model
Nx, Ny, Dx, Dy = model.grid
hx, hy = Dx/Nx, Dy/Ny


def field(ax, zz, **kwargs):
    """Contour-plot the field contained in `zz`."""

    # Need to transpose coz model assumes shape (Nx, Ny),
    # and contour() uses the same orientation as array printing.
    Z = zz.reshape(model.gridshape).T

    ax.set(**axprops(kwargs))

    # ax.imshow(Z[::-1])
    collections = ax.contourf(
        Z, **kwargs,
        # Using origin="lower" puts the points in the gridcell centers.
        # This means that the plot wont extend all the way to the edges.
        # Unfortunately, there does not seem to be a way to pad the margins,
        # except manually padding Z on all sides, or using origin=None
        # (the mpl default), which would be wrong because it merely
        # stretches rather than pads.
        origin="lower", extent=(0, 1, 0, 1))

    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))

    if ax.is_first_col():
        ax.set_ylabel("y")
    if ax.is_last_row():
        ax.set_xlabel("x")

    return collections


# Colormap for saturation
lin_cm = mpl.colors.LinearSegmentedColormap.from_list
def ccnvrt(c): return np.array(mpl.colors.colorConverter.to_rgb(c))


# Plain:
cOil   = "red"
cWater = "blue"
# Pastel/neon:
# cWater = "#01a9b4"
# cOil   = "#d8345f"
# Pastel:
# cWater = "#086972"
# cOil   = "#e58a8a"
# middle = .3*ccnvrt(cWater) + .7*ccnvrt(cOil)
# cm_ow = lin_cm("", [cWater,middle,cOil])
# Pastel:
cm_ow = lin_cm("", [(0, "#1d9e97"), (.3, "#b2e0dc"), (1, "#f48974")])

# cm_ow = mpl.cm.viridis


def oilfield(ax, ss, **kwargs):
    levels = np.linspace(0 - 1e-7, 1 + 1e-7, 11)
    return field(ax, 1-ss, levels=levels, cmap=cm_ow, **kwargs)


def corr_field(ax, A, b, title="", **kwargs):
    N = len(b)
    # CovMat = X.T @ X / (N-1)
    # CovMat = np.cov(E.T)
    # vv = diag(CovMat)
    # CorrMat = CovMat/sqrt(vv)/sqrt(vv[:,None])
    # corrs = CorrMat[i]
    A     = center(A)
    b     = center(b)
    covs  = b @ A / (N-1)
    varA  = np.sum(A*A, 0) / (N-1)
    varb  = np.sum(b*b, 0) / (N-1)
    corrs = covs/np.sqrt(varb)/np.sqrt(varA)

    ax.set(title=title)
    cc = field(ax, corrs, levels=np.linspace(-1, 1, 11), cmap=mpl.cm.bwr, **kwargs)
    return cc


def corr_field_vs(ax, E, xy, title="", **kwargs):
    i = model.xy2ind(*xy)
    b = E[:, i]
    cc = corr_field(ax, E, b, title, **kwargs)
    ax.plot(*xy, '*k', ms=4)
    return cc


def well_scatter(ax, ww, inj=True):
    if inj:
        c = "w"
        d = "k"
        m = "v"
    else:
        c = "k"
        d = "w"
        m = "^"
    # Marker
    ax.plot(*ww.T[:2], m+c, ms=16, mec="k", clip_on=False)
    # Text
    if not inj:
        ww.T[1] -= 0.01
    for i, w in enumerate(ww):
        ax.text(*w[:2], i, color=d, ha="center", va="center")


def production1(ax, production, obs=None):
    hh = []
    tt = 1+np.arange(len(production))
    for i, p in enumerate(1-production.T):
        hh += ax.plot(tt, p, "-", label=i)

    if obs is not None:
        for i, y in enumerate(1-obs.T):
            ax.plot(tt, y, "*", c=hh[i].get_color())

    ax.legend(title="Prod.\nwell #.")
    ax.set_ylabel("Oil saturation (rel. production)")
    ax.set_xlabel("Time index")
    ax.set_ylim(-0.01, 1.01)
    return hh


def productions(fignum, water_prod_series, title=""):
    """Plot production series, including ensembles. 1 well/axes."""
    nAx   = len(model.producers)
    ncols = 4
    nrows = int(np.ceil(nAx/ncols))
    fig, axs = freshfig(fignum, figsize=(16, 7),
                        ncols=ncols, nrows=nrows, sharex=True, sharey=True)

    # Plot properties
    props = dict(
        color  = dict(default="k",  pri="C0", ES="C1", EnKS="C2", ES_dir="C6"),
        alpha  = dict(default=1.0,  pri=.20, ES=.20, EnKS=.20, ES_dir=.20),
        lw     = dict(default=0.5,  tru=2.0),
        ls     = dict(default="-",  obs=""),
        marker = dict(default="",   obs="*"),
    )

    def pprop(label):
        """Fuzzy lookup of property."""
        out = dict(label=label)
        for a in props:
            dct = props[a]
            out[a] = dct["default"]
            for key in dct:
                if key.lower() in label.lower():
                    out[a] = dct[key]
        return out

    # For each well
    for i, ax in enumerate(axs.ravel()):

        if i == 0:
            fig.suptitle("Oil saturations -- " + title)
        if i >= nAx:
            ax.set_visible(False)
            continue

        # Well number
        ax.text(1, 1, f"Well #{i}", ha="right", va="top", transform=ax.transAxes)

        # Plot
        for label, series in water_prod_series.items():
            ll = ax.plot(1 - series.T[i].T, **pprop(label))
            plt.setp(ll[1:], label="_nolegend_")

        # Legend
        if i == nAx-1:
            leg = ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
            for ln in leg.get_lines():
                ln.set(alpha=1, linewidth=1)


def oilfields(fignum, water_sat_fields, label="", **kwargs):
    fig, axs = freshfig(fignum, nrows=3, ncols=4, sharex=True, sharey=True)

    for i, (ax, water_sat) in enumerate(zip(axs.ravel(), water_sat_fields[label])):
        ax.text(0, .85*Dy, str(i), c="w", size=12)

        cc = oilfield(ax, water_sat, xticks=[0, Dx], yticks=[0, Dy], **kwargs)

    fig.suptitle(f"Oil saturation (some realizations) - {label}")
    fig_colorbar(fig, cc)


def oilfield_means(fignum, water_sat_fields, title=""):
    ncols = 2
    nAx   = len(water_sat_fields)
    nrows = int(np.ceil(nAx/ncols))
    fig, axs = freshfig(fignum, figsize=(8, 4*nrows),
                        ncols=ncols, nrows=nrows, sharex=True, sharey=True)

    fig.subplots_adjust(hspace=.3)
    fig.suptitle(f"Oil saturation (mean fields) - {title}")
    for i, (ax, label) in enumerate(zip(axs.ravel(), water_sat_fields)):

        field = water_sat_fields[label]
        if field.ndim == 2:
            field = field.mean(axis=0)

        handle = oilfield(ax, field, title=label)

    fig_colorbar(fig, handle)


def correlation_fields(fignum, field_ensembles, xy_coord, title=""):
    field_ensembles = {k: v for k, v in field_ensembles.items() if v.ndim == 2}

    ncols = 2
    nAx   = len(field_ensembles)
    nrows = int(np.ceil(nAx/ncols))
    fig, axs = freshfig(fignum, figsize=(8, 4*nrows),
                        ncols=ncols, nrows=nrows, sharex=True, sharey=True)

    fig.subplots_adjust(hspace=.3)
    fig.suptitle(title)
    for i, ax in enumerate(axs.ravel()):

        if i >= nAx:
            ax.set_visible(False)
        else:
            label  = list(field_ensembles)[i]
            field  = field_ensembles[label]
            handle = corr_field_vs(ax, field, xy_coord, label)

    fig_colorbar(fig, handle, ticks=[-1, -0.4, 0, 0.4, 1])


def animate1(saturation, production, pause=200):
    fig, axs = freshfig(19, ncols=2, nrows=2, figsize=(12, 10))
    if is_notebook_or_qt:
        plt.close()  # ttps://stackoverflow.com/q/47138023

    tt = 1+np.arange(len(saturation))

    axs[0, 0].set_title("Oil saturation (Initial)")
    axs[0, 0].cc = oilfield(axs[0, 0], saturation[0])

    axs[0, 1].set_title("Oil saturation")
    axs[0, 1].cc = oilfield(axs[0, 1], saturation[-1])
    well_scatter(axs[0, 1], model.injectors)
    well_scatter(axs[0, 1], model.producers, False)

    axs[1, 0].set_title("Saturation (production)")
    axs[1, 0].set(ylim=(0, 1))
    prod_handles = production1(axs[1, 0], production)
    axs[1, 0].legend(loc="upper right", title="Well num.")
    axs[1, 0].set(ylabel=None)

    axs[1, 1].set_title("Saturation (production)")
    scat_handles = axs[1, 1].scatter(
        *model.producers.T[:2], 24**2, 1-production[-1],
        marker="^", clip_on=False, cmap=cm_ow, vmin=0, vmax=1)
    axs[1, 1].set(xlim=(0, Dx), ylim=(0, Dy))

    fig.tight_layout()
    fig_colorbar(fig, axs[0, 0].cc)

    def animate(iT):
        for c in axs[0, 1].cc.collections:
            try:
                axs[0, 1].collections.remove(c)
            except ValueError:
                pass  # occurs when re-running script
        axs[0, 1].cc = oilfield(axs[0, 1], saturation[iT])

        for h, p in zip(prod_handles, 1-production.T):
            h.set_data(tt[:iT], p[:iT])

        scat_handles.set_array(1-production[iT])

    from matplotlib import animation
    ani = animation.FuncAnimation(
        fig, animate, len(tt), blit=False, interval=pause)

    return ani


def hists(fignum, samples, xlabel=""):
    fig, ax = freshfig(fignum, figsize=(8, 4))
    for label in samples:
        sample = samples[label].ravel()
        ax.hist(sample, density=True, alpha=0.4, bins=20, label=label)
    ax.legend()
    ax.set(xlabel=xlabel, ylabel=(
        "Rel. frequency\n"
        "(over all members and all cells)"
    ))
