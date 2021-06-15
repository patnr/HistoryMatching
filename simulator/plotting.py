"""Plot functions for reservoir model"""

import IPython.display as ip_disp
import matplotlib as mpl
import numpy as np
from ipywidgets import HBox, VBox, interactive
from matplotlib import pyplot as plt
from mpl_tools import fig_layout, is_notebook_or_qt
from mpl_tools.misc import axprops, fig_colorbar, nRowCol
from struct_tools import DotDict, get0


def center(E):
    return E - E.mean(axis=0)


COORD_TYPE = "relative"


def lims(self):
    if "rel" in COORD_TYPE:
        Lx, Ly = 1, 1
    elif "abs" in COORD_TYPE:
        Lx, Ly = self.Lx, self.Ly
    elif "ind" in COORD_TYPE:
        Lx, Ly = self.Nx, self.Ny
    else:
        raise ValueError("Unsupported coordinate type: %s" % COORD_TYPE)
    return Lx, Ly


def field(self, ax, zz, **kwargs):
    """Contour-plot the field contained in `zz`."""
    # Need to transpose coz model assumes shape (Nx, Ny),
    # and contour() uses the same orientation as array printing.
    Z = zz.reshape(self.shape).T

    Lx, Ly = lims(self)

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
        origin="lower", extent=(0, Lx, 0, Ly))

    ax.set_xlim((0, Lx))
    ax.set_ylim((0, Ly))

    ss = ax.get_subplotspec()
    if ss.is_first_col():
        ax.set_ylabel("y")
    if ss.is_last_row():
        ax.set_xlabel("x")

    return collections


def fields(self,
           fignum, plotter, ZZ,
           figsize=None,
           title="",
           txt_color="k",
           colorbar=True,
           **kwargs):

    fig, axs = fig_layout.freshfig(
        fignum, figsize=figsize,
        **nRowCol(min(12, len(ZZ))),
        sharex=True, sharey=True)

    # Turn off redundant axes
    for ax in axs[len(ZZ):]:
        ax.set_visible(False)

    # Convert list-like ZZ into dict
    if not isinstance(ZZ, dict):
        ZZ = {i: Z for (i, Z) in enumerate(ZZ)}

    # Get min/max across all fields
    flat = np.array(list(ZZ.values())).ravel()
    vmin = flat.min()
    vmax = flat.max()

    hh = []
    for ax, label in zip(axs.ravel(), ZZ):

        ax.text(0, 1, label, ha="left", va="top",
                c=txt_color, transform=ax.transAxes)

        # Call plotter
        hh.append(plotter(self, ax, ZZ[label],
                          vmin=vmin, vmax=vmax, **kwargs))

    if colorbar:
        fig_colorbar(fig, hh[0])

    if title:
        fig.suptitle(title)

    return fig, axs, hh


def oilfields(self, fignum, water_sat_fields, **kwargs):
    return fields(self, fignum, oilfield, **kwargs)


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


def oilfield(self, ax, ss, **kwargs):
    levels = np.linspace(0 - 1e-7, 1 + 1e-7, 11)
    return field(self, ax, 1-ss, levels=levels, cmap=cm_ow, **kwargs)


def corr_field(self, ax, A, b, title="", **kwargs):
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
    cc = field(self, ax, corrs, levels=np.linspace(-1, 1, 11),
               cmap=mpl.cm.bwr, **kwargs)
    return cc


def corr_field_vs(self, ax, E, xy, title="", **kwargs):
    i = self.xy2ind(*xy)
    b = E[:, i]
    cc = corr_field(self, ax, E, b, title, **kwargs)
    ax.plot(*xy, '*k', ms=4)
    return cc


def scale_well_geometry(self, ww):
    """Wells use absolute scaling. Scale to coord_type instead."""
    ww = ww.copy()  # dont overwrite
    if "rel" in COORD_TYPE:
        s = 1/self.Lx, 1/self.Ly
    elif "abs" in COORD_TYPE:
        s = 1, 1
    elif "ind" in COORD_TYPE:
        s = self.Nx/self.Lx, self.Ny/self.Ly
    else:
        raise ValueError("Unsupported coordinate type: %s" % COORD_TYPE)
    ww[:, :2] = ww[:, :2] * s
    return ww


def well_scatter(self, ax, ww, inj=True, text=True, color=None):
    ww = scale_well_geometry(self, ww)

    # Style
    if inj:
        c = "w"
        d = "k"
        m = "v"
    else:
        c = "k"
        d = "w"
        m = "^"

    if color:
        c = color

    # Markers
    # sh = ax.plot(*ww.T[:2], m+c, ms=16, mec="k", clip_on=False)
    sh = ax.scatter(*ww.T[:2], s=16**2, c=c, marker=m,
                    edgecolors="k",
                    clip_on=False,
                    zorder=1.5,  # required on Jupypter
                    )

    # Text labels
    if text:
        if not inj:
            ww.T[1] -= 0.01
        for i, w in enumerate(ww):
            ax.text(*w[:2], i, color=d,
                    ha="center", va="center")

    return sh


def production1(ax, production, obs=None):
    hh = []
    tt = 1+np.arange(len(production))
    for i, p in enumerate(1-production.T):
        hh += ax.plot(tt, p, "-", label=i)

    if obs is not None:
        for i, y in enumerate(1-obs.T):
            ax.plot(tt, y, "*", c=hh[i].get_color())

    ax.legend(title="Well #.",
              loc="upper left",
              bbox_to_anchor=(1, 1),
              ncol=1+len(production.T)//10)
    ax.set_ylabel("Oil saturation (rel. production)")
    ax.set_xlabel("Time index")
    # ax.set_ylim(-0.01, 1.01)
    ax.axhline(0, c="xkcd:light grey", ls="--", zorder=1.8)
    ax.axhline(1, c="xkcd:light grey", ls="--", zorder=1.8)
    plt.pause(.1)
    return hh


def style(label, N=100):
    """Line styling."""
    style = DotDict(
        label=label,
        c="k", alpha=1.0, lw=0.5,
        ls="-", marker="", ms=4,
    )
    if label == "Truth":
        style.lw     = 2
        style.zorder = 2.1
    if label == "Noisy":
        style.label = "Obs"
        style.ls     = ""
        style.marker = "*"
    if label == "Prior":
        style.c      = "C0"
        style.alpha  = .3
    if label == "ES":
        # style.label = "Ens. Smooth."
        style.c      = "C1"
        style.alpha  = .3
    if label == "ES0":
        style.c      = "C2"
        style.alpha  = .3
        style.zorder = 1.9
    if label == "iES":
        style.c      = "C4"
        style.alpha  = .3

    # Incrase alpha if N is small
    style.alpha **= (1 + np.log10(N/100))

    return style


def toggle_series(plotter):
    """Include checkboxes/checkmarks to toggle plotted data series on/off."""
    # NB: this was pretty darn complicated to get working
    # with the right layout and avoiding double plotting.
    # So exercise great caution when changing it!

    def interactive_plot(*args, **kwargs):
        dct, *args = args  # arg0 must be dict of line data to plot
        kwargs["legend"] = False  # Turn off legend

        handles = []

        def plot_these(**labels):
            included = {k: v for k, v in dct.items() if labels[k]}
            hh = plotter(included, *args, **kwargs)
            if not handles:
                handles.extend(hh)

        widget = interactive(plot_these, **{label: True for label in dct})
        widget.update()
        # Could end function here. The rest is adjustments.

        # Place checkmarks to the right
        *checkmarks, figure = widget.children
        widget = HBox([figure, VBox(checkmarks)])
        ip_disp.display(widget)

        # Narrower checkmark boxes
        widget.children[1].layout.width = "15ex"
        for CX in widget.children[1].children:
            CX.layout.width = '10ex'
            CX.style.description_width = '0ex'

        # Hack to color borders of checkmarks.
        # Did not find how to color background/face.
        # Anyways, there is no general/good way to style widgets, ref:
        # https://github.com/jupyter-widgets/ipywidgets/issues/710#issuecomment-409448282
        import matplotlib as mpl
        for cm, lh in zip(checkmarks, handles):
            c = mpl.colors.to_hex(lh.get_color(), keep_alpha=False)
            cm.layout.border = "solid 5px" + c

        return widget

    return interactive_plot


# TODO: implement with plotting.fields ?
@toggle_series
def productions(dct, fignum, figsize=None, title="", nProd=None, legend=True):

    if nProd is None:
        nProd = get0(dct).shape[1]
        nProd = min(23, nProd)
    fig, axs = fig_layout.freshfig(
        fignum, figsize=figsize,
        **nRowCol(nProd), sharex=True, sharey=True)
    # fig.suptitle("Oil productions " + title)

    # Turn off redundant axes
    for ax in axs.ravel()[nProd:]:
        ax.set_visible(False)

    handles = []

    # For each well
    for i in range(nProd):
        ax = axs.ravel()[i]
        ax.text(1, 1, f"Well {i}" if i == 0 else i, c="k",
                ha="right", va="top", transform=ax.transAxes)

        for label, series in dct.items():

            # Get style props
            some_ensemble = list(dct.values())[-1]
            props = style(label, N=len(some_ensemble))

            # Plot
            ll = ax.plot(1 - series.T[i], **props)

            # Rm duplicate labels
            plt.setp(ll[1:], label="_nolegend_")

            # Store 1 handle of series
            if i == 0:
                handles.append(ll[0])

        # Legend
        if legend:
            leg = ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
            for ln in leg.get_lines():
                ln.set(alpha=1, linewidth=max(1, ln.get_linewidth()))

    return handles


def oilfield_means(self, fignum, water_sat_fields, title="", **kwargs):
    ncols = 2
    nAx   = len(water_sat_fields)
    nrows = int(np.ceil(nAx/ncols))
    fig, axs = fig_layout.freshfig(fignum, figsize=(8, 4*nrows),
                                   ncols=ncols, nrows=nrows, sharex=True, sharey=True)

    fig.subplots_adjust(hspace=.3)
    fig.suptitle(f"Oil saturation (mean fields) - {title}")
    for ax, label in zip(axs.ravel(), water_sat_fields):

        field = water_sat_fields[label]
        if field.ndim == 2:
            field = field.mean(axis=0)

        handle = oilfield(self, ax, field, title=label, **kwargs)

    fig_colorbar(fig, handle)  # type: ignore


def correlation_fields(self, fignum, field_ensembles, xy_coord, title="", **kwargs):
    field_ensembles = {k: v for k, v in field_ensembles.items() if v.ndim == 2}

    ncols = 2
    nAx   = len(field_ensembles)
    nrows = int(np.ceil(nAx/ncols))
    fig, axs = fig_layout.freshfig(
        fignum, figsize=(8, 4*nrows),
        ncols=ncols, nrows=nrows, sharex=True, sharey=True)

    fig.subplots_adjust(hspace=.3)
    fig.suptitle(title)
    for i, ax in enumerate(axs.ravel()):

        if i >= nAx:
            ax.set_visible(False)
        else:
            label  = list(field_ensembles)[i]
            field  = field_ensembles[label]
            handle = corr_field_vs(self, ax, field, xy_coord, label, **kwargs)

    fig_colorbar(fig, handle, ticks=[-1, -0.4, 0, 0.4, 1])  # type: ignore


# TODO: Use nrows=1 ?
def dashboard(self, saturation, production,
              pause=200, animate=True, title="", **kwargs):
    fig, axs = fig_layout.freshfig(231, ncols=2, nrows=2, figsize=(12, 10))
    if is_notebook_or_qt:
        plt.close()  # ttps://stackoverflow.com/q/47138023

    tt = np.arange(len(saturation))

    axs[0, 0].set_title("Initial")
    axs[0, 0].cc = oilfield(self, axs[0, 0], saturation[0], **kwargs)
    axs[0, 0].set_ylabel(f"y ({COORD_TYPE})")

    axs[0, 1].set_title("Evolution")
    axs[0, 1].cc = oilfield(self, axs[0, 1], saturation[-1], **kwargs)
    well_scatter(self, axs[0, 1], self.injectors)
    well_scatter(self, axs[0, 1], self.producers, False,
                 color=[f"C{i}" for i in range(len(self.producers))])

    axs[1, 0].set_title("Production")
    prod_handles = production1(axs[1, 0], production)

    axs[1, 1].set_visible(False)

    # fig.tight_layout()
    fig_colorbar(fig, axs[0, 0].cc)

    if title:
        fig.suptitle(f"Oil saturation -- {title}")

    if animate:
        from matplotlib import animation

        def update_fig(iT):
            # Update field
            for c in axs[0, 1].cc.collections:
                try:
                    axs[0, 1].collections.remove(c)
                except ValueError:
                    pass  # occurs when re-running script
            axs[0, 1].cc = oilfield(self, axs[0, 1], saturation[iT], **kwargs)

            # Update production lines
            if iT >= 1:
                for h, p in zip(prod_handles, 1-production.T):
                    h.set_data(tt[:iT-1], p[:iT-1])

        ani = animation.FuncAnimation(
            fig, update_fig, len(tt), blit=False, interval=pause)

        return ani
