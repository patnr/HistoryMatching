"""Plot functions for reservoir model.

Note: before using any function, you must set the module vairable `model`.
"""

# TODO: unify (nRowCol, turn off, ax.text, etc) for
#       fields() and productions() ?

import IPython.display as ip_disp
import matplotlib as mpl
import numpy as np
from ipywidgets import HBox, VBox, interactive
from matplotlib import pyplot as plt
from mpl_tools import place, place_ax
from mpl_tools.misc import axprops, nRowCol
from struct_tools import DotDict, get0

_is_inline = "inline" in mpl.get_backend()


def dash(*txts):
    """Join non-empty txts by a dash."""
    return " -- ".join([t for t in txts if t != ""])


def field(ax, zz, **kwargs):
    """Contour-plot the field contained in `zz`."""
    levels     = kwargs.pop("levels"    , field.levels)
    cmap       = kwargs.pop("cmap"      , field.cmap)
    coord_type = kwargs.pop("coord_type", field.coord_type)

    ax.set(**axprops(kwargs))

    # Plotting with extent=(0, Lx, 0, Ly), rather than merely changing ticks
    # has the advantage that set_aspect("equal") yields correct axes size,
    # and that mouse hovering (with interactive backends) reports correct pos.
    # Disadvantage: well_scatter must also account for coord_type.
    if "rel" in coord_type:
        Lx, Ly = 1, 1
    elif "abs" in coord_type:
        Lx, Ly = model.Lx, model.Ly
    elif "ind" in coord_type:
        Lx, Ly = model.Nx, model.Ny
    else:
        raise ValueError(f"Unsupported coord_type: {coord_type}")

    # Need to transpose coz model assumes shape (Nx, Ny),
    # and contour() uses the same orientation as array printing.
    Z = zz.reshape(model.shape).T

    # ax.imshow(Z[::-1])
    collections = ax.contourf(
        Z, levels, cmap=cmap, **kwargs,
        # Using origin="lower" puts the points in the gridcell centers.
        # This is great (agrees with finite-volume point definition)
        # but means that the plot wont extend all the way to the edges,
        # which can only be circumvented by manuallly padding.
        # Using `origin=None` stretches the field to the edges, which
        # might be slightly erroneous compared with finite-volume defs.
        # However, it is the definition that agrees with line and scatter
        # plots (e.g. well_scatter), and that correspondence is more important.
        origin=None, extent=(0, Lx, 0, Ly))

    ax.set_xlim((0, Lx))
    ax.set_ylim((0, Ly))
    ax.set_aspect("equal")
    if "abs" in coord_type:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    else:
        ax.set_xlabel(f"x ({field.coord_type})")
        ax.set_ylabel(f"y ({field.coord_type})")

    return collections


# "Self"
model = None


# Defaults
field.coord_type = "relative"
field.cmap = "jet"
field.levels = 10
field.ticks = None
# Use a list of levels for more control, including vmin/vmax.
# Note that providing vmin/vmax (and not a levels list) to mpl
# yields prettier colobar ticks, but destorys the consistency
# of the colorbars from one figure to another.


def fields(plotter, ZZ,
           title="",
           figsize=(1.7, 1),
           txt_color="k",
           colorbar=True,
           **kwargs):

    # Get plotter defaults
    title = dash("Fields", getattr(plotter, "title", ""), title)
    ticks = getattr(plotter, "ticks", None)

    # Setup figure
    fig, axs = place.freshfig(title, figsize=figsize, rel=True)
    fig.clear()
    from mpl_toolkits.axes_grid1 import AxesGrid
    axs = AxesGrid(fig, 111,
                   nrows_ncols=nRowCol(min(12, len(ZZ))).values(),
                   cbar_mode='single', cbar_location='right',
                   share_all=True,
                   axes_pad=0.1,
                   cbar_pad=0.1)
    # Turn off redundant axes
    for ax in axs[len(ZZ):]:
        ax.set_visible(False)

    # Convert (potential) list-like ZZ into dict
    if not isinstance(ZZ, dict):
        ZZ = {i: Z for (i, Z) in enumerate(ZZ)}

    hh = []
    for ax, label in zip(axs, ZZ):

        # Label axes
        ax.text(0, 1, label, c=txt_color, fontsize="large",
                ha="left", va="top", transform=ax.transAxes)

        # Call plotter
        hh.append(plotter(ax, ZZ[label], **kwargs))

    # suptitle
    suptitle = ""
    print("A", suptitle)
    if len(ZZ) > len(axs):
        suptitle += f"First {len(axs)} instances"
        print("B", suptitle)
    pre_existing = fig._suptitle
    print("C", pre_existing)
    if pre_existing:
        print("D", pre_existing)
        suptitle = dash(pre_existing.get_text(), suptitle)
        print("E", pre_existing)
    if suptitle:
        fig.suptitle(suptitle)
        print("F", pre_existing)

    if colorbar:
        fig.colorbar(hh[0], cax=axs.cbar_axes[0], ticks=ticks)

    return fig, axs, hh


# Colormap for saturation
lin_cm = mpl.colors.LinearSegmentedColormap.from_list
# def ccnvrt(c): return np.array(mpl.colors.colorConverter.to_rgb(c))


# Plain:
# cOil   = "red"
# cWater = "blue"
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


def oilfield(ax, wsat, **kwargs):
    lvls = np.linspace(0 - 1e-7, 1 + 1e-7, 20)
    return field(ax, 1-wsat, levels=lvls, cmap=cm_ow, **kwargs)
oilfield.title = "Oil saturation"  # noqa
oilfield.ticks = np.linspace(0, 1, 6)


def corr_field(ax, corr, **kwargs):
    lvls = np.linspace(-1, 1, 20)
    return field(ax, corr, levels=lvls, cmap="bwr", **kwargs)
corr_field.title = "Correlations"  # noqa
corr_field.ticks = np.linspace(-1, 1, 6)


def scale_well_geometry(ww):
    """Wells use absolute scaling. Scale to `field.coord_type` instead."""
    ww = ww.copy()  # dont overwrite
    if "rel" in field.coord_type:
        s = 1/model.Lx, 1/model.Ly
    elif "abs" in field.coord_type:
        s = 1, 1
    elif "ind" in field.coord_type:
        s = model.Nx/model.Lx, model.Ny/model.Ly
    else:
        raise ValueError("Unsupported coordinate type: %s" % field.coord_type)
    ww[:, :2] = ww[:, :2] * s
    return ww


def well_scatter(ax, ww, inj=True, text=None, color=None):
    ww = scale_well_geometry(ww)

    # Style
    if inj:
        c  = "w"
        ec = "gray"
        d  = "k"
        m  = "v"
    else:
        c  = "k"
        ec = "gray"
        d  = "w"
        m  = "^"

    if color:
        c = color

    # Markers
    # sh = ax.plot(*ww.T[:2], m+c, ms=16, mec="k", clip_on=False)
    sh = ax.scatter(*ww.T[:2], s=16**2, c=c, marker=m, ec=ec,
                    clip_on=False,
                    zorder=1.5,  # required on Jupypter
                    )

    # Text labels
    if text != False:
        if not inj:
            ww.T[1] -= 0.01
        for i, w in enumerate(ww):
            ax.text(*w[:2], i if text is None else text, color=d, fontsize="large",
                    ha="center", va="center")

    return sh


def production1(ax, production, obs=None):
    """Production time series. Multiple wells in 1 axes => not ensemble compat."""
    hh = []
    tt = 1+np.arange(len(production))
    for i, p in enumerate(1-production.T):
        hh += ax.plot(tt, p, "-", label=i)

    if obs is not None:
        for i, y in enumerate(1-obs.T):
            ax.plot(tt, y, "*", c=hh[i].get_color())

    # Add legend
    place_ax.adjust_position(ax, w=-0.05)
    ax.legend(title="Well #.",
              bbox_to_anchor=(1, 1),
              loc="upper left",
              ncol=1+len(production.T)//10)

    ax.set_ylabel("Production (saturations)")
    ax.set_xlabel("Time index")
    # ax.set_ylim(-0.01, 1.01)
    ax.axhline(0, c="xkcd:light grey", ls="--", zorder=1.8)
    ax.axhline(1, c="xkcd:light grey", ls="--", zorder=1.8)
    return hh


def ens_style(label, N=100):
    """Line styling for ensemble production plots."""
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
    if label == "IES":
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


@toggle_series
def productions2(dct, title="", figsize=(2, 1), nProd=None, legend=True):

    if nProd is None:
        nProd = get0(dct).shape[1]
        nProd = min(23, nProd)
    title = dash("Production profiles", title)
    fig, axs = place.freshfig(
        title, figsize=figsize, rel=True,
        **nRowCol(nProd), sharex=True, sharey=True)

    # Turn off redundant axes
    for ax in axs.ravel()[nProd:]:
        ax.set_visible(False)

    handles = []

    # For each well
    for i in range(nProd):
        ax = axs.ravel()[i]
        ax.text(1, 1, f"Well {i}", c="k", fontsize="large",
                ha="right", va="top", transform=ax.transAxes)

        for label, series in dct.items():

            # Get style props
            some_ensemble = list(dct.values())[-1]
            props = ens_style(label, N=len(some_ensemble))

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


def toggler(plotter):
    """Include checkboxes/checkmarks to toggle plotted data series on/off."""
    def new(*args, **kwargs):
        update = plotter(*args, **kwargs)
        arg0 = args[0]
        widget = interactive(update, **{label: True for label in arg0})
        widget.update()

        # Could end function now. The following styles the checkboxes.
        *checkmarks, figure = widget.children

        # Place checkmarks to the right -- only works with mpl inline?
        widget = HBox([figure, VBox(checkmarks)])
        try:
            import google.colab  # noqa
            ip_disp.display(widget)
        except ImportError:
            pass

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
        for cm, lbl in zip(checkmarks, arg0):
            c = ens_style(lbl, N=1)['c']
            c = mpl.colors.to_hex(c, keep_alpha=False)
            cm.layout.border = "solid 5px" + c
        return widget
    return new


@toggler
def productions(dct, title="", figsize=(2, 1), nProd=None):

    if nProd is None:
        nProd = get0(dct).shape[1]
        nProd = min(23, nProd)
    title = dash("Production profiles", title)
    fig, axs = place.freshfig(
        title, figsize=figsize, rel=True,
        **nRowCol(nProd), sharex=True, sharey=True)

    # Turn off redundant axes
    for ax in axs.ravel()[nProd:]:
        ax.set_visible(False)

    def update(**labels):
        # For each well
        for iWell in range(nProd):
            ax = axs.ravel()[iWell]
            ax.clear()
            ax.text(1, 1, f"Well {iWell}", c="k", fontsize="large",
                    ha="right", va="top", transform=ax.transAxes)

            for label, series in dct.items():
                if not labels[label]:
                    continue

                # Get style props
                some_ensemble = list(dct.values())[-1]
                props = ens_style(label, N=len(some_ensemble))

                # Plot
                ll = ax.plot(1 - series.T[iWell], **props)

                # Rm duplicate labels
                plt.setp(ll[1:], label="_nolegend_")

    return update


# Note: See note in mpl_setup.py about properly displaying the animation.
def dashboard(key, *dcts, figsize=(2.0, 1.3), pause=200, animate=True, **kwargs):
    perm, wsats, prod = [d[key] for d in dcts]  # unpack

    # Create figure and axes
    title = dash("Dashboard", key)
    # NB: constrained_layout seems to put too much space between axes.
    # Could be remedied by configuring h_pad, w_pad?
    fig = plt.figure(num=title,
                     figsize=place.relative_figsize(figsize))
    fig.clear()
    fig.suptitle(title)  # coz animation never (any backend) displays title
    gs = fig.add_gridspec(100, 100)
    w, h, p, c = 45, 45, 3, 3
    ax11 = fig.add_subplot(gs[:+h, :w])
    ax12 = fig.add_subplot(gs[:+h, -w-p-c:-p-c])
    ax21 = fig.add_subplot(gs[-h:, :w]         , sharex=ax12, sharey=ax12)
    ax22 = fig.add_subplot(gs[-h:, -w-p-c:-p-c], sharex=ax12, sharey=ax12)
    # Colorbars
    ax12c = fig.add_subplot(gs[:+h, -c:])
    ax22c = fig.add_subplot(gs[-h:, -c:])

    # Perm
    ax12.cc = field(ax12, perm, **kwargs)
    fig.colorbar(ax12.cc, ax12c, ticks=field.ticks)
    ax12c.set_ylabel("Permeability")

    # Saturation0
    ax21.cc = oilfield(ax21, wsats[+0], **kwargs)
    # Saturations
    ax22.cc = oilfield(ax22, wsats[-1], **kwargs)
    ax21.text(.01, .99, "Initial", c="w", fontsize="x-large",
              ha="left", va="top", transform=ax21.transAxes,
              bbox=dict(edgecolor="w", facecolor="k", alpha=.15,
                        boxstyle="round,pad=0"))
    # Add wells
    well_scatter(ax22, model.injectors)
    well_scatter(ax22, model.producers, False,
                 color=[f"C{i}" for i in range(len(model.producers))])
    fig.colorbar(ax22.cc, ax22c, ticks=oilfield.ticks)
    ax22c.set_ylabel(oilfield.title)

    # Production
    hh = production1(ax11, prod)
    ax11.set_xlabel(ax11.get_xlabel(), labelpad=-5)

    ax12.yaxis.set_tick_params(labelleft=False)
    ax22.yaxis.set_tick_params(labelleft=False)
    ax12.xaxis.set_tick_params(labelbottom=False)
    ax12.set_ylabel(None)
    ax22.set_ylabel(None)
    ax12.set_xlabel(None)
    # ax12c.yaxis.set_label_position('left')
    # ax11.yaxis.set_ticks_position("right")

    if animate:
        from matplotlib import animation
        tt = np.arange(len(wsats))

        def update_fig(iT):
            # Update field
            for c in ax22.cc.collections:
                try:
                    ax22.collections.remove(c)
                except ValueError:
                    pass  # occurs when re-running script
            ax22.cc = oilfield(ax22, wsats[iT], **kwargs)
            ax22.set_ylabel(None)

            # Update production lines
            if iT >= 1:
                for h, p in zip(hh, 1-prod.T):
                    h.set_data(tt[:iT-1], p[:iT-1])

        ani = animation.FuncAnimation(
            fig, update_fig, len(tt), blit=False, interval=pause)

        return ani
