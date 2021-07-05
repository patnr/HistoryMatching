"""Plot functions for reservoir model.

Note: before using any function, you must set the module vairable `model`.
"""

# TODO: unify (nRowCol, turn off, ax.text, etc) for
#       fields() and productions() ?
# TODO: Should simply check for inline, not colab?


import warnings

import IPython.display as ip_disp
import matplotlib as mpl
import numpy as np
from ipywidgets import HBox, VBox, interactive
from matplotlib import pyplot as plt
from mpl_tools import is_inline, place, place_ax
from mpl_tools.misc import axprops, nRowCol
from struct_tools import DotDict, get0

# Module "self"
model = None

coord_type = "relative"

# Colormap for saturation
lin_cm = mpl.colors.LinearSegmentedColormap.from_list
cm_ow = lin_cm("", [(0, "#1d9e97"), (.3, "#b2e0dc"), (1, "#f48974")])
# cOil, cWater  = "red", "blue"  # Plain
# cOil, cWater = "#d8345f", "#01a9b4"  # Pastel/neon
# cOil, cWater = "#e58a8a", "#086972"  # Pastel
# ccnvrt = lambda c: np.array(mpl.colors.colorConverter.to_rgb(c))
# cMiddle = .3*ccnvrt(cWater) + .7*ccnvrt(cOil)
# cm_ow = lin_cm("", [cWater, cMiddle, cOil])

# Defaults
styles = dict(
    default = dict(
        title  = "",
        transf = lambda x: x,
        cmap   = "viridis",
        levels = 10,
        ticks  = None,
        # Note that providing vmin/vmax (and not a levels list) to mpl
        # yields prettier colobar ticks, but destorys the consistency
        # of the colorbars from one figure to another.
    ),
    pperm = dict(
        title  = "Pre-Perm.",
        cmap   = "jet",
    ),
    oil = dict(
        title  = "Oil saturation",
        transf = lambda x: 1 - x,
        cmap   = cm_ow,
        levels = np.linspace(0 - 1e-7, 1 + 1e-7, 20),
        ticks  = np.linspace(0, 1, 6),
    ),
    corr = dict(
        title  = "Correlations",
        cmap   = "bwr",
        levels = np.linspace(-1, 1, 20),
        ticks  = np.linspace(-1, 1, 6),
    ),
)


def pop_style_with_fallback(key, style, kwargs):
    """`kwargs.pop(key)`, defaulting to `styles[style or "default"]`."""
    x = styles["default"][key]
    x = styles[style or "default"].get(key, x)
    x = kwargs.pop(key, x)
    return x


def field(ax, Z, style=None, wells=False, argmax=False, colorbar=False, **kwargs):
    """Contour-plot of the (flat) field `Z`."""
    kw = lambda k: pop_style_with_fallback(k, style, kwargs)
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
    Z = kw("transf")(Z)
    Z = Z.reshape(model.shape).T

    # ax.imshow(Z[::-1])
    collections = ax.contourf(
        Z, kw("levels"), cmap=kw("cmap"), **kwargs,
        # Using origin="lower" puts the points in the gridcell centers.
        # This is great (agrees with finite-volume point definition)
        # but means that the plot wont extend all the way to the edges,
        # which can only be circumvented by manuallly padding.
        # Using `origin=None` stretches the field to the edges, which
        # might be slightly erroneous compared with finite-volume defs.
        # However, it is the definition that agrees with line and scatter
        # plots (e.g. well_scatter), and that correspondence is more important.
        origin=None, extent=(0, Lx, 0, Ly))

    if wells:
        if wells == "color":
            c = [f"C{i}" for i in range(len(model.producers))]
        else:
            c = None
        well_scatter(ax, model.injectors)
        well_scatter(ax, model.producers, False, color=c)

    if argmax:
        idx = Z.T.argmax()  # reverse above transpose
        ax.plot(*model.ind2xy(idx), "g*", ms=12, label="Max")

    ax.set_xlim((0, Lx))
    ax.set_ylim((0, Ly))
    ax.set_aspect("equal")
    if "abs" in coord_type:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    else:
        ax.set_xlabel(f"x ({coord_type})")
        ax.set_ylabel(f"y ({coord_type})")
    # Add colorbar
    if colorbar:
        if isinstance(colorbar, type(ax)):
            cax = dict(cax=colorbar)
        else:
            cax = dict(ax=ax)
        ax.figure.colorbar(collections, **cax, ticks=kw("ticks"))

    return collections


def fields(ZZ, style=None, title="", figsize=(1.7, 1),
           label_color="k", colorbar=True, **kwargs):
    """Do `field(Z)` for each `Z` in `ZZ`."""
    kw = lambda k: pop_style_with_fallback(k, style, kwargs)

    # Create figure using freshfig
    title = dash("Fields", kw("title"), title)
    fig, axs = place.freshfig(title, figsize=figsize, rel=True)
    # Store suptitle (exists if mpl is inline) coz gets cleared below
    try:
        suptitle = fig._suptitle.get_text()
    except AttributeError:
        suptitle = ""
    # Create axes using AxesGrid
    fig.clear()
    from mpl_toolkits.axes_grid1 import AxesGrid
    axs = AxesGrid(fig, 111,
                   nrows_ncols=nRowCol(min(12, len(ZZ))).values(),
                   cbar_mode='single', cbar_location='right',
                   share_all=True,
                   axes_pad=0.2,
                   cbar_pad=0.1)
    # Turn off redundant axes
    for ax in axs[len(ZZ):]:
        ax.set_visible(False)

    # Convert (potential) list-like ZZ into dict
    if not isinstance(ZZ, dict):
        ZZ = {i: Z for (i, Z) in enumerate(ZZ)}

    hh = []
    for ax, label in zip(axs, ZZ):
        label_ax(ax, label, c=label_color)
        hh.append(field(ax, ZZ[label], style, **kwargs))

    # Suptitle
    if len(ZZ) > len(axs):
        suptitle = dash(suptitle, f"First {len(axs)} instances")
    # Re-set suptitle (since it got cleared above)
    if suptitle:
        fig.suptitle(suptitle)

    if colorbar:
        fig.colorbar(hh[0], cax=axs.cbar_axes[0], ticks=kw("ticks"))

    return fig, axs, hh


def field_interact(compute, style=None, title="", **kwargs):
    """Field computed on-the-fly controlled by interactive sliders."""
    kw = lambda k: pop_style_with_fallback(k, style, kwargs)

    # Init figure (provides full-(sup)title, axes layout)
    # NB: This should only be run once for interactive mpl backends.
    # Moreover, putting it inside of the widget-wrapped function (update)
    # (including checks to make sure it only runs once) causes the issue that
    # the figure goes blank after moving the slider a few times.
    # Another issue is that if the cell containing the figure is closed, or
    # even sometimes if re-running the cell, the figure won't display again.
    # I think this is also related because it's inside an ipython widget.
    # It seems that changing the figure label/title/number is sufficient to fix it.
    def fig_ax():
        fig, ax = place.freshfig(dash(kw("title"), title),
                                 figsize=(1.5, 1), rel=True)
        did_init = True
        return fig, ax, did_init

    if not is_inline():
        fig, ax, did_init = fig_ax()

    def update(**controls):
        """Update plot(s)."""
        nonlocal fig, ax, did_init

        # Ignore warnings due to computing and plotting contour/nan
        with warnings.catch_warnings(), \
                np.errstate(divide="ignore", invalid="ignore"):
            warnings.filterwarnings(
                "ignore", category=UserWarning, module="matplotlib.contour")

            Z = compute(**controls)

            # (Re-)init
            if is_inline():
                fig, ax, did_init = fig_ax()
            else:
                ax.clear()

            # Update
            field(ax, Z, style, colorbar=did_init, **kwargs)

            if did_init:
                did_init = False

            # Add crosshairs
            try:
                # A stretched version [like field()] of model.sub2xy
                x = model.Lx * controls["iX"]/(model.Nx-1)
                y = model.Ly * controls["iY"]/(model.Ny-1)
                ax.plot(x, y, "k+", ms=15, markeredgewidth=2)
            except KeyError:
                pass

    # Make widget/interactive plot
    return interactive(update, **compute.controls)


def scale_well_geometry(ww):
    """Wells use absolute scaling. Scale to `coord_type` instead."""
    ww = ww.copy()  # dont overwrite
    if "rel" in coord_type:
        s = 1/model.Lx, 1/model.Ly
    elif "abs" in coord_type:
        s = 1, 1
    elif "ind" in coord_type:
        s = model.Nx/model.Lx, model.Ny/model.Ly
    else:
        raise ValueError("Unsupported coordinate type: %s" % coord_type)
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
            ax.text(*w[:2], i if text is None else text,
                    color=d, fontsize="large", ha="center", va="center")

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
        label_ax(ax, f"Well {i}", x=.99, ha="right")

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
            import google.colab  # type: ignore # noqa
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
    ax12.cc = field(ax12, perm, "pperm", colorbar=ax12c, **kwargs)
    ax12c.set_ylabel("Permeability")

    # Saturation0
    ax21.cc = field(ax21, wsats[+0], "oil", **kwargs)
    # Saturations
    ax22.cc = field(ax22, wsats[-1], "oil", wells="color", colorbar=ax22c, **kwargs)
    label_ax(ax21, "Initial", c="k", fontsize="x-large")
    # Add wells
    ax22c.set_ylabel(styles["oil"]["title"])

    # Production
    hh = production1(ax11, prod)
    ax11.set_xlabel(ax11.get_xlabel(), labelpad=-10)

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
            ax22.cc = field(ax22, wsats[iT], "oil", **kwargs)
            ax22.set_ylabel(None)

            # Update production lines
            if iT >= 1:
                for h, p in zip(hh, 1-prod.T):
                    h.set_data(tt[:iT-1], p[:iT-1])

        ani = animation.FuncAnimation(
            fig, update_fig, len(tt), blit=False, interval=pause)

        return ani


def dash(*txts):
    """Join non-empty txts by a dash."""
    return " -- ".join([t for t in txts if t != ""])


def label_ax(ax, txt, x=.01, y=.99, ha="left", va="top",
             c="k", fontsize="large", bbox=None):
    if bbox is None:
        bbox = dict(edgecolor="w", facecolor="w", alpha=.4,
                    boxstyle="round,pad=0")
    return ax.text(x, y, txt, c=c, fontsize=fontsize,
                   ha=ha, va=va, transform=ax.transAxes, bbox=bbox)
