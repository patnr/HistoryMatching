"""Plot functions for reservoir model.

Note: before using any function, you must set the module vairable `model`.

Snippet to reload module:

    >>> import importlib
    >>> import simulator.plotting as m
    >>> m = importlib.reload(m)
    >>> plotting.model = model
    >>> plotting.coord_type = "absolute"
"""

import copy
import warnings

import ipywidgets as wg
import matplotlib as mpl
import numpy as np
import struct_tools
from IPython.display import clear_output, display
from matplotlib import pyplot as plt
from mpl_tools import place, place_ax
from mpl_tools.misc import axprops, nRowCol
from struct_tools import DotDict as Dict

# The "self"
model = None

# Axes limits
coord_type = "relative"   # ==> (0, 1)  x (0, 1)
# coord_type = "absolute" # ==> (0, Lx) x (0, Ly)
# coord_type = "index"    # ==> (0, Ny) x (0, Ny)

# Colormap for saturation
lin_cm = mpl.colors.LinearSegmentedColormap.from_list
cm_ow = lin_cm("", [(0, "#1d9e97"), (.3, "#b2e0dc"), (1, "#f48974")])
# cOil, cWater  = "red", "blue"  # Plain
# cOil, cWater = "#d8345f", "#01a9b4"  # Pastel/neon
# cOil, cWater = "#e58a8a", "#086972"  # Pastel
# ccnvrt = lambda c: np.array(mpl.colors.colorConverter.to_rgb(c))
# cMiddle = .3*ccnvrt(cWater) + .7*ccnvrt(cOil)
# cm_ow = lin_cm("", [cWater, cMiddle, cOil])

# Colormap for correlations
cmap_corr = plt.cm.get_cmap("bwr")
# Set out-of-bounds colors for correlation plot
cmap_corr = copy.copy(cmap_corr)  # avoid warning
cmap_corr.set_under("green")
cmap_corr.set_over("orange")
cmap_corr.set_bad("black")

# Defaults
styles = dict(
    pperm = dict(
        title  = "Pre-Perm",
        cmap   = "jet",
    ),
    oil = dict(
        title  = "Oil saturation",
        transf = lambda x: 1 - x,
        cmap   = cm_ow,
        levels = np.linspace(0 - 1e-7, 1 + 1e-7, 20),
        cticks = np.linspace(0, 1, 6),
    ),
    corr = dict(
        title  = "Correlations",
        cmap   = cmap_corr,
        levels = np.linspace(-1.00001, 1.00001, 20),
        cticks = np.linspace(-1, 1, 6),
    ),
)


def _field(ax, Z, wells=False, argmax=False, colorbar=False,
           cmap="viridis", cticks=None, levels=10, transf=lambda x: x,  # style kwargs
           **kwargs):
    """Contour-plot of the (flat) field `Z`."""
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
    Z = transf(Z)
    Z = Z.reshape(model.shape).T

    # Did we bother to specify set_over/set_under/set_bad ?
    has_out_of_range = getattr(cmap, "_rgba_over", None) is not None

    # ax.imshow(Z[::-1])
    collections = ax.contourf(
        Z, levels, cmap=cmap, **kwargs,
        extend="both" if has_out_of_range else "neither",
        # Using origin="lower" puts the points in the gridcell centers.
        # This is great (agrees with finite-volume point definition)
        # but means that the plot wont extend all the way to the edges, which is ugly.
        # We could make it pretty by padding Z with its border values,
        # and specifying X = [0, .5, 1.5, 2.5, ...] (and likewise for Y).
        # But, instead, we simply stretch the field, using the following:
        origin=None, extent=(0, Lx, 0, Ly),
        )

    # Contourf does not plot (at all) the bad regions. "Fake it" by facecolor
    if has_out_of_range:
        ax.set_facecolor(getattr(cmap, "_rgba_bad", "w"))

    # Axis lims & labels
    ax.set_xlim((0, Lx))
    ax.set_ylim((0, Ly))
    ax.set_aspect("equal")
    if "abs" in coord_type:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    else:
        ax.set_xlabel(f"x ({coord_type})")
        ax.set_ylabel(f"y ({coord_type})")

    # Add well markers
    if wells:
        if wells == "color":
            c = [f"C{i}" for i in range(len(model.producers))]
        else:
            c = None
        well_scatter(ax, model.injectors)
        well_scatter(ax, model.producers, False, color=c)

    # Add argmax marker
    if argmax:
        idx = Z.T.argmax()  # reverse above transpose
        ax.plot(*model.ind2xy_stretched(idx), "y*", ms=15, label="max", zorder=98)
        ax.plot(*model.ind2xy_stretched(idx), "k*", ms=4 , label="max", zorder=99)

    # Add colorbar
    if colorbar:
        if isinstance(colorbar, type(ax)):
            cax = dict(cax=colorbar)
        else:
            cax = dict(ax=ax, shrink=.8)
        ax.figure.colorbar(collections, **cax, ticks=cticks)

    return collections


def field(ax, Z, style=None, **kwargs):
    """Contour-plot of the (flat) field `Z`."""
    if style:
        for key, val in styles[style].items():
            if key != "title":
                kwargs.setdefault(key, val)
    return _field(ax, Z, **kwargs)


def fields(ZZ, style, title="", figsize=(1.7, 1),
           label_color="k", colorbar=True, **kwargs):
    """Do `field(Z) for `Z in ZZ`."""
    # Create figure using freshfig
    title = dash_join("Fields", styles[style]["title"], title)
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
        suptitle = dash_join(suptitle, f"First {len(axs)} instances")
    # Re-set suptitle (since it got cleared above)
    if suptitle:
        fig.suptitle(suptitle)

    if colorbar:
        fig.colorbar(hh[0], cax=axs.cbar_axes[0],
                     ticks=kwargs.pop("cticks", styles[style]["cticks"]))

    return fig, axs, hh


def captured_fig(output, num, **kwargs):
    """Create decorator that provides `fig, ax` for use in IPywidget layouts.

    Including the output of mpl figures in the widget **layout** is quite difficult.
    Especially cross-compatibility local/Colab (which this approach provides!).

    ## Brief intro

    - Use `w.interact` for basic functionality
        - Use explicit controls like `w.IntSlider` if you wish to specify
          - `orientation`
          - `description`
          - `continuous_update`
            Also see: `{'manual': True}` and `w.interact_manual`
    - Use `w.interactive` to delay display.
        - Allows delay/reuse-ing resulting widgets,
          and accessing the data bound to the UI controls.
        - Display with `IPython.display.display`.
        - Inspect the resulting widget's `.children` to find
          the controls and (lastly) the `w.Output`, which contains
          stdout, stderr, mpl figures (see note below),
          and which allows CSS styling (like borders, etc).
    - Use `w.interactive_output` to avoid generating the control widgets,
      but still linking the controls to the function
      (PS: I found that using `with w.Output` worked better).
      Allows specifying layout (`VBox`, `HBox`, `AppLayout`, `GridspecLayout`)
      properly, without hacks like modifying it after creation e.g.
      https://stackoverflow.com/q/52980565 .
    - Another way to link is to use the `.observe` attr of widgets.

    ## Cautions

    - In order to include an mpl figure in a ipywidget **layout**,
      we must capture its output; it is essential that
      the **figure creation** and `plt.show()` is done therein.
      Treatment differs from inline to interactive backends.
      For example, using `with w.Output` and creating the figure thereunder
      seems to necessitate using `IPython.display.clear_output` when `inline`.
    - `tight_layout` must render. Better to use `constrained_layout`?

    ## Refs

    None of these quite worked on Colab or my Mac, but were useful:

    - Use of `fig.canvas.flush_events()` and `fig.canvas.draw()`:
      From https://stackoverflow.com/a/58561439
    - Similar to the docs, but better:
      https://coderzcolumn.com/tutorials/python/interactive-widgets-in-jupyter-notebook-using-ipywidgets
    - Fancy widget layout:
      https://medium.com/kapernikov/ipywidgets-with-matplotlib-93646718eb84
        - Uses ipympl (doesn't display on my mac)
        - When testing on Colab (`inline` backend) the layout works,
          except that the figure is placed below, not on the side.
    - Side-by-side figures with interactivity
      https://github.com/matplotlib/ipympl/issues/203#issuecomment-600500051

    Example for use in a notebook:
    >>> output = wg.Output()
    ... @captured_fig(output, "Title", figsize=(1.2, 1), rel=True)
    ... def plot(fig, ax, _newfig, x, y):
    ...     A = np.arange(10)
    ...     X, Y = np.meshgrid(A, A)
    ...     X = x*X
    ...     Y = y*Y
    ...     h = ax.imshow(X + Y)
    ...
    ... xy0 = 1, 1
    ... sx = wg.IntSlider(xy0[0], 0, 10)
    ... sy = wg.IntSlider(xy0[1], 0, 10,
    ...                  orientation='vertical', continuous_update=False)
    ... linked = wg.interactive(plot, x=sx, y=sy)
    ... widgets = wg.VBox([wg.HBox([output, sy]), sx])
    ... display(widgets)
    ... plot(*xy0)
    """
    backend = mpl.get_backend()
    inline_ish = "inline" in backend or "ipympl" in backend

    def fig_ax(num):
        """Create fig, axs. Deserving of particular attention, so factored out."""
        # Figure creation
        # Of course, for *interactive* mpl backends, this should only be run once.
        # But running it from inside f (with appropriate checks for single execution)
        # causes blank figure => Run outside of f().
        # However, using `ipywidgets.Output` to capture output requires that it runs
        # inside f. In this case it actually seems to work though (no blank figures).
        if inline_ish:
            # Rm previous (static) image. Necssary when using `ipywidgets.Output`
            # Use `wait=True` because to avoid flickering, ref ipywidgets/issues/1582
            clear_output(wait=True)
        else:
            # Check for existance, otherwise the first time it is run
            # (no error is thrown but) duplicate figures are created
            # (no longer seems to be an issue, but the check doesn't hurt)
            if plt.fignum_exists(num):
                # Fix issue: figure doesn't display **when cell is re-run**.
                # I think it's related to being in an ipython widget, but can also
                # be fixed by changing num (so that freshfig creates a new one).
                plt.close(num)
        fig, axs = place.freshfig(num, ipympl_show=False, **kwargs)
        return fig, axs

    def decorator(f):
        """The actual decorator."""
        fig, axs = None, None

        def new(*args, **kwargs):
            # Persistent figure (re-used after slider updates)
            nonlocal fig, axs

            with output:
                if inline_ish or fig is None:
                    fig, axs = fig_ax(num)
                    newfig = True
                else:
                    newfig = False
                    try:
                        axs.clear()
                    except AttributeError:
                        for ax in axs.ravel():
                            ax.clear()

                # Main
                f(fig, axs, newfig, *args, **kwargs)

                if not inline_ish:
                    # From https://stackoverflow.com/a/58561439
                    fig.canvas.flush_events()
                    fig.canvas.draw()
                plt.show()

        return new
    return decorator


def field_console(compute, style, title="", figsize=(1.5, 1), **kwargs):
    """Field computed on-the-fly controlled by interactive sliders."""
    title  = dash_join(styles[style]["title"], title)
    ctrls  = compute.controls.copy()  # gets modified
    output = wg.Output()

    @captured_fig(output, title, figsize=figsize, rel=True)
    def plot(fig, ax, newfig, **kw):
        # Ignore warnings due to computing and plotting contour/nan
        with warnings.catch_warnings(), \
                np.errstate(divide="ignore", invalid="ignore"):
            warnings.filterwarnings(
                "ignore", category=UserWarning, module="matplotlib.contour")

            Z = compute(**kw)
            field(ax, Z, style, colorbar=newfig, **kwargs)
            if newfig:
                fig.tight_layout()

        # Add crosshairs
        if "x" in kw and "y" in kw:
            x, y = model.sub2xy_stretched(kw["x"], kw["y"])
            d = dict(c="k", ls="--", lw=1)
            ax.axhline(y, **d)
            ax.axvline(x, **d)

    # Make widget/interactive plot
    linked = wg.interactive(plot, **ctrls)
    *ww, _ = linked.children

    # Adjust control styles
    for w in ww:
        if "Slider" in str(type(w)):
            w.continuous_update = False  # => faster
        elif "Dropdown" in str(type(w)):
            w.layout.width = 'max-content'

    # Layout
    try:
        layout = layout1(ww, output)
    except (ValueError, IndexError):
        # Fallback
        cpanel = wg.VBox(ww, layout=dict(align_items='center'))
        layout = wg.HBox([output, cpanel])

    # Display
    display(layout)
    plot(**{w.description: w.value for w in ww})


def layout1(ww, output):
    """Compose a layout.

    ```
    -----------------
     cN | cF  | cP
        | cFt | cPt
    -----------------
        output  | cY
    -----------------
          cX
    -----------------
    ```
    """
    cN, cF, cFt, cP, cPt, cX, cY = ww

    try:
        import google.colab  # type: ignore # noqa
        isColab = True
    except ImportError:
        isColab = False

    # Adjust control styles
    for w in ww:
        w.style.description_width = "max-content"
        if "Slider" in str(type(w)):
            w.layout.width = "16em"
            if w.description == "x":
                w.layout.width = "100%"
                # top right bottom left
                w.layout.padding = "0 84px 0 30px" if isColab else "0 146px 0 65px"
            elif w.description == "y":
                w.orientation = "vertical"
                w.layout.width = "2em"
                w.layout.height = "100%"
                w.layout.padding = "0" if isColab else "63px 0 72px 0"

    # Compose layout
    # PS: Use flexboxes (scale automatically, unlike AppLayout, TwoByTwoLayout)
    V, H = wg.VBox, wg.HBox
    cF = V([cF, cFt])
    cP = V([cP, cPt])
    # Centering inside another set of boxes is for safety/fallback
    center = {"justify_content": "space-around"}
    cX = H([cX], layout=center)
    cY = V([cY], layout=center)
    # hspace = H([], layout={"width": "50px"})
    cH = H([cN, cF, cP], layout={"justify_content": "space-between"})
    layout = H([V([cH, H([output, cY]), cX])])

    return layout


def well_scatter(ax, ww, inj=True, text=None, color=None):
    """Scatter-plot the wells in `ww`."""
    # Well coordinates, stretched for plotting (ref plotting.fields)
    ww = model.sub2xy_stretched(*model.xy2sub(*ww.T[:2])).T
    # NB: make sure ww array data is not overwritten (avoid in-place)
    if   "rel" in coord_type: s = 1/model.Lx, 1/model.Ly                   # noqa
    elif "abs" in coord_type: s = 1, 1                                     # noqa
    elif "ind" in coord_type: s = model.Nx/model.Lx, model.Ny/model.Ly     # noqa
    else: raise ValueError("Unsupported coordinate type: %s" % coord_type) # noqa
    ww = ww * s

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
    sh = ax.scatter(*ww.T[:2], s=26**2, c=c, marker=m, ec=ec,
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


def ens_style(label, N=100):
    """Line styling for ensemble production plots."""
    style = Dict(
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
        style.c      = "C5"
        style.alpha  = .3
    if label == "LES":
        style.c      = "C4"
        style.alpha  = .3
    if label == "ILES":
        style.c      = "C8"
        style.alpha  = .3

    # Incrase alpha if N is small
    style.alpha **= (1 + np.log10(N/100))

    return style


# NOTE: This uses IPython/jupyter widgets. Another solution, using interactive
# mpl backends (=> not available on Colab), can be found in mpl_tools.
def toggle_items(wrapped):
    """Include checkboxes/checkmarks to toggle plotted data series on/off."""
    def new(*args, **kwargs):
        plotter, kw_subplots = wrapped(*args, **kwargs)
        checkmarks = {label: True for label in args[0]}

        # To disable the interactivity, simply uncomment following line.
        # (if figure doesn't show, plt.close() it first, or change its title)
        # plotter(*place.freshfig(**kw_subplots), None, **checkmarks); return

        output = wg.Output()
        plot = captured_fig(output, **kw_subplots)(plotter)

        linked = wg.interactive(plot, **checkmarks)

        # Adjust layout
        *ww, _ = linked.children
        for w in ww:
            # Narrower checkmark boxes (incl. text)
            w.layout.width = "100%"  # or "auto"
            w.indent = False
            # Alternative method:
            # w.style.description_width = "0", and use
            # VBox(ww, layout=Layout(width="12ex"))

            # Color borders (background/face is impossible, see refs) of checkmarks
            # - https://stackoverflow.com/a/54896280/
            # - https://github.com/jupyter-widgets/ipywidgets/issues/710
            c = ens_style(w.description, N=1)['c']
            c = mpl.colors.to_hex(c, keep_alpha=False)
            w.layout.border = "solid 7px" + c

            # Center vertically inside boxes
            w.layout.align_items = "center"
            w.layout.padding = ".9em 0 .7em 0"

        cpanel = wg.VBox(ww)
        layout = wg.HBox([output, cpanel])
        display(layout)
        plot(**{w.description: w.value for w in ww})
    return new


@toggle_items
def productions(dct, title="", figsize=(1.5, 1), nProd=None):
    """Production time series with data toggler. 1 axes/well. Ensemble compatible."""
    title = dash_join("Production profiles", title)

    if nProd is None:
        nProd = struct_tools.get0(dct).shape[1]
        nProd = min(23, nProd)

    kw_subplots = dict(num=title, figsize=figsize, rel=True,
                       **nRowCol(nProd), sharex=True, sharey=True)

    def plot(fig, axs, _newfig, **labels):
        axs = axs.ravel()

        # Turn off redundant axes
        for ax in axs[nProd:]:
            ax.set_visible(False)

        # For each axis/well
        for i in range(nProd):
            ax = axs[i]
            label_ax(ax, f"Well {i}", x=.99, ha="right")

            for label, series in dct.items():
                if not labels[label]:
                    continue

                # Get style props
                some_ensemble = list(dct.values())[-1]
                props = ens_style(label, N=len(some_ensemble))

                # Plot
                ll = ax.plot(1 - series.T[i], **props)  # noqa

                # Rm duplicate labels
                # plt.setp(ll[1:], label="_nolegend_")

    return plot, kw_subplots


def production1(ax, production, obs=None, legend_outside=True):
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
    if legend_outside:
        kws = dict(
              bbox_to_anchor=(1, 1),
              loc="upper left",
              ncol=1+len(production.T)//10,
        )
    else:
        kws = dict(loc="lower left")
    ax.legend(title="Well #.", **kws)

    ax.set_ylabel("Production (saturations)")
    ax.set_xlabel("Time index")
    # ax.set_ylim(-0.01, 1.01)
    ax.axhline(0, c="xkcd:light grey", ls="--", zorder=1.8)
    ax.axhline(1, c="xkcd:light grey", ls="--", zorder=1.8)
    return hh


# Note: See note in mpl_setup.py about properly displaying the animation.
def anim(key, *dcts, figsize=(2.0, .7), pause=200, animate=True, **kwargs):
    perm, wsats, prod = [d[key] for d in dcts]  # unpack

    # Create figure and axes
    title = dash_join("Animation", key)
    fig, (ax1, ax2) = place.freshfig(title, ncols=2, figsize=figsize, rel=True)
    fig.suptitle(title)  # coz animation never (any backend) displays title
    # Saturations
    ax2.cc = field(ax2, wsats[-1], "oil", wells="color", colorbar=True, **kwargs)
    # Production
    hh = production1(ax1, prod, legend_outside=False)

    if animate:
        from matplotlib import animation
        tt = np.arange(len(wsats))

        def update_fig(iT):
            # Update field
            for c in ax2.cc.collections:
                try:
                    ax2.collections.remove(c)
                except ValueError:
                    pass  # occurs when re-running script
            ax2.cc = field(ax2, wsats[iT], "oil", **kwargs)

            # Update production lines
            if iT >= 1:
                for h, p in zip(hh, 1-prod.T):
                    h.set_data(tt[:iT-1], p[:iT-1])

        ani = animation.FuncAnimation(
            fig, update_fig, len(tt), blit=False, interval=pause,
            # Prevent busy/idle indicator constantly flashing, despite %%capture
            # and even manually clearing the output of the calling cell.
            repeat=False,  # flashing stops once the (unshown) animation finishes.
            # An alternative solution is to do this in the next cell:
            # animation.event_source.stop()
            # but it does not work if using "run all", even with time.sleep(1).
        )

        return ani


def spectrum(ydata, title="", figsize=(1.6, .7), semilogy=False, **kwargs):
    """Plotter specialized for spectra."""
    title = dash_join("Spectrum", title)
    fig, ax = place.freshfig(title, figsize=figsize, rel=True)
    if semilogy:
        h = ax.semilogy(ydata)
    else:
        h = ax.loglog(ydata)
    ax.grid(True, "both", axis="both")
    ax.set(xlabel="eigenvalue index", ylabel="variance")
    fig.tight_layout()
    return h


def dash_join(*txts):
    """Join non-empty txts by a dash."""
    return " -- ".join([t for t in txts if t != ""])


def label_ax(ax, txt, x=.01, y=.99, ha="left", va="top",
             c="k", fontsize="large", bbox=None):
    if bbox is None:
        bbox = dict(edgecolor="w", facecolor="w", alpha=.4,
                    boxstyle="round,pad=0")
    return ax.text(x, y, txt, c=c, fontsize=fontsize,
                   ha=ha, va=va, transform=ax.transAxes, bbox=bbox)
