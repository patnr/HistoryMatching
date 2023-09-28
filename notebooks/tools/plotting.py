"""Plot functions building on `simulator.plotting`."""

import copy
import warnings

import ipywidgets as wg
import matplotlib as mpl
import mpl_tools
import numpy as np
import struct_tools
from IPython.display import clear_output, display
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator, LogLocator
from mpl_tools.place import freshfig
from mpl_tools.misc import nRowCol
from struct_tools import DotDict as Dict
from TPFA_ResSim.plotting import styles

from .utils import mnorm

# Colormap for correlations
cmap_corr = plt.get_cmap("bwr")
# Set out-of-bounds colors for correlation plot
cmap_corr = copy.copy(cmap_corr)  # avoid warning
cmap_corr.set_under("green")
cmap_corr.set_over("orange")
cmap_corr.set_bad("black")

# Add some styles
styles["pperm"] = dict(
    title  = "Pre-Perm",
    levels = np.linspace(-4, 4, 21),
    cticks = np.arange(-4, 4+1),
    cmap   = "jet",
)
styles["perm"] = dict(
    title  = "Perm",
    locator=LogLocator(),
)
styles["corr"] = dict(
    title  = "Correlations",
    cmap   = cmap_corr,
    levels = np.linspace(-1.00001, 1.00001, 20),
    cticks = np.linspace(-1, 1, 6),
)
styles["NPV"] = dict(
    title  = "NPV",
    cmap   = plt.get_cmap("inferno"),
)
styles["domain"] = dict(
    title  = "Model domain",
    cmap   = plt.get_cmap("inferno"),
    vmin   = 99,
    vmax   = 99,
)


def fields(model, Zs, style, title="", figsize=(1.7, 1), cticks=None,
           label_color="k", wells=False, colorbar=True, **kwargs):
    """Do `model.plt_field(Z) for Z in Zs`."""

    # Create figure using freshfig
    title = dash_join("Fields", styles[style]["title"], title)
    fig, axs = freshfig(title, figsize=figsize, rel=True)
    # Store suptitle (exists if mpl is inline) coz gets cleared below
    try:
        suptitle = fig._suptitle.get_text()
    except AttributeError:
        suptitle = ""
    # Create axes using AxesGrid
    fig.clear()
    from mpl_toolkits.axes_grid1 import AxesGrid
    axs = AxesGrid(fig, 111,
                   nrows_ncols=nRowCol(min(12, len(Zs))).values(),
                   cbar_mode='single', cbar_location='right',
                   share_all=True,
                   axes_pad=0.2,
                   cbar_pad=0.1)
    # Turn off redundant axes
    for ax in axs[len(Zs):]:
        ax.set_visible(False)

    # Convert (potential) list-like Zs into dict
    if not isinstance(Zs, dict):
        Zs = dict(enumerate(Zs))

    # Plot
    hh = []
    for ax, label in zip(axs, Zs):
        label_ax(ax, label, c=label_color)
        hh.append(model.plt_field(ax, Zs[label], style, wells=wells,
                                  colorbar=False, title=None, **kwargs))

    # Suptitle
    if len(Zs) > len(axs):
        suptitle = dash_join(suptitle, f"First {len(axs)} instances")
    # Re-set suptitle (since it got cleared above)
    if suptitle:
        fig.suptitle(suptitle)

    if colorbar:
        if not cticks:
            cticks = styles[style].get("cticks", styles["default"]["cticks"])
        fig.colorbar(hh[0], cax=axs.cbar_axes[0], ticks=cticks)

    warnings.filterwarnings("ignore", category=UserWarning)
    fig.tight_layout()  # Not necessary with ipympl
    # warnings.resetwarnings()  # Don't! Causes warnings from (mpl?) libraries

    return fig, axs, hh


def init():
    """Configure mpl.

    ## On the choice of backend:

    - In scripts, `Qt5Agg` is nice coz it (is interactive and) allows
      programmatic (automatic) placement of figures on screen.
    - In notebooks `%matplotib notebook` (nbAgg) is interactive,
      so cooler than `%matplotlib inline`.
    - On my local machine, `%matplotlib inline` always makes figures display
      as string: <Figure size ...>, despite `plt.show()`, `plt.pause(0.1)`, etc.
    - However, `%matplotlib widget/ipympl` ('module://ipympl.backend_nbagg')
      is also interactive, and compatible with BOTH jupyter-notebook and -lab.
      It may also be selected via `import ipympl`.
    - Colab only supports `%matplotlib inline` (but could look into "plotly")
      https://stackoverflow.com/a/64297121

    ## On IPython magics vs `mpl.use()`:

    The magics (like `%matplotlib notebook/ipympl`) set `plt.ion()` and some rcParams.
    They could probably be used in a module via `get_ipython().run_line_magic()`
    however, here I choose to instead use `mpl.use(...)` or `import ipympl`.
    - If you do `import ipympl` BEFORE `import matplotlib.pyplot` then
      the figures only display as the text string <Figure size ...>.
    - If you do mpl.use("nbAgg") BEFORE `import matplotlib.pyplot` then
      you must remember to do `plt.ion()` too.
      PS: this "interactive" is not to be confused with "interactive backends".
      PS: check status with `plt.isinteractive`.

    ## About figures not displaying on 2nd run (cell execution):

    This seems to be an issue with ipympl when the figure `num` is re-used.
    Should be fixed in `freshfig` as of mpl-tools 0.2.55.

    ## About "run all (cells)":

    On my Mac, the figures sometimes don't display, or are "inline" instead of "nbAgg".

    ## About animation displaying twice:

    The solution seems to be to split the creation and display cells, and using %%capture.
    This worked both locally and on Colab.
    - [Ref](https://stackoverflow.com/q/47138023)
    - [Ref](https://stackoverflow.com/a/36685236)
    On my Mac, `%matplotlib inline` did not have this issue, but plenty others (see above).
    However, on Colab (i.e. `%matplotlib inline`), the animation still displayed double.
    """
    if mpl_tools.is_notebook_or_qt:
        mpl.rc('animation', html="jshtml")

        # mpl.rcParams["figure.figsize"] = [5, 3.5]
        # NB: Non-default figsize/fontsize may cause axis labels/titles
        # that do not fit within the figure, or trespass into the axes
        # (unless fixed by tight_layout, but that is sometimes not possible)
        # Moreover, in general, for presentations, you will use the web-browser
        # zoom functionality, which also zooms in figure and font sizes,
        # reducing the need to change defaults.
        mpl.rcParams.update({"legend.fontsize": "large"})
        mpl.rcParams["font.size"] = 12

        try:
            # Colab
            import google.colab  # type: ignore # noqa

            # [colab-specific adjustments]

        except ImportError:
            # Local Jupyter
            try:
                # Similar to `%matplotlib widget/ipympl`
                # Equivalently: mpl.use('module://ipympl.backend_nbagg')
                import ipympl  # noqa
                # pass  # revert to inline

            except ImportError:
                # Similar to `%matplotlib notebook`.
                mpl.use("nbAgg")

    else:
        # Script run
        mpl.rcParams.update({'font.size': 10})
        try:
            mpl.use("Qt5Agg")
        except ImportError:
            pass  # fall back to e.g. MacOS backend
    plt.ion()


def captured_fig(output, num, **kwargs):
    """Decorator that provides `fig, ax` for use in Jupyter (IPywidget) *layouts*.

    In general, I advise to create dashboards with custom layouts using `ìnteractive`:
    The source code `ipywidgets/widgets/interaction.py` is fairly readable!
    - Like `interact()`, it creates control widgets from simple kwargs.
    - Like `interactive_output()`, it delays `display()` until manually called.

    Example:

    >>> linked = interactive()
    ... *ww, out = linked.children
    ... dashboard = HBox([ww[0], out, ww[1]])
    ... display(dashboard)

    BUT, there is some trickery about making figures actually show up.
    Especially making it work simultaneously on local/Colab.

    - The approach taken here uses `with w.Output`.
    - In `DA-tutorials`, I found you can make the figures appear on Colab
      also initially (which was a problem) by using `linked.update()`.

    ## Cautions

    - **figure creation** (done by this function) and `plt.show()` must be in callback,
      i.e. part of the interactively called function.
    - `tight_layout` must render. Better to use `constrained_layout`?

    ## Refs

    Main ref: <https://github.com/jupyter-widgets/ipywidgets/issues/3352>
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
        fig, axs = freshfig(num, ipympl_show=False, **kwargs)
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


def field_console(model, compute, style, title="", figsize=(1.5, 1), rel=True, **kwargs):
    """Field computed on-the-fly controlled by interactive sliders."""
    title  = dash_join(styles[style]["title"], title)
    ctrls  = compute.controls.copy()  # gets modified
    output = wg.Output()

    @captured_fig(output, title, figsize=figsize, rel=rel)
    def plot(fig, ax, newfig, **kw):
        # Ignore warnings due to computing and plotting contour/nan
        with warnings.catch_warnings(), \
                np.errstate(divide="ignore", invalid="ignore"):
            warnings.filterwarnings(
                "ignore", category=UserWarning, module="matplotlib.contour")

            Z = compute(**kw)
            model.plt_field(ax, Z, style, colorbar=newfig, **kwargs)
            if newfig:
                fig.tight_layout()

        # Add crosshairs
        if "x" in kw and "y" in kw:
            x, y = model.sub2xy(kw["x"], kw["y"])
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
        cpanel = wg.VBox(ww, layout=dict(align_items='center',
                                         justify_content='center'))
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

    There are many interesting attributes and CSS possibilities.
    - https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20List.html
    - https://stackoverflow.com/q/52980565 .
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
        # plotter(*freshfig(**kw_subplots), None, **checkmarks); return

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


def spectrum(ydata, title="", figsize=(1.6, .7), semilogy=False, **kwargs):
    """Plotter specialized for spectra."""
    title = dash_join("Spectrum", title)
    fig, ax = freshfig(title, figsize=figsize, rel=True)
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
        bbox = dict(edgecolor="w", facecolor="w", alpha=.5,
                    boxstyle="round,pad=0")
    return ax.text(x, y, txt, c=c, fontsize=fontsize,
                   ha=ha, va=va, transform=ax.transAxes, bbox=bbox)


def figure12(title="", *args, figsize=(10, 3.5), **kwargs):
    """Call `freshfig`. Add axes laid out with 1 panel on right, two on left."""
    title = dash_join("Optim. trajectories", title)
    fig, _ax = freshfig(title, *args, figsize=figsize, **kwargs)
    _ax.remove()
    gs = GridSpec(2, 10)
    ax0 = fig.add_subplot(gs[:, :7])
    ax1 = fig.add_subplot(gs[0, 7:])
    ax2 = fig.add_subplot(gs[1, 7:])
    for ax in (ax1, ax2):
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
    return fig, (ax0, ax1, ax2)


def add_path12(ax1, ax2, ax3, path, objs=None, color=None, labels=True):
    """Plot 2d path in `ax1`, `objs` in `ax2`, step size magnitude to `ax3`."""
    # Path line
    ax1.plot(*path.T[:2], c=color or "g")
    # Path scatter and text
    if len(path) > 1:
        ii = set(np.logspace(-1e-9, np.log10(len(path))-1e-9, 15, dtype=int))
    else:
        ii = [0]
    cm = plt.get_cmap('viridis_r')(np.linspace(0.0, 1, len(ii)))
    for k, c in zip(ii, cm):
        if color:
            c = color
        xy = path[k][:2]
        if labels:
            ax1.text(*xy, k, c=c)
        ax1.scatter(*xy, s=4**2, color=c, zorder=5)

    if objs is not None:
        # Objective values
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.set_ylabel('Obj')
        ax2.plot(objs, color=color or 'C0', marker='.', ms=3**2)
        # ax2.set_ylim(-objs.max(), -objs.min())
        ax2.grid()

        # Step magnitudes
        ax3.sharex(ax2)
        ax2.tick_params(labelbottom=False)
        xx = np.arange(len(path)-1) + .5
        yy = mnorm(np.diff(path, axis=0), -1)
        ax3.plot(xx, yy, color=color or "C1", marker='.', ms=3**2)
        ax3.set_ylabel('|Step|')
        ax3.grid()
        ax3.set(xlabel="itr")
