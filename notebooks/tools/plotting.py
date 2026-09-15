"""Plot functions building on `simulator.plotting`."""

import copy
import warnings
from math import sqrt

import ipywidgets as wg
import matplotlib as mpl
import numpy as np
import struct_tools
from IPython import get_ipython
from IPython.display import display
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator, LogLocator
from struct_tools import DotDict as Dict
from minires.plotting import styles

# Colormap for correlations
cmap_corr = plt.get_cmap("bwr")
# Set out-of-bounds colors for correlation plot
cmap_corr = copy.copy(cmap_corr)  # avoid warning
cmap_corr.set_under("green")
cmap_corr.set_over("orange")
cmap_corr.set_bad("black")

# Add some styles
styles["pperm"] = dict(
    title="Pre-Perm",
    levels=np.linspace(-4, 4, 21),
    cticks=np.arange(-4, 4 + 1),
    cmap="jet",
)
styles["perm"] = dict(
    title="Perm",
    locator=LogLocator(),
)
styles["corr"] = dict(
    title="Correlations",
    cmap=cmap_corr,
    levels=np.linspace(-1.00001, 1.00001, 20),
    cticks=np.linspace(-1, 1, 6),
)
styles["NPV"] = dict(
    title="NPV",
    cmap=plt.get_cmap("inferno"),
)
styles["domain"] = dict(
    title="Model domain",
    cmap=plt.get_cmap("inferno"),
    vmin=99,
    vmax=99,
)


show = plt.show  # noqa  # type: ignore
from matplotlib.colors import LogNorm, AsinhNorm  # noqa  # type: ignore


# The following 3 helpers used to come from `mpl-tools` (also by patnr).
# They were vendored (and reduced to what this repo uses) so that the tutorials
# depend only on stable, public matplotlib/IPython API. This matters because
# on Colab the base environment is outside our control (see pyproject.toml).


def freshfig(num=None, figsize=None, sup=True, **kwargs):
    """Like `plt.subplots(num=num, figsize=figsize, **kwargs)`, but re-using figures.

    If a figure labelled `num` (a descriptive string, please) already exists,
    it is *cleared*, not closed and re-opened. Thus its window keeps its
    position and size on screen, which is what you want when re-running
    the `.py` mirror of a notebook with `%run` in IPython (GUI backend).

    The inline (Jupyter/Colab) backend does not show the figure label,
    so there (if `sup`) the label is set as `fig.suptitle` instead.

    Unlike `mpl_tools.place.freshfig`, this does not save/load window placement
    across sessions (`.fig_layout.*` files are inert): that only ever worked on
    Qt backends, which the tutorials no longer depend on.
    """
    if figsize is None:
        figsize = (8, 4)
    fig, ax = plt.subplots(num=num, figsize=figsize, clear=True, **kwargs)
    if sup and isinstance(num, str) and "inline" in mpl.get_backend():
        fig.suptitle(num)
    return fig, ax


def nRowCol(nTotal, figsize=None, axsize=None):
    """Compute `(nrows, ncols)` such that `nTotal ≈ nrows*ncols`.

    Takes into account the `figsize` and `axsize`, either of which
    may be given as a tuple, or a number (the ratio width/height).
    Defaults are taken from `mpl.rcParams`.

    Examples
    --------
    >>> nRowCol(4)
    {'nrows': 2, 'ncols': 2}
    >>> nRowCol(4, (4, 1), (1, 1))
    {'nrows': 1, 'ncols': 4}
    >>> nRowCol(4, (4, 1), (4, 1))
    {'nrows': 2, 'ncols': 2}
    >>> nRowCol(12, (4, 3), (1, 1))
    {'nrows': 3, 'ncols': 4}
    """
    if figsize is None:
        figsize = mpl.rcParams["figure.figsize"]
    if axsize is None:
        rc = {k: mpl.rcParams["figure.subplot." + k] for k in ["bottom", "left", "right", "top"]}
        axsize = (rc["right"] - rc["left"], rc["top"] - rc["bottom"])

    def ratio(size):
        try:
            w, h = size
            return w / h
        except TypeError:
            return size

    ratio = ratio(figsize) / ratio(axsize)
    nrows = round(sqrt(nTotal / ratio)) or 1
    ncols = nTotal // nrows
    if nrows * ncols < nTotal:  # since `//` rounds down
        ncols += 1
    return {"nrows": nrows, "ncols": ncols}


def is_notebook():
    """Detect a Jupyter-type frontend (local Jupyter, Colab, VS Code, nbclient, ...).

    Returns `False` in plain Python and in terminal IPython.
    Relies only on the shell class, which has been stable for a decade:
    `ZMQInteractiveShell` for Jupyter kernels, `google.colab._shell.Shell` on Colab.
    """
    ip = get_ipython()
    if ip is None:
        return False
    cls = type(ip)
    return cls.__name__ == "ZMQInteractiveShell" or "colab" in cls.__module__


def fields(
    model,
    Zs,
    style,
    title="",
    figsize=(9, 4.05),
    cticks=None,
    label_color="k",
    wells=False,
    colorbar=True,
    **kwargs,
):
    """Do `model.plt_field(Z) for Z in Zs`."""

    # Create figure using freshfig
    title = dash_join("Fields", styles[style]["title"], title)
    fig, axs = freshfig(title, figsize=figsize)
    # Store suptitle (exists if mpl is inline) coz gets cleared below
    try:
        suptitle = fig._suptitle.get_text()
    except AttributeError:
        suptitle = ""
    # Create axes using AxesGrid
    fig.clear()
    from mpl_toolkits.axes_grid1 import AxesGrid

    axs = AxesGrid(
        fig,
        111,
        nrows_ncols=nRowCol(min(12, len(Zs))).values(),
        cbar_mode="single",
        cbar_location="right",
        share_all=True,
        axes_pad=0.2,
        cbar_pad=0.1,
    )
    # Turn off redundant axes
    for ax in axs[len(Zs) :]:
        ax.set_visible(False)

    # Convert (potential) list-like Zs into dict
    if not isinstance(Zs, dict):
        Zs = dict(enumerate(Zs))

    # Plot
    hh = []
    for ax, label in zip(axs, Zs):
        label_ax(ax, label, c=label_color)
        hh.append(
            model.plt_field(
                ax,
                Zs[label],
                style,
                wells=wells,
                labels=False,
                colorbar=False,
                title=None,
                finalize=False,
                **kwargs,
            )
        )

    # Axis labels
    fig.supxlabel("x")
    fig.supylabel("y")

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
    fig.tight_layout()
    # warnings.resetwarnings()  # Don't! Causes warnings from (mpl?) libraries
    plt.show()

    return fig, axs, hh


def init():
    """Configure mpl (backend, rcParams).

    ## Backend
    In notebooks, we always use the "inline" backend, which is the default
    of Jupyter kernels, and the only one supported by Colab. The interactive
    alternatives (`%matplotlib widget`, i.e. `ipympl`, and `nbAgg`) were
    dropped: they don't work on Colab, are slower and flicker when embedded
    in `ipywidgets` controls, and testing a single backend is simpler.
    NB: this means `init()` must not `mpl.use()` anything in notebooks.

    In scripts (e.g. `%run HistoryMatch.py` in IPython), `Qt5Agg` is preferred
    since it allows programmatic placement of figure windows. Otherwise
    (e.g. on macOS without Qt) the default GUI backend is used, and figure
    windows at least keep their placement across re-runs thanks to `freshfig`.

    ## Animations
    `model.anim(...)` (minires >= 0.3.1) displays itself as the cell's output
    (its `_repr_html_` is `to_jshtml`) and closes its figure, so it needs no
    `rc("animation", html="jshtml")`, and no `%%capture` against the figure
    being displayed a second time.
    """
    if is_notebook():
        # NB: Non-default figsize/fontsize may cause axis labels/titles
        # that do not fit within the figure, or trespass into the axes
        # (unless fixed by tight_layout, but that is sometimes not possible)
        # Moreover, in general, for presentations, you will use the web-browser
        # zoom functionality, which also zooms in figure and font sizes,
        # reducing the need to change defaults.
        mpl.rcParams.update({"legend.fontsize": "large"})
        mpl.rcParams["font.size"] = 11

    else:
        # Script run
        mpl.rcParams.update({"font.size": 10})
        try:
            mpl.use("Qt5Agg")
        except ImportError:
            pass  # fall back to e.g. MacOS backend

    plt.ion()  # not necessary in Jupyter?


def interact(side="top", wrap=True, **kwargs):
    """Like `ipywidgets.interact(**kwargs)` with some extras:

    - `side` specifies control panel placement relative to figure output.
    - If `wrap`: stack controls vertically (otherwise: horizontally).

    Only the inline mpl backend is used (see `init`).
    """

    def decorator(plotter):
        # NB: `plotter` must use `freshfig` and call `plt.show()` at the end.
        #
        # - Wrapping `plotter` with a function that takes care of it
        #   doesn't work because then `interactive` fails to parse its kwargs.
        # - Doing it in the decorator (at the very end) makes it impossible to
        #   control the layout (controls always wind up above figure output).
        #
        # Earlier python/Jupyter/Colab/mpl versions used to (see cdbb4c15)
        # require even more trickery to make figures actually show up,
        # ref <https://github.com/jupyter-widgets/ipywidgets/issues/3352>
        # and `fig.canvas.{flush_events,draw}`.

        # Auto-parse kwargs, add 'observers'
        # - Like `interact()`, it creates control widgets from simple kwargs.
        # - Like `interactive_output()`, it delays `display()` until manually called.
        # - See source `ipywidgets/widgets/interaction.py` for other nuances.
        linked = wg.interactive(plotter, **kwargs)
        *ww, out = linked.children

        # Styling of individual control widgets
        for w in ww:
            try:
                # Disable continuous_update on Colab
                import google.colab  # type: ignore  # noqa: F401

                w.continuous_update = False
            except ImportError:
                pass

        if wrap:
            cpanel = wg.HBox(ww, layout=dict(flex_flow="row wrap"))
        else:
            cpanel = wg.VBox(ww, layout=dict(align_items="center", justify_content="center"))
        match side:
            case "left":
                dashboard = wg.HBox([cpanel, out])
            case "right":
                dashboard = wg.HBox([out, cpanel])
            case "bottom":
                dashboard = wg.VBox([out, cpanel])
            case _:  # top
                dashboard = wg.VBox([cpanel, out])

        display(dashboard)
        linked.update()

    return decorator


# TODO: unify with layout1 (and make use of interactive() above)
def field_console(model, compute, style, title="", figsize=None, **kwargs):
    """`model.plt_field(compute())` in particular layout along w/ `compute.controls`."""
    title = dash_join("Console", title)
    ctrls = compute.controls.copy()  # gets modified

    def plot(**kw):
        fig, ax = freshfig(title, figsize=figsize)

        # Ignore warnings due to computing and plotting contour/nan
        with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
            warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.contour")
            Z = compute(**kw)

        model.plt_field(ax, Z, style, colorbar=True, finalize=False, **kwargs)

        # Add crosshairs
        if "x" in kw and "y" in kw:
            x, y = model.sub2xy(kw["x"], kw["y"])
            d = dict(c="k", ls="--", lw=1)
            ax.axhline(y, **d)
            ax.axvline(x, **d)

        fig.tight_layout()
        plt.show()

    # Make widget/interactive plot
    linked = wg.interactive(plot, **ctrls)
    *ww, output = linked.children

    # Adjust control styles
    for w in ww:
        if "Slider" in str(type(w)):
            w.continuous_update = False  # => faster
        elif "Dropdown" in str(type(w)):
            w.layout.width = "max-content"

    # Layout
    try:
        layout = layout1(ww, output)
    except (ValueError, IndexError):
        # Fallback
        cpanel = wg.VBox(ww, layout=dict(align_items="center", justify_content="center"))
        layout = wg.HBox([output, cpanel])

    # Display
    display(layout)
    linked.update()


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

    Layouts (better docs):

    - https://coderzcolumn.com/tutorials/python/interactive-widgets-in-jupyter-notebook-using-ipywidgets
    - https://medium.com/kapernikov/ipywidgets-with-matplotlib-93646718eb84

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
        c="k",
        alpha=1.0,
        lw=0.5,
        ls="-",
        marker="",
        ms=4,
    )
    if label == "Truth":
        style.lw = 2
        style.zorder = 2.1
    if label == "Noisy":
        style.label = "Obs"
        style.ls = ""
        style.marker = "*"
    if label == "Prior":
        style.c = "C0"
        style.alpha = 0.3
    if label == "ES":
        # style.label = "Ens. Smooth."
        style.c = "C1"
        style.alpha = 0.3
    if label == "ES0":
        style.c = "C2"
        style.alpha = 0.3
        style.zorder = 1.9
    if label == "IES":
        style.c = "C5"
        style.alpha = 0.3
    if label == "LES":
        style.c = "C4"
        style.alpha = 0.3
    if label == "ILES":
        style.c = "C8"
        style.alpha = 0.3

    # Incrase alpha if N is small
    style.alpha **= 1 + np.log10(N / 100)

    return style


# NOTE: This uses IPython/jupyter widgets (works on Colab), rather than
# an interactive mpl backend (not available on Colab).
def toggle_items(wrapped):
    """Include checkboxes/checkmarks to toggle plotted data series on/off."""

    def new(*args, **kwargs):
        checkmarks = {label: i < 4 for i, label in enumerate(args[0])}
        linked = wg.interactive(wrapped(*args, **kwargs), **checkmarks)

        # Adjust layout
        *ww, output = linked.children
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
            c = ens_style(w.description, N=1)["c"]
            c = mpl.colors.to_hex(c, keep_alpha=False)
            w.layout.border = "solid 7px" + c

            # Center vertically inside boxes
            w.layout.align_items = "center"
            w.layout.padding = ".9em 0 .7em 0"

        cpanel = wg.VBox(ww, layout=dict(align_items="center", justify_content="center"))
        layout = wg.HBox([output, cpanel])
        display(layout)
        linked.update()

    return new


@toggle_items
def productions(dct, title="", figsize=(8, 3.5), nProd=None):
    """Production time series with data toggler. 1 axes/well. Ensemble compatible."""
    title = dash_join("Production profiles", title)

    if nProd is None:
        nProd = struct_tools.get0(dct).shape[1]
        nProd = min(23, nProd)

    kw_subplots = dict(num=title, figsize=figsize, **nRowCol(nProd), sharex=True, sharey=True)

    def plot(**labels):
        _fig, axs = freshfig(**kw_subplots)
        axs = axs.ravel()

        # Turn off redundant axes
        for ax in axs[nProd:]:
            ax.set_visible(False)

        # For each axis/well
        for i in range(nProd):
            ax = axs[i]
            label_ax(ax, f"Well {i}", x=0.99, ha="right")

            for label, series in dct.items():
                if not labels[label]:
                    continue

                # Get style props
                some_ensemble = list(dct.values())[-1]
                props = ens_style(label, N=len(some_ensemble))

                # Plot
                _ll = ax.plot(1 - series.T[i], **props)  # noqa

                # Rm duplicate labels
                # plt.setp(_ll[1:], label="_nolegend_")
        plt.show()

    return plot


def spectrum(ydata, title="", figsize=(6, 3), semilogy=False, **kwargs):
    """Plotter specialized for spectra."""
    title = dash_join("Spectrum", title)
    fig, ax = freshfig(title, figsize=figsize)
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


def label_ax(ax, txt, x=0.01, y=0.99, ha="left", va="top", c="k", fontsize="large", bbox=None):
    if bbox is None:
        bbox = dict(edgecolor="w", facecolor="w", alpha=0.5, boxstyle="round,pad=0")
    return ax.text(
        x, y, txt, c=c, fontsize=fontsize, ha=ha, va=va, transform=ax.transAxes, bbox=bbox
    )


def iterative(title, rms):
    ax1 = freshfig(title)[1]
    ax1.grid()
    ax1.set_xlabel("iteration")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twinx()
    ax2.tick_params(axis="y", labelcolor="C1")

    lines = [
        ax1.plot(rms.prior, label="wrt. Prior", lw=3, c="C0")[0],
        ax2.plot(rms.obsrv, label="wrt. Obsrv", lw=3, c="C1")[0],
        ax1.plot(rms.error, label="wrt. Truth", lw=3, c="C2")[0],
    ]
    ax1.legend(handles=lines, loc="upper right")
    ax1.set_ylabel("RMS")
    plt.tight_layout()
    plt.show()


def figure12(title="", *args, figsize=(9, 3.15), **kwargs):
    """Call `freshfig`. Add axes laid out with 1 panel on right, two on left."""
    title = dash_join("Optim. trajectories", title)
    fig, _ax = freshfig(title, *args, figsize=figsize, **kwargs)
    _ax.remove()
    gs = GridSpec(2, 10)
    ax0 = fig.add_subplot(gs[:, :7])
    ax1 = fig.add_subplot(gs[0, 7:])
    ax2 = fig.add_subplot(gs[1, 7:])

    ax2.sharex(ax1)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_ylabel("Obj")
    ax2.set_ylabel("|Step|")
    ax1.tick_params(labelbottom=False)
    ax2.set(xlabel="itr")

    for ax in (ax1, ax2):
        ax.grid(True, "both", axis="both")
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.yaxis.set_tick_params(labelsize="small")
    return fig, (ax0, ax1, ax2)


def add_path12(ax1, ax2, ax3, path, objs=None, color=None, labels=True):
    """Plot 2d path in `ax1`, `objs` in `ax2`, step size magnitude to `ax3`."""
    # Path line
    ax1.plot(*path.T[:2], c=color or "g")
    # Path scatter and text
    if len(path) > 1:
        ii = set(np.logspace(-1e-9, np.log10(len(path)) - 1e-9, 15, dtype=int))
    else:
        ii = [0]
    cm = plt.get_cmap("viridis_r")(np.linspace(0.0, 1, len(ii)))
    for k, c in zip(ii, cm):
        if color:
            c = color
        xy = path[k][:2]
        if labels:
            ax1.text(*xy, k, c=c)
        ax1.scatter(*xy, s=4**2, color=c, zorder=5)

    if objs is not None:
        # Objective values
        ax2.plot(objs, color=color or "C0", marker=".", ms=3**2)

        # Step magnitudes
        xx = np.arange(len(path) - 1) + 0.5
        dd = np.diff(path, axis=0)
        yy = np.sqrt(np.mean(dd * dd, -1))  # == norm / sqrt(len)
        ax3.plot(xx, yy, color=color or "C1", marker=".", ms=3**2)
