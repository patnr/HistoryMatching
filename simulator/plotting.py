"""Plot functions for reservoir model."""

import matplotlib as mpl
import numpy as np
from mpl_tools import place, place_ax
from mpl_tools.misc import axprops

# The "self"
# Why not explicit OOP?
# - unnecessary indent
# - not every realisation of the model needs a copy of this
model = None

# Axes limits
coord_type = "relative"   # ==> (0, 1)  x (0, 1)
# coord_type = "absolute" # ==> (0, Lx) x (0, Ly)
# coord_type = "index"    # ==> (0, Ny) x (0, Ny)

# Colormap for saturation
lin_cm = mpl.colors.LinearSegmentedColormap.from_list
cm_ow = lin_cm("", [(0, "#1d9e97"), (.3, "#b2e0dc"), (1, "#f48974")])
# cOil, cWater = "red", "blue"        # Plain
# cOil, cWater = "#d8345f", "#01a9b4" # Pastel/neon
# cOil, cWater = "#e58a8a", "#086972" # Pastel
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
        cticks = None,
        # Note that providing vmin/vmax (and not a levels list) to mpl
        # yields prettier colobar ticks, but destorys the consistency
        # of the colorbars from one figure to another.
    ),
    oil = dict(
        title  = "Oil saturation",
        transf = lambda x: 1 - x,
        cmap   = cm_ow,
        levels = np.linspace(0 - 1e-7, 1 + 1e-7, 20),
        cticks = np.linspace(0, 1, 6),
    ),
)


def field(ax, Z, style="default", wells=False, argmax=False, colorbar=False, **kwargs):
    """Contour-plot of the (flat) field `Z`. Styles can be overriden by `kwargs`."""
    # Get style parms, with "default" fallback.
    style = {**styles["default"], **styles[style]}
    # Pop styles from kwargs
    for key in style:
        if key in kwargs:
            style[key] = kwargs.pop(key)
    # Pop axis styles from kwargs
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
    Z = style["transf"](Z)
    Z = Z.reshape(model.shape).T

    # Did we bother to specify set_over/set_under/set_bad ?
    has_out_of_range = getattr(style["cmap"], "_rgba_over", None) is not None

    # ax.imshow(Z[::-1])
    collections = ax.contourf(
        Z, style["levels"], cmap=style["cmap"], **kwargs,
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
        ax.set_facecolor(getattr(style["cmap"], "_rgba_bad", "w"))

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
        ax.figure.colorbar(collections, **cax, ticks=style["cticks"])

    return collections


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


def production(ax, production, obs=None, legend_outside=True):
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
    title = "Animation -- " + key
    fig, (ax1, ax2) = place.freshfig(title, ncols=2, figsize=figsize, rel=True)
    fig.suptitle(title)  # coz animation never (any backend) displays title
    # Saturations
    ax2.cc = field(ax2, wsats[-1], "oil", wells="color", colorbar=True, **kwargs)
    # Production
    hh = production(ax1, prod, legend_outside=False)

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
