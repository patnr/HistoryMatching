"""Configure mpl.

Why not leave defaults:

- When running as script, Qt5Agg is nice coz it allows
  programmatic (automatic) placement of figures on screen.
- nbAgg (%matplotib notebook) is interactive, so cooler.
  Also for animation to work, we need the "jshtml" stuff.
- Colab does not support nbAgg, but it does appear to support plotly
  https://stackoverflow.com/a/64297121

Caution: On my Mac, when I step through the cells quickly (or "run all")
then the figures sometimes do not display, or display as "inline" instead of "nbAgg".

Caution: Getting the dashboard animation to work properly was complicated.
In particular, the figure tended to display twice: once statically,
and once as an animation.
When working locally (on Mac), using `%matplotlib inline` was sufficient.
But that was problematic for the following figures, which did not display
with that backend (despite using `plt.show()` and `plt.pause(.1)` etc),
although switching back to the interactive backend with `%matplotib notebook`
seemed to work ok. However, on Colab, which only supports `%matplotlib inline`,
the animation figure still displayed double. The alternative solution
seems to be to split the creation and displaying cells, and using %%capture.
This worked both locally and on Colab. Refs:
- https://stackoverflow.com/q/47138023
- https://stackoverflow.com/a/36685236
"""
import matplotlib as mpl
import mpl_tools
from matplotlib import pyplot as plt


def init():
    if mpl_tools.is_notebook_or_qt:
        mpl.rc('animation', html="jshtml")
        try:
            import google.colab  # noqa
        except ImportError:
            mpl.use("nbAgg")
            mpl.rcParams.update({'font.size': 15})
            mpl.rcParams["figure.figsize"] = [9, 7]
    else:
        try:
            mpl.use("Qt5Agg")
        except ImportError:
            pass  # fall back to e.g. MacOS backend
        plt.ion()
