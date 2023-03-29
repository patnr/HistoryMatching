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
import matplotlib as mpl
import mpl_tools
from matplotlib import pyplot as plt


def init():
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
