"""Configure mpl."""
import IPython.display as ipy_disp
import matplotlib as mpl
import mpl_tools
from matplotlib import pyplot as plt


def init():
    try:
        import google.colab  # noqa
    except ImportError:
        if mpl_tools.is_notebook_or_qt:
            mpl.use("nbAgg")
            mpl.rcParams.update({'font.size': 15})
            mpl.rcParams["figure.figsize"] = [9, 7]
        else:
            mpl.use("Qt5Agg")
            plt.ion()


def display_anim(animation):
    if mpl_tools.is_notebook_or_qt:
        ipy_disp.display(ipy_disp.HTML(animation.to_jshtml()))
    else:
        plt.show(block=False)
