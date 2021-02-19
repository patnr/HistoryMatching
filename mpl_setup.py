"""Configure mpl."""
import matplotlib as mpl
from matplotlib import pyplot as plt
import mpl_tools
import IPython.display as ipy_disp


def init():
    if mpl_tools.is_notebook_or_qt:
        mpl.rcParams.update({'font.size': 15})
        mpl.rcParams["figure.figsize"] = [9, 7]
    else:
        try:
            import google.colab  # noqa
        except ImportError:
            mpl.use("Qt5Agg")
            plt.ion()


def display_anim(animation):
    if mpl_tools.is_notebook_or_qt:
        ipy_disp.display(ipy_disp.HTML(animation.to_jshtml()))
    else:
        plt.show(block=False)
