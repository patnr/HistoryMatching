import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_tools.misc import fig_placement_load, is_notebook_or_qt
import IPython.display as ipy_disp

def init():
    if is_notebook_or_qt:
        mpl.rcParams.update({'font.size': 13})
        mpl.rcParams["figure.figsize"] = [9, 7]
    else:
        try:
            import google.colab  # noqa
        except ImportError:
            mpl.use("Qt5Agg")
            plt.ion()
            fig_placement_load()


def display(animation):
    if is_notebook_or_qt:
        ipy_disp.display(ipy_disp.HTML(animation.to_jshtml()))
    else:
        plt.show(block=False)
