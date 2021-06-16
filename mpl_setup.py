"""Configure mpl."""
import matplotlib as mpl
import mpl_tools
from matplotlib import pyplot as plt


def init():
    try:
        import google.colab  # noqa
        mpl.rc('animation', html="jshtml")
    except ImportError:
        if mpl_tools.is_notebook_or_qt:
            mpl.use("nbAgg")
            mpl.rc('animation', html="jshtml")
            mpl.rcParams.update({'font.size': 15})
            mpl.rcParams["figure.figsize"] = [9, 7]
        else:
            try:
                mpl.use("Qt5Agg")
            except ImportError:
                pass
            plt.ion()
