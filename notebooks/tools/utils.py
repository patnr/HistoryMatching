"""Common tools."""

import numpy as np
import scipy.linalg as sla

from adjustText import adjust_text  # noqa


def progbar(*args, **kwargs):
    """Essentially `tqdm()`, but with some defaults."""
    # Remove '<{remaining}' because it is somwhat unreliable,
    # and hard to distinguish at a glance from 'elapsed')
    frmt = "{l_bar}|{bar}| {n_fmt}/{total_fmt}, ⏱️ {elapsed}, {rate_fmt}{postfix}"
    kwargs.setdefault('bar_format', frmt)

    # Choose between Jupyter, std
    from tqdm.auto import tqdm

    # Choose between std, rich (does not support 'bar_format'!)
    # from tqdm.notebook import tqdm_notebook
    # from tqdm.std import TqdmExperimentalWarning
    # import warnings
    # if not isinstance(dummy:=tqdm(disable=True), tqdm_notebook):
    #     try:
    #         from tqdm.rich import tqdm
    #     except ImportError:
    #         pass
    # NB: To ignore initial warning:
    # >>> with warnings.catch_warnings():
    # ...     warnings.simplefilter("ignore", category=TqdmExperimentalWarning)

    pbar = tqdm(*args, **kwargs)
    return pbar


def rinv(A, reg, tikh=True, nMax=None):
    """Reproduces `sla.pinv(..., rtol=reg)` for `tikh=False`."""
    # Decompose
    U, s, VT = sla.svd(A, full_matrices=False)

    # "Relativize" the regularisation param
    reg = reg * s[0]

    # Compute inverse (regularized or truncated)
    if tikh:
        s1 = s / (s**2 + reg**2)
    else:
        s0 = s >= reg
        s1 = np.zeros_like(s)
        s1[s0] = 1/s[s0]

    if nMax:
        s1[nMax:] = 0

    # Re-compose
    return (VT.T * s1) @ U.T


def pCircle(degree, Lx, Ly, p=4, norm_val=.87):
    """Compute `(x, y)` at angle `degree` with `p-norm = norm_val`."""
    # Also center in, and scale by, model domain, i.e. `Lx, Ly`
    radians = 2 * np.pi * degree / 360
    c = np.cos(radians)
    s = np.sin(radians)
    norm = (np.abs(c)**p + np.abs(s)**p)**(1/p)
    x = norm_val/norm * c
    y = norm_val/norm * s
    x = Lx/2 * (1 + x)
    y = Ly/2 * (1 + y)
    x = np.round(x, 2)
    y = np.round(y, 2)
    return x, y


def _mnorm(x, axis=0):
    """L2 norm. Uses `mean` (unlike usual `sum`) for dimension agnosticity."""
    # return numpy.linalg.norm(xx/sqrt(len(xx)), ord=2)
    return np.sqrt(np.mean(x*x, axis))


def print_RMSMs(series, ref):
    """Print RMS err. and dev., from the Mean (along axis 0), for each item in `series`.

    The `ref` must point to a data series that is *not* an ensemble.
    All series (including `ref`) can have both singleton and `squeeze`d axis 0.
    """
    x = series[ref]

    # Ensure reference's axis 0 is singleton.
    if x.shape[0] != 1:
        x = x[None, :]

    # Print table heading
    header = "Series    rms err  rms dev"
    print(header, "-"*len(header), sep="\n")

    for k, y in series.items():

        # Ensure non-ensemble series also has singleton axis 0
        if y.ndim < x.ndim:
            y = y[None, :]
            assert y.shape == x.shape

        err = x - y.mean(0)
        dev = y - y.mean(0)
        print(f"{k:8}: {_mnorm(err, None):6.4f}   {_mnorm(dev, None):6.4f}")


def center(E, axis=0, rescale=False):
    """Center ensemble, `E`. Also return the mean.

    Makes use of `keepdims` and broadcasting.

    If the true/theoretical mean of (the members of) `E` is known and zero,
    it might be beneficial center `E` but also compensate for the reduction
    in the (expected) variance this implies. This is done if `rescale`.
    """
    x = np.mean(E, axis=axis, keepdims=True)
    X = E - x

    if rescale:
        N = E.shape[axis]
        X *= np.sqrt(N/(N-1))

    x = x.squeeze()

    return X, x


def cov(a, b):
    """Compute covariance between a sample of two multivariate variables.

    Unlike `np.cov`, `a` and `b` need not have the same shape,
    but must must of course have equal ensemble size, i.e. `shape[0]`.
    """
    A, _ = center(a)
    B, _ = center(b)
    return A.T @ B / (len(B) - 1)


def corr(a, b):
    """Compute correlation using `cov`."""
    C = cov(a, b)

    sa = np.std(a.T, axis=-1, ddof=1)
    sb = np.std(b  , axis=+0, ddof=1, keepdims=True)
    # with np.errstate(divide="ignore", invalid="ignore"):
    Corr = C / sa / sb

    # Convert inf to 999. Either way it means that the correlation is ill-defined,
    # but contourf colors inf as nan's (given by set_bad(color), not set_over())
    Corr = Corr.clip(-999, 999)

    return Corr


nCPU = 1
"Number of CPUs to use in parallelization"


def apply(fun, *args, pbar=True, **kwargs):
    """Apply `fun` along 0th axis of each `args` and `kwargs`.

    - Multiprocessing
        - Progressbar (also re-useable), not using `p_tqdm`.
        - Alternative for-loop implementation for easier debugging.
        - Zipping and unpacking input `args`.
    - Robust interpretation of `nCPU`.
    """
    # Set nCore
    is_int = type(nCPU) == int  # NB: `isinstance(True, int)` is True
    if not is_int and nCPU in [True, None, "auto"]:
        nCore = 999
    else:
        nCore = nCPU

    # Convert kwargs to positional
    nPositional = len(args)
    args = list(args) + list(kwargs.values())
    # Zip, i.e. transpose, for `map` compatibility
    inputs = list(zip(*args, strict=True))

    def _fun(x):
        """Unpack zipped args `x` and call `fun`."""
        positional, named = x[:nPositional], x[nPositional:]
        kws = dict(zip(kwargs, named))
        return fun(*positional, **kws)

    # Progress-bar initialisation
    if "tqdm" in str(type(pbar)).lower():
        # Reset existing
        pbar.do_close = False
        pbar.reset(total=len(args[0]))
    elif pbar:
        # Create new
        kws = dict(total=len(args[0]), desc=f"map({fun.__name__}, ...)", leave=True)
        if isinstance(pbar, str):
            kws["desc"] = pbar
        elif isinstance(pbar, dict):
            kws.update(pbar)
        pbar = progbar(**kws)
    else:
        # NOP
        pbar = progbar(disable=True)

    # Main
    if nCore > 1:
        import multiprocessing
        nCore = min(multiprocessing.cpu_count(), nCore)

        # make np use only 1 core
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)

        # Similar to built-in multiprocessing, but with improved pickle (dill)
        from pathos.multiprocessing import ProcessPool
        pool = ProcessPool(nCore)

        # Map must be non-blocking (allow tqdm to update) and ordered ⇒ imap
        # PS: Blocking is provided by the exterior list/for-loop.
        work = pool.imap(_fun, inputs)
        # output = list(tqdm(work, total))
        output = []
        for y in work:
            output.append(y)
            pbar.update()
        pool.clear()

    else:
        # Without multiprocessing (⇒ easier debugging)
        output = []
        for x in inputs:
            output.append(_fun(x))
            pbar.update()

    # Finalize progress-bar
    pbar.refresh()
    if getattr(pbar, "do_close", True):
        pbar.close()

    return output
