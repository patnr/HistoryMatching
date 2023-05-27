"""Common tools."""

import numpy as np
import scipy.linalg as sla


def norm(xx):
    # return numpy.linalg.norm(xx/sqrt(len(xx)), ord=2)
    return np.sqrt(np.mean(xx * xx))


def RMSMs(series, ref):
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
        print(f"{k:8}: {norm(err):6.4f}   {norm(dev):6.4f}")


def svd0(A):
    """Similar to Matlab's svd(A,0).

    Compute the

     - full    svd if nrows > ncols
     - reduced svd otherwise.

    As in Matlab: svd(A,0),
    except that the input and output are transposed, in keeping with DAPPER convention.
    It contrasts with scipy.linalg's svd(full_matrice=False) and Matlab's svd(A,'econ'),
    both of which always compute the reduced svd.

    .. seealso:: tsvd() for rank (and threshold) truncation.
    """
    M, N = A.shape
    if M > N:
        return sla.svd(A, full_matrices=True)
    return sla.svd(A, full_matrices=False)


def pad0(ss, N):
    """Pad ss with zeros so that len(ss)==N."""
    out = np.zeros(N)
    out[:len(ss)] = ss
    return out


def pows(U, sig):
    """Prepare the computation of the matrix power of a symmetric matrix.

    The input matrix is specified by its eigen-vectors (U) and -values (sig).
    """
    def compute(expo):
        return (U * sig**expo) @ U.T
    return compute


def center(E, axis=0, rescale=False):
    """Center ensemble, `E`.

    Makes use of np features: keepdims and broadcasting.

    If it is known that the true/theoretical mean of (the members of) `E`
    is actually zero, it might be beneficial make it so for `E`, but at the same
    time compensate for the reduction in the (expected) variance this implies.
    This is done if `rescale` is `True`.
    """
    x = np.mean(E, axis=axis, keepdims=True)
    X = E - x

    if rescale:
        N = E.shape[axis]
        X *= np.sqrt(N/(N-1))

    x = x.squeeze()

    return X, x


def mean0(E, axis=0, rescale=True):
    """Like `center`, but only return the anomalies (not the mean).

    Uses `rescale=True` by default, which is beneficial
    when used to center observation perturbations.
    """
    return center(E, axis=axis, rescale=rescale)[0]


def inflate_ens(E, factor):
    """Inflate the ensemble (center, inflate, re-combine)."""
    if factor == 1:
        return E
    X, x = center(E)
    return x + X*factor


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


def ens_run(fun, *inputs, leave=True):
    """Apply `fun` to *ensembles* (2D arrays) of `inputs`.

    This is mainly a wrapper around `multiprocessing`.

    - Emits progressbar.
    - Contains alternative for-loop implementation through equal interface.
    - Takes care of transpose and un-transpose vars.
    - `nCPU` specification shenanigans.

    >>> xx = [1, 2, 3]
    >>> yy = [10, 20, 30]
    >>> ens_run(lambda x: x, xx)
    [1, 2, 3]
    >>> ens_run(lambda xy: xy[0] + xy[1], xx, yy)
    [11, 22, 33]
    >>> ens_run(lambda xy: (xy[0], xy[1]), xx, yy)
    [array([1, 2, 3]), array([10, 20, 30])]
    """

    global nCPU
    is_int = type(nCPU) == int  # `isinstance(True, int)` is True
    if not is_int and nCPU in [True, None, "auto"]:
        nCPU = 999

    tqdm_kws = dict(
        desc=f"{fun.__name__} on ens",
        total=len(inputs[0]),
        leave=leave,
    )

    if len(inputs) > 1:
        # Tranpose such that "member index" is on axis 0.
        xx = zip(*inputs)
    else:
        # Squeeze
        xx = inputs[0]

    if nCPU > 1:
        import multiprocessing
        import threadpoolctl
        from p_tqdm import p_map
        nCPU = min(multiprocessing.cpu_count(), nCPU)
        threadpoolctl.threadpool_limits(1)  # make np use only 1 core
        yy = p_map(fun, list(xx), num_cpus=nCPU, **tqdm_kws)

    else:
        from tqdm.auto import tqdm
        # yy = tqdm(map(fun, xx), **tqdm_kws)
        yy = []
        for x in tqdm(xx, **tqdm_kws):
            yy.append(fun(x))

    try:
        # Un-transpose
        yy = [np.array(y) for y in zip(*yy)]
    except TypeError:
        pass

    return yy
