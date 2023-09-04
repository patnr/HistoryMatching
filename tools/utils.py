"""Common tools."""

import numpy as np
import scipy.linalg as sla


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


def xy_p_normed(degree, Lx, Ly, p=4, norm_val=.87):
    """Compute `(x, y)` of `degree`, scale so `p-norm = norm_val`."""
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


def mnorm(x, axis=0):
    """L2 norm. Uses `mean` (unlike usual `sum`) for dimension agnosticity."""
    # return numpy.linalg.norm(xx/sqrt(len(xx)), ord=2)
    return np.sqrt(np.mean(x*x, axis))


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
        print(f"{k:8}: {mnorm(err, None):6.4f}   {mnorm(dev, None):6.4f}")


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

    Makes use of `keepdims` and broadcasting.

    If it is known that the true/theoretical mean of (the members of) `E`
    is actually zero, it might be beneficial make it so for `E`, but at the same
    time compensate for the reduction in the (expected) variance this implies.
    This is done if `rescale`.
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


def apply(fun, *args, unzip=True, pbar=True, leave=True, desc=None, **kwargs):
    """Apply `fun` along 0th axis of (zipped) `args` and `kwargs`.

    Provides

    - Multiprocessing.
    - Alternative for-loop implementation for easier debugging.
    - Zipping and unpacking `inputs`. NB: all of `inputs` must be iterable.
    - If `unzip`: unpack output. NB: `fun` must output a *doubly* iterable.
    - Robust interpretation of `nCPU`.
    - Emits progressbar.

    Note: Only (but always) axis 0 is treated special (and requires length conformity).
    This means that the input can be >2D, which is convenient,
    but makes it impossible to say whether a 1d vector (i.e. neither row nor column)
    is a single vector realisation or an ensemble.
    ⇒ Cannot know whether to (i) show p-bar or (ii) squeeze output.
    I.e. these decisions must be left to caller, which has more particulars.
    """
    # Set nCPU
    global nCPU
    is_int = type(nCPU) == int  # `isinstance(True, int)` is True
    if not is_int and nCPU in [True, None, "auto"]:
        nCPU = 999

    # Convert kwargs to positional
    nPositional = len(args)
    args = list(args) + list(kwargs.values())

    def ensure_equal_lengths(xx):
        """Prevent losing data via `zip`."""
        L = len(xx[0])
        assert all(len(x) == L for x in xx)
        return L

    # Pack ("transpose") and atleast_2d (using lists, not np, for objs of any len).
    ensure_equal_lengths(args)
    xx = list(zip(*args))

    # Unpacker for arg
    def function_with_unpacking(x):
        positional, named_vals = x[:nPositional], x[nPositional:]
        kws = dict(zip(kwargs, named_vals))
        return fun(*positional, **kws)

    # Setup or disable (be it with or w/o multiprocessing) tqdm.
    if pbar:
        tqdm_kws = dict(
            desc=desc or f"{fun.__name__}'s",
            total=len(xx),
            leave=leave,
        )
    else:
        tqdm_kws = dict(
            # passthrough
            tqdm=(lambda x, **_: x),
        )

    # Main work
    if nCPU > 1:
        import multiprocessing

        import threadpoolctl
        from p_tqdm import p_map
        nCPU = min(multiprocessing.cpu_count(), nCPU)
        threadpoolctl.threadpool_limits(1)  # make np use only 1 core
        yy = p_map(function_with_unpacking, xx, num_cpus=nCPU, **tqdm_kws)

    else:
        from tqdm.auto import tqdm
        tqdm = tqdm_kws.pop('tqdm', tqdm)
        # yy = tqdm(map(fun, xx), **tqdm_kws)
        yy = []
        for x in tqdm(xx, **tqdm_kws):
            yy.append(function_with_unpacking(x))

    # Unpack
    if unzip:
        ensure_equal_lengths(yy)
        yy = list(zip(*yy))
        yy = [np.asarray(y) for y in yy]
    else:
        yy = np.asarray(yy)

    return yy
