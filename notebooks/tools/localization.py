"""Localization tools, including distance and tapering comps.

Copied from DAPPER and then simplified.
"""

import numpy as np


def pairwise_distances(A, B=None, domain=None):
    """Euclidian distance (not squared) between pts. in `A` and `B`.

    Parameters
    ----------
    A: array of shape `(nPoints, nDims)`.
        A collection of points.

    B:
        Same as `A`, but `nPoints` can differ.

    domain: tuple
        Assume the domain is a **periodic** hyper-rectangle whose
        edges along dimension `i` span from 0 to `domain[i]`.
        NB: Behaviour not defined if `any(A.max(0) > domain)`, and likewise for `B`.

    Returns
    -------
    Array of of shape `(nPointsA, nPointsB)`.

    Examples
    --------
    >>> A = [[0, 0], [0, 1], [1, 0], [1, 1]]
    >>> with np.printoptions(precision=2):
    ...     print(pairwise_distances(A))
    [[0.   1.   1.   1.41]
     [1.   0.   1.41 1.  ]
     [1.   1.41 0.   1.  ]
     [1.41 1.   1.   0.  ]]

    The function matches `pdist(..., metric='euclidean')`, but is faster:
    >>> from scipy.spatial.distance import pdist, squareform
    >>> (pairwise_distances(A) == squareform(pdist(A))).all()
    True

    As opposed to `pdist`, it also allows comparing `A` to a different set of points,
    `B`, without the augmentation/block tricks needed for pdist.

    >>> A = np.arange(4)[:, None]
    >>> pairwise_distances(A, [[2]]).T
    array([[2., 1., 0., 1.]])

    Illustration of periodicity:
    >>> pairwise_distances(A, domain=(4, ))
    array([[0., 1., 2., 1.],
           [1., 0., 1., 2.],
           [2., 1., 0., 1.],
           [1., 2., 1., 0.]])

    NB: If an input array is 1-dim, it is seen as a single point.
    >>> pairwise_distances(np.arange(4))
    array([[0.]])
    """
    if B is None:
        B = A

    # Prep
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    mA, nA = A.shape
    mB, nB = B.shape
    assert nA == nB, "The last axis of A and B must have equal length."

    # Diff
    d = A[:, None] - B  # shape: (mA, mB, nDims)

    # Make periodic
    if domain:
        domain = np.reshape(domain, (1, 1, -1))  # for broadcasting
        d = abs(d)
        d = np.minimum(d, domain - d)

    distances = np.sqrt((d * d).sum(axis=-1))

    return distances.reshape(mA, mB)


def bump_function(distances, sharpness=1):
    mask = np.abs(distances) < 1  # only compute for |distances|<1
    x = distances[mask]
    v = np.exp(1 - 1 / (1 - x * x)) ** sharpness
    coeffs = np.zeros_like(distances)
    coeffs[mask] = v
    return coeffs


def rectangular_partitioning(shape, steps, do_ind=True):
    """N-D rectangular batch generation.

    Parameters
    ----------
    shape: (len(grid[dim]) for dim in range(ndim))
    steps: (step_len[dim]  for dim in range(ndim))

    Returns
    -------
    A list of batches,
    where each element (batch) is a list of indices.

    Example
    -------
    >>> shape   = [4, 13]
    ... batches = rectangular_partitioning(shape, [2, 4], do_ind=False)
    ... nB      = len(batches)
    ... values  = np.random.choice(np.arange(nB), nB, 0)
    ... Z       = np.zeros(shape)
    ... for ib, b in enumerate(batches):
    ...     Z[tuple(b)] = values[ib]
    ... plt.imshow(Z)  # doctest: +SKIP
    """
    import itertools

    assert len(shape) == len(steps)
    # ndim = len(steps)

    # An ndim list of (average) local grid lengths:
    nLocs = [round(n / d) for n, d in zip(shape, steps)]
    # An ndim list of (marginal) grid partitions
    # [array_split() handles non-divisibility]:
    edge_partitions = [np.array_split(np.arange(n), nLoc) for n, nLoc in zip(shape, nLocs)]

    batches = []
    for batch_edges in itertools.product(*edge_partitions):
        # The 'indexing' argument below is actually inconsequential:
        # it merely changes batch's internal ordering.
        batch_rect = np.meshgrid(*batch_edges, indexing="ij")
        coords = [ii.flatten() for ii in batch_rect]
        batches += [coords]

    if do_ind:

        def sub2ind(sub):
            return np.ravel_multi_index(sub, shape)

        batches = [sub2ind(b) for b in batches]

    return batches
