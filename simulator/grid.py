"""Tools for working with model grid coordinates.

Most functions here are barely in use.
So they're here mostly just for reference.
After all, it is surprisingly hard to remember
which direction/index is for x and which is for y.

Index ordering/labels: `x` is 1st coord., `y` is 2nd.
This choice has been hardcoded in the model code, in what takes place
**between** `np.ravel` and `np.reshape` (using standard "C" ordering).
This choice also means that the letters `x` and `y` tend to occur in alphabetic order.
However, the plots.py module depicts x from left to right, and y from bottom to top.

Example:
>>> grid = Grid2D(Lx=4, Ly=10, Nx=2, Ny=5)
>>> X, Y = grid.mesh();
>>> X
array([[1., 1., 1., 1., 1.],
       [3., 3., 3., 3., 3.]])
>>> Y
array([[1., 3., 5., 7., 9.],
       [1., 3., 5., 7., 9.]])

>>> XY = np.stack((X, Y), axis=-1)
>>> grid.xy2sub(*XY[1,3])
(1, 3)

>>> grid.sub2xy(1,3) == XY[1,3]
array([ True,  True])
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Grid2D:
    """Defines a 2D rectangular grid."""

    Lx: float = 1.0
    Ly: float = 1.0
    Nx: int = 32
    Ny: int = 32

    def __post_init__(self):
        self.shape = self.Nx, self.Ny
        self.grid  = self.shape + (self.Lx, self.Ly)
        self.M     = np.prod(self.shape)

        # Resolution
        self.hx, self.hy = self.Lx/self.Nx, self.Ly/self.Ny
        self.h2 = self.hx*self.hy

    def mesh(self, centered=True):
        """Generate 2D coordinate grids."""
        xx = np.linspace(0, self.Lx, self.Nx, endpoint=False)
        yy = np.linspace(0, self.Ly, self.Ny, endpoint=False)

        if centered:
            xx += self.hx/2
            yy += self.hy/2

        return np.meshgrid(xx, yy, indexing="ij")

    def sub2ind(self, ix, iy):
        """Convert index `(ix, iy)` to index in flattened array."""
        idx = np.ravel_multi_index((ix, iy), self.shape)
        return idx

    def ind2sub(self, ind):
        """Inv. of `self.sub2ind`."""
        ix, iy = np.unravel_index(ind, self.shape)
        return np.asarray([ix, iy])

    def xy2sub(self, x, y):
        """Convert physical coordinate tuple to tuple `(ix, iy)`.

        Here, ix ∈ {0, ..., Nx-1}.

        Note: rounds to nearest mesh center (i.e. is not injective).
        The alternative would be to return some kind of interpolation weights.
        """
        # Clip to allow for case x==Lx (arguably, Lx [but not 0!] is out-of-domain).
        # Warning: don't simply subtract 1e-8; causes issue if x==0.
        x = np.asarray(x).clip(max=self.Lx-1e-8)
        y = np.asarray(y).clip(max=self.Ly-1e-8)
        ix = np.floor(x / self.Lx * self.Nx).astype(int)
        iy = np.floor(y / self.Ly * self.Ny).astype(int)
        return np.asarray([ix, iy])

    def xy2ind(self, x, y):
        """Convert physical coord to flat indx. NB: see caution in `xy2sub`."""
        i, j = self.xy2sub(x, y)
        return self.sub2ind(i, j)

    def sub2xy(self, ix, iy):
        """Inverse of `self.xy2sub` (outputs mesh CENTERS)."""
        x = (np.asarray(ix) + .5) * self.hx
        y = (np.asarray(iy) + .5) * self.hy
        return np.asarray([x, y])

    def ind2xy(self, ind):
        """Inv. of `self.xy2ind` (outputs mesh CENTERS)."""
        i, j = self.ind2sub(ind)
        return self.sub2xy(i, j)

    def sub2xy_stretched(self, ix, iy):
        """Like `self.xy2sub`, but inflating. Puts node `i=0` at `0`, and `i=N-1` at `L`.

        This is wrong. Only use with `plotting.field()`, which also stretches the field
        (because it does not use `origin="lower"`).
        """
        x = np.asarray(ix) * self.Lx/(self.Nx-1)
        y = np.asarray(iy) * self.Ly/(self.Ny-1)
        return np.asarray([x, y])

    def ind2xy_stretched(self, ind):
        """Like `self.xy2ind`, but using `sub2xy_stretched`."""
        i, j = self.ind2sub(ind)
        return self.sub2xy_stretched(i, j)
