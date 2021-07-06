"""Tools for working with model grid coordinates.

Most functions here are barely in use.
So they're here mostly just for reference.
After all, it is surprisingly hard to remember
which direction/index ix for x and which is for y.

Index ordering/labels: `x` is 1st coord., `y` is 2nd.
This is hardcoded in the model code, in what takes place
**between** `np.ravel` and `np.reshape` (using standard "C" ordering).
It also means that the letters `x` and `y` tend to occur in alphabetic order.

However, the plots.py module depicts
x from left to right, and y from bottom to top.

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
        return ix, iy

    def xy2sub(self, x, y):
        """Convert physical coordinate tuple to tuple `(ix, iy)`.

        Note: rounds to nearest integers (i.e. is not injective).
        The alternative would be to return some kind of interpolation weights.
        """
        # ix = int(round(x/self.Lx*(self.Nx-1)))
        # iy = int(round(y/self.Ly*(self.Ny-1)))
        ix = (np.array(x) / self.Lx*(self.Nx-1)).round().astype(int)
        iy = (np.array(y) / self.Ly*(self.Ny-1)).round().astype(int)
        return ix, iy

    def sub2xy(self, ix, iy):
        """Approx. inverse of `self.xy2sub`. NB: see caution in `xy2sub`."""
        x = self.Lx * (ix + .5)/self.Nx
        y = self.Ly * (iy + .5)/self.Ny
        return x, y

    def sub2xy_stretched(self, ix, iy):
        """Like `self.xy2sub`, but inflating put node `i=0` at `0`, and `i=N-1` at `L`.

        This is wrong. Only use with `plotting.field()`, which also stretches the field.
        """
        x = self.Lx * ix/(self.Nx-1)
        y = self.Ly * iy/(self.Ny-1)
        return x, y

    def xy2ind(self, x, y):
        """Convert physical coord to flat indx. NB: see caution in `xy2sub`."""
        return self.sub2ind(*self.xy2sub(x, y))

    def ind2xy(self, ind):
        """Inv. of `self.xy2ind`. NB: see caution in `xy2sub`."""
        i, j = self.ind2sub(ind)
        # TODO: should be able to use sub2xy here,
        # but that seems different.
        x    = i/(self.Nx-1)*self.Lx
        y    = j/(self.Ny-1)*self.Ly
        return x, y
