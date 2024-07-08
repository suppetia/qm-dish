import numpy as np

from typing import Union


class DistanceGrid:
    """
    Class to hold the information about the grid and the grid points itself.
    The grid is constructed lazily, i.e. the grid points are calculated the first time they are required.
    The grid has the form :math:`r(t) = r0 \\cdot \\left(\\exp{t} - 1\\right)` where :math:`t(i) = i\\cdot h` is a linear grid with ``N`` points.
    The number of values ``N`` can be constructed from the maximal r value ``r_max``.
    """

    def __init__(self,
                 h: float,
                 r0: float,
                 N: int = None,
                 r_max: float = None):

        if (N is None and r_max is None) or (N is not None and r_max is not None):
            raise ValueError("Either 'N' or 'r_max' must be specified but not both.")
        if N is not None and not isinstance(N, int):
            ValueError(f"Number of grid points 'N' must be an integer but is {type(N)}")
        self._h = h
        self._r0 = r0
        if r_max is not None:
            self._N = int(np.ceil(np.log(r_max/r0 + 1)/h))
        else:
            self._N = N

        # some grids are just constructed for the parameters and don't need the actual arrays
        # so just initialize them when required
        self._is_initialized = False
        self._t = None
        self._r = None
        self._rp = None  # r_prime -> derivative

    def _init_grid(self):
        self._t = np.arange(self._N) * self._h
        self._r = self.r0 * (np.exp(self._t) - 1)
        self._rp = self.r0 * np.exp(self._t)  # r_prime -> derivative
        self._is_initialized = True


    @property
    def t(self):
        if not self._is_initialized:
            self._init_grid()
        return self._t
    @property
    def r(self):
        if not self._is_initialized:
            self._init_grid()
        return self._r
    @property
    def h(self):
        return self._h

    @property
    def rp(self):
        if not self._is_initialized:
            self._init_grid()
        return self._rp

    @property
    def r0(self):
        return self._r0
    @property
    def N(self):
        return self._N

    @property
    def r_max(self):
        return self.r0 * (np.exp(self.h*(self.N-1)) - 1)

    def __eq__(self, other):
        if not isinstance(other, DistanceGrid):
            return NotImplemented
        return np.allclose(np.array([self.r0, self.h, self.N]), np.array([other.r0, other.h, other.N]))

    def __repr__(self):
        return f"DistanceGrid(r0={self.r0}, h={self.h}, N={self.N}, r_max={self.r_max})"


class RombergIntegrationGrid(DistanceGrid):
    """
    Class to hold the information about the grid and the grid points itself.
    The grid is constructed lazily, i.e. the grid points are calculated the first time they are required.
    The grid has the form :math:`r(t) = r0 \\cdot \\left(\\exp{t} - 1\\right)` where :math:`t(i) = i \\cdot h` is a linear grid with ``N`` points.

    The number of grid points ``N`` needs to be of the form :math:`N = 2^k + 1` where k is a positive integer.
    By using this number of grid points this grid is suited to perform Romberg integration on it which yields more accurate results than the naive trapezoidal rule.
    """

    def __init__(self,
                 h: float,
                 r0: float,
                 N: int = None,
                 k: int = None):

        if (N is None and k is None) or (N is not None and k is not None):
            raise ValueError("Either 'N' or 'k' must be specified but not both.")

        if N is not None:
            k = np.log2(N - 1)
            if not np.isclose(k - np.floor(k), 0):
                raise ValueError(f"N must be 2^k+1 for a positive integer k but is {N}")

        if k is not None:
            N = 2**k+1

        super().__init__(h=h, r0=r0, N=N)

    @property
    def k(self):
        return int(np.log2(self.N - 1))

    @classmethod
    def construct_similar_grid_from_distance_grid(cls, grid: Union[np.ndarray, DistanceGrid]):
        """
        Construct a RombergIntegrationGrid from a DistanceGrid.
        The number of points is increased to the next integer k so that :math:`N = 2^k + 1` is fulfilled by decreasing the value of ``h``.

        :param grid: DistanceGrid
        :return: RombergIntegrationGrid similar to grid
        """
        if isinstance(grid, DistanceGrid):
            # test for Romberg integration
            N = grid.N
            r0 = grid.r0
            last_r = grid.r[-1]
        else:
            raise ValueError(f"pass a DistanceGrid, for array not yet implemented")
            # N = len(grid)
            # r0 =
            # last_r = grid[-1]
        k = int(np.ceil(np.log2(N - 1)))
        N = 2 ** k + 1
        h = np.log(last_r / r0 + 1) / (N-1)

        return cls(h=h, r0=grid.r0, k=k)
