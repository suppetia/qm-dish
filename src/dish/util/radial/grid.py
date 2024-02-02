import numpy as np
from scipy.optimize import curve_fit

from typing import Union


class DistanceGrid:

    def __init__(self,
                 h: float,
                 r0: float,
                 N: int = None,
                 r_max: float = None):

        if (N is None and r_max is None) or (N is not None and r_max is not None):
            raise ValueError("Either 'N' or 'r_max' must be specified but not both.")

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
    def rmax(self):
        return self.r0 * (np.exp(self.N-1) - 1)

    def __eq__(self, other):
        if not isinstance(other, DistanceGrid):
            return False
        return np.allclose(np.array([self.r0, self.h, self.N]), np.array([other.r0, other.h, other.N]))


class RombergIntegrationGrid(DistanceGrid):

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


def construct_grid_from_points(r: np.array) -> DistanceGrid:
    N = len(r)

    def grid_func(x, r0, h):
        return r0*(np.exp(x*h)-1)

    # construct the grid parameters from fitting a few points
    num_fit_points = min(30, len(r))
    fit_pts = np.arange(num_fit_points, dtype=np.int64)*N//num_fit_points
    popt, pcov = curve_fit(grid_func, fit_pts, r[fit_pts], p0=[1e-5, 1e-8])
    threshold = 1e-10
    if (pcov > threshold).any():
        for r0 in [1e-5, 1e-7, 1e-9, 1e-11]:
            for h in [1e-9, 1e-7, 1e-5, 1e-3, 1e-1]:
                popt, pcov = curve_fit(grid_func, fit_pts, r[fit_pts], p0=[r0, h])
                if not (pcov > threshold).any():
                    break
    if (pcov > threshold).any():
        raise ValueError("failed to construct grid from given points")
    h = popt[1]
    r0 = popt[0]

    k = np.log2(N - 1)
    if np.isclose(k - np.floor(k), 0):
        return RombergIntegrationGrid(h=h, r0=r0, k=k)
    else:
        return DistanceGrid(h, r0, N)


if __name__ == "__main__":

    r0 = 1e-8
    h = 2.9198e-7
    r_max = 3

    d_grid = DistanceGrid(h, r0, r_max=r_max)
    print(d_grid.N)
    # print(d_grid.r[-1])

    ri_grid = RombergIntegrationGrid.construct_similar_grid_from_distance_grid(d_grid)
    print(ri_grid.N)
    print(ri_grid.k)
    # print(ri_grid.r[-1])

    new_grid = construct_grid_from_points(ri_grid.r)
    print(type(new_grid))
    print(new_grid.h)
    # print(new_grid.r[-1])

    print(d_grid == new_grid)

