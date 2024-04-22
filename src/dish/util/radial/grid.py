import numpy as np
from scipy.optimize import curve_fit

from typing import Union

from dish.util.atom import Nucleus, QuantumNumberSet


class DistanceGrid:
    """
    Class to hold the information about the grid and the grid points itself.
    The grid is constructed lazily, i.e. the grid points are calculated the first time they are required.
    The grid has the form :math:`r(t) = r0 * \\left(\\e{t} - 1\\right)` where :math:`t(i) = i*h` is a linear grid with ``N`` points.
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
    The grid has the form :math:`r(t) = r0 * \\left(\\e{t} - 1\\right)` where :math:`t(i) = i*h` is a linear grid with ``N`` points.

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
        :param grid:
        :return:
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


def construct_grid_from_dict(r_grid: dict,
                             nucleus: Nucleus,
                             state: QuantumNumberSet,
                             *, relativistic=True):
    """
    Construct a DistanceGrid (or RombergIntegrationGrid if possible) from the parameters of the grid given in the dict ``r_grid``.
    If not provided the values default to ``h = 5e-3``, ``r0 = 2e-6``, ``N = "auto"``.
    If ``N = "auto"`` the number is derived from the ``state`` as a suited value from the analytic solution.
    :param r_grid: parameter dict to construct the grid from
    :param nucleus: The nucleus for which the radial SE/DE should be solved. Only actually required if ``N = "auto"``.
    :param state:
    :param relativistic:
    :return:
    """
    h = r_grid.get("h", 0.005)
    r0 = r_grid.get("r0", 2e-6)
    if r_grid.get("r_max") is not None:
        return DistanceGrid(h, r0, r_max=r_grid.get("r_max"))
    elif r_grid.get("N", "auto") == "auto":
        if relativistic:
            from dish.util.misc import find_suitable_number_of_integration_points_dirac

            N = find_suitable_number_of_integration_points_dirac(Z=nucleus.Z,
                                                                 M=nucleus.M,
                                                                 n=state.n,
                                                                 kappa=state.kappa,
                                                                 r_0=r0,
                                                                 h=h)
        else:
            from dish.util.misc import find_suitable_number_of_integration_points_schrodinger

            N = find_suitable_number_of_integration_points_schrodinger(Z=nucleus.Z,
                                                                       M=nucleus.M,
                                                                       n=state.n,
                                                                       l=state.l,
                                                                       r_0=r0,
                                                                       h=h)
    else:
        try:
            N = int(r_grid['N'])
        except Exception:
            raise ValueError(f"Number of grid points 'N' must be an integer but is {type(r_grid['N'])}")

    k = np.log2(N - 1)
    if np.isclose(k - np.floor(k), 0):
        return RombergIntegrationGrid(h, r0, N)
    else:
        return DistanceGrid(h, r0, N)


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

