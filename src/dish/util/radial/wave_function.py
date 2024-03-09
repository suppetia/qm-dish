import numpy as np
from scipy.interpolate import make_interp_spline

from typing import Union

from dish.util.radial.grid import DistanceGrid, construct_grid_from_points
from dish.util.atom import QuantumNumberSet


class RadialWaveFunction:

    def __init__(self,
                 r_grid: Union[np.array, DistanceGrid],
                 Psi: np.array,
                 state: QuantumNumberSet):

        if not isinstance(r_grid, DistanceGrid):
            r_grid = construct_grid_from_points(r_grid)

        assert len(r_grid.r) == len(Psi)
        self._grid = r_grid
        self._psi = Psi

        self._state = state

    @property
    def r(self):
        """
        :return: radial distance
        """
        return self._grid.r
    @property
    def grid(self):
        """
        :return: the grid of the wave function values
        """
        return self._grid

    @property
    def Psi(self):
        """
        :return: the values of the wave function at points r
        """
        return self._psi

    @property
    def state(self):
        return self._state
    @property
    def n(self):
        return self._state.n
    @property
    def l(self):
        return self._state.l
    @property
    def j(self):
        return self._state.j
    @property
    def kappa(self):
        return self._state.kappa

    def __getitem__(self, item):
        return self.Psi.__getitem__(item)


class RadialSchrodingerWaveFunction(RadialWaveFunction):

    def __init__(self, r_grid: Union[np.ndarray, DistanceGrid],
                 Psi: np.ndarray,
                 state: QuantumNumberSet,
                 Psi_prime: np.ndarray = None):

        if Psi_prime is None:
            temp_grid = DistanceGrid(r_grid.h, r_grid.r0, r_grid.N+1)
            Psi_prime = np.gradient(Psi, temp_grid.r[1:] - r_grid.r)
        self._psi_prime = Psi_prime
        super().__init__(r_grid=r_grid, Psi=Psi, state=state)

    @property
    def Psi(self):
        return self._psi

    @property
    def Psi_prime(self):
        return self._psi_prime

    def interpolate_at(self, r: Union[np.ndarray, DistanceGrid]):
        """
        interpolate the wave function at points r using a cubic spline
        :param r: np.array of points to interpolate on
        :return: interpolated wave_function
        """
        if not isinstance(r, DistanceGrid):
            r = construct_grid_from_points(r)
        spline_psi = make_interp_spline(self.r, self.Psi, k=3)
        return RadialSchrodingerWaveFunction(r,
                                             spline_psi(r.r),
                                             state=self.state,
                                             Psi_prime=spline_psi.derivative()(r))

    def write_to_file(self, filename):
        return np.savetxt(filename, np.array([self.r, self.Psi, self.Psi_prime]).T,
                          header="r\tPsi\tPsi'", delimiter="\t")


class RadialDiracWaveFunction(RadialWaveFunction):
    @property
    def f(self):
        """
        :return: large component of wave function
        """
        return self._psi[:, 0]
    @property
    def g(self):
        """
        :return: small component of wave function
        """
        return self._psi[:, 1]

    def __iter__(self):
        yield self.f
        yield self.g

    def interpolate_at(self, r: Union[np.ndarray, DistanceGrid]):
        """
        interpolate the wave function at points r using a cubic spline
        :param r: np.array of points to interpolate on
        :return: interpolated wave_function
        """
        if not isinstance(r, DistanceGrid):
            r = construct_grid_from_points(r)
        spline_f = make_interp_spline(self.r, self.f, k=3)
        spline_g = make_interp_spline(self.r, self.g, k=3)
        return RadialDiracWaveFunction(r,
                                       np.array([
                                           spline_f(r.r),
                                           spline_g(r.r)
                                       ]).T,
                                       state=self.state)

    def write_to_file(self, filename):
        return np.savetxt(filename, np.array([self.r, self.f, self.g]).T,
                          header="r\tf\tg", delimiter="\t")
