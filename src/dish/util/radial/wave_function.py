import numpy as np
from scipy.interpolate import make_interp_spline

from typing import Union
from abc import ABC, abstractmethod

from dish.util.radial.grid.grid import DistanceGrid
from dish.util.radial.grid.construct_grid import construct_grid_from_points


class RadialWaveFunction(ABC):
    """
    Base class to store wave functions for a specific state.
    Stores the following information:
    - r_grid: A dish.util.radial.grid.DistanceGrid on which the wave function is evaluated.
    - Psi: The values of the radial wave function. This is a multidimensional array for relativistic wave functions.
    - state: The associated state as a dish.util.atom.QuantumNumberSet.
    """

    def __init__(self,
                 r_grid: Union[np.array, DistanceGrid],
                 Psi: np.array,
                 state: "dish.util.atom.QuantumNumberSet"):

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

    def _get_state(self):
        if self._state is None:
            raise ValueError("'state' of this wave function is not defined. Probably the wave function is a result of an operator applied and therefore not neccessarily an energy eigenfunction.")
        return self._state
    @property
    def state(self):
        return self._state  # this might return None
    @property
    def n(self):
        return self._get_state().n
    @property
    def l(self):
        return self._get_state().l
    @property
    def j(self):
        return self._get_state().j
    @property
    def kappa(self):
        return self._get_state().kappa

    def __getitem__(self, item):
        return self.Psi.__getitem__(item)

    @abstractmethod
    def interpolate_at(self, r: Union[np.ndarray, DistanceGrid]):
        """
        interpolate the wave function at points r using a cubic spline
        :param r: np.ndarray of points or dish.util.radial.grid.DistanceGrid to interpolate on
        :return: interpolated wave_function
        """
        ...

    @abstractmethod
    def interpolate_values(self, r: Union[np.ndarray, DistanceGrid]):
        """
        Interpolate the wave function at points r using a cubic spline.
        Use this method to interpolate on arbitrary grid points.
        :param r: np.ndarray of points or dish.util.radial.grid.DistanceGrid to interpolate on
        :return: the values of the interpolated wave function
        """
        ...

    @abstractmethod
    def write_to_file(self, filename):
        """
        store the values of the radial wave function in a comma separated file
        :param filename: name of the file
        """
        ...


class RadialSchrodingerWaveFunction(RadialWaveFunction):
    """
    Class to store non-relativistic wave functions for a specific state that solves the radial Schrödinger equation.
    Stores the following information:
    - r_grid: A dish.util.radial.grid.DistanceGrid on which the wave function is evaluated.
    - Psi: The values of the radial wave function.
    - Psi_prime: The values of the derivative of the radial wave function.
    - state: The associated state as a dish.util.atom.QuantumNumberSet.
    """

    def __init__(self, r_grid: Union[np.ndarray, DistanceGrid],
                 Psi: np.ndarray,
                 state: "dish.util.atom.QuantumNumberSet",
                 Psi_prime: np.ndarray = None):

        if not isinstance(r_grid, DistanceGrid):
            r_grid = construct_grid_from_points(r_grid)

        if Psi_prime is None:
            temp_grid = DistanceGrid(r_grid.h, r_grid.r0, r_grid.N+1)
            Psi_prime = np.gradient(Psi, temp_grid.r[1:] - r_grid.r)
        self._psi_prime = Psi_prime
        super().__init__(r_grid=r_grid, Psi=Psi, state=state)

    @property
    def Psi(self):
        return self._psi

    @property
    def R(self):
        return self._psi

    @property
    def Psi_prime(self):
        return self._psi_prime

    @property
    def Q(self):
        return self._psi_prime

    def interpolate_at(self, r: Union[np.ndarray, DistanceGrid]):
        """
        interpolate the wave function at points r using a cubic spline
        :param r: np.ndarray of points or dish.util.radial.grid.DistanceGrid to interpolate on
        :return: interpolated wave_function
        """
        if not isinstance(r, DistanceGrid):
            r = construct_grid_from_points(r)
        spline_psi = make_interp_spline(self.r, self.Psi, k=3)
        return RadialSchrodingerWaveFunction(r,
                                             spline_psi(r.r),
                                             state=self.state,
                                             Psi_prime=spline_psi.derivative()(r))

    def interpolate_values(self, r: Union[np.ndarray, DistanceGrid]):
        """
        Interpolate the wave function at points r using a cubic spline.
        Use this method to interpolate on arbitrary grid points.
        :param r: np.ndarray of points or dish.util.radial.grid.DistanceGrid to interpolate on
        :return: the values of the interpolated wave function and it's derivative
        """
        if isinstance(r, DistanceGrid):
            r = r.r
        spline_psi = make_interp_spline(self.r, self.Psi, k=3)
        return np.array([spline_psi(r), spline_psi.derivative()(r)]).T

    def write_to_file(self, filename):
        return np.savetxt(filename, np.array([self.r, self.Psi, self.Psi_prime]).T,
                          header="r\tPsi\tPsi'", delimiter="\t")

    def __add__(self, other):
        if not isinstance(other, RadialSchrodingerWaveFunction):
            return NotImplemented
        if not self.grid == other.grid:
            raise ValueError("wave functions to add must be defined on the same grid")

        return RadialSchrodingerWaveFunction(r_grid=self.grid,
                                             Psi=self.R + other.R,
                                             Psi_prime=self.Q + other.Q,
                                             state=None)


class RadialDiracWaveFunction(RadialWaveFunction):
    """
    Class to store relativistic wave functions for a specific state that solves the radial Dirac equation.
    Stores the following information:
    - r_grid: A dish.util.radial.grid.DistanceGrid on which the wave function is evaluated.
    - Psi: The values of the radial wave function. A two-dimensional array.

      - f: The large component.
      - g: The small component.

    - state: The associated state as a dish.util.atom.QuantumNumberSet.
    """
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
        Interpolate the wave function at a DistanceGrid r using a cubic spline.
        Use this method to interpolate on arbitrary grid points.
        :param r: np.ndarray of points or dish.util.radial.grid.DistanceGrid to interpolate on
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

    def interpolate_values(self, r: Union[np.ndarray, DistanceGrid]):
        """
        interpolate the wave function at points r using a cubic spline
        :param r: np.ndarray of points or dish.util.radial.grid.DistanceGrid to interpolate on
        :return: the values of the interpolated wave function
        """
        if isinstance(r, DistanceGrid):
            r = r.r
        spline_f = make_interp_spline(self.r, self.f, k=3)
        spline_g = make_interp_spline(self.r, self.g, k=3)
        return np.array([spline_f(r), spline_g(r)]).T

    def write_to_file(self, filename):
        """
        store the values of the radial wave function in a comma separated file
        :param filename: name of the file
        """
        return np.savetxt(filename, np.array([self.r, self.f, self.g]).T,
                          header="r\tf\tg", delimiter="\t")

    def __add__(self, other):
        if not isinstance(other, RadialDiracWaveFunction):
            return NotImplemented
        if not self.grid == other.grid:
            raise ValueError("wave functions to add must be defined on the same grid")

        return RadialDiracWaveFunction(r_grid=self.grid,
                                       Psi=self._psi + other._psi,
                                       state=None)
