import numpy as np


class RadialDiracWaveFunction:

    def __init__(self, r_grid: np.array, Psi: np.array):

        assert len(r_grid) == len(Psi)
        self._r = r_grid
        self._psi = Psi

    @property
    def r(self):
        """
        :return: radial distance
        """
        return self._r

    @property
    def Psi(self):
        """
        :return: the values of the wave function at points r
        """
        return self._psi
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

    def interpolate_at(self, r: np.array):
        """
        interpolate the wave function at points r
        :param r: np.array of points to interpolate on
        :return: interpolated wave_function
        """
        return RadialDiracWaveFunction(r,
                                       np.array([
                                           np.interp(r, self.r, self.f),
                                           np.interp(r, self.r, self.g)
                                       ]).T
                                       )

# def overlap(psi1, psi2):
#

