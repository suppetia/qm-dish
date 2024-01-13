import numpy as np


class RadialWaveFunction:

    def __init__(self, r_grid: np.array, Psi: np.array):

        assert len(r_grid) == len(Psi)
        self._r = r_grid
        self._psi = Psi
        if not r_grid[0] == 0:
            self._r = np.insert(self._r, 0, 0)
            self._psi = np.insert(self._psi, 0, 0, axis=0)

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


# TODO:
class RadialSchrödingerWaveFunction(RadialWaveFunction):
    def interpolate_at(self, r: np.array):
        """
        interpolate the wave function at points r
        :param r: np.array of points to interpolate on
        :return: interpolated wave_function
        """
        return RadialSchrödingerWaveFunction(r,
                                       np.array([
                                           np.interp(r, self.r, self.f),
                                           np.interp(r, self.r, self.g)
                                       ]).T
                                       )

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

    def write_to_file(self, filename):
        return np.savetxt(filename, np.array([self.r, self.f, self.g]).T,
                          header="r\tf\tg", delimiter="\t")
