from scipy.integrate import romb, trapezoid
import numpy as np
from mpmath import mp

from dish.util.radial.grid import DistanceGrid, RombergIntegrationGrid
from dish.util.radial.wave_function import RadialWaveFunction
from dish.util.math_util.linear_algebra import matmul_pointwise

import logging
log = logging.getLogger(__name__)


def integrate_on_grid(y, grid: DistanceGrid):
    if isinstance(grid, RombergIntegrationGrid):
        return romb(y*grid.rp, dx=grid.h)

    log.warning("Integration is performed on an arbitrary DistanceGrid. For better results use a RombergIntegrationGrid.")
    return trapezoid(y*grid.rp, dx=grid.h)


def radial_integral(y, grid:DistanceGrid):
    return integrate_on_grid(y * grid.r**2, grid=grid)


def matrix_element(bra: RadialWaveFunction,
                   operator: np.ndarray,
                   ket: RadialWaveFunction) -> float:
    """
    calculate the matrix element <bra|operator|ket>
    :param bra: RadialWaveFunction
    :param operator: np.array of same length as the bra-wavefunction
    :param ket: RadialWaveFunction on the same grid as the bra-wavefunction
    :return:
    """
    assert bra.grid == ket.grid
    assert bra.grid.N == operator.shape[0]

    new_ket = matmul_pointwise(operator, ket.Psi)
    return radial_integral(np.sum(bra.Psi * new_ket, axis=1), grid=bra.grid)


def mp_matrix_element(bra: RadialWaveFunction,
                      operator: np.ndarray,
                      ket: RadialWaveFunction) -> float:
    """
    calculate the matrix element <bra|operator|ket>
        using arbitrary-precision floating-point arithmetic to calculate the integrand
    :param bra: RadialWaveFunction
    :param operator: np.array of same length as the bra-wavefunction
    :param ket: RadialWaveFunction on the same grid as the bra-wavefunction
    :return:
    """
    assert bra.grid == ket.grid
    assert bra.grid.N == operator.shape[0]

    integrand = np.zeros(bra.grid.N)
    for i in range(bra.grid.N):
        integrand[i] = mp.fdot(bra.Psi[i], mp.matrix(operator[i, :, :]) * mp.matrix(ket.Psi[i]))

    return radial_integral(integrand, grid=bra.grid)
