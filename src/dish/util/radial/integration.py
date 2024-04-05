from scipy.integrate import romb, trapezoid
import numpy as np
from mpmath import mp

from dish.util.radial.grid.grid import DistanceGrid, RombergIntegrationGrid
from dish.util.radial.wave_function import RadialWaveFunction
from dish.util.math_util.linear_algebra import matmul_pointwise

import logging
log = logging.getLogger(__name__)


def integrate_on_grid(y, grid: DistanceGrid, suppress_warning=False):
    if isinstance(grid, RombergIntegrationGrid):
        return romb(y*grid.rp, dx=grid.h)
    if not suppress_warning:
        log.warning("Integration is performed on an arbitrary DistanceGrid. For better results use a RombergIntegrationGrid.")
    return trapezoid(y*grid.rp, dx=grid.h)


def radial_integral(y, grid:DistanceGrid):
    return integrate_on_grid(y * grid.r**2, grid=grid)


def matrix_element(bra: RadialWaveFunction,
                   operator: np.ndarray,
                   ket: RadialWaveFunction) -> complex:
    """
    calculate the matrix element <bra|operator|ket>
    :param bra: RadialWaveFunction
    :param operator: np.array of same length as the bra-wavefunction
    :param ket: RadialWaveFunction on the same grid as the bra-wavefunction
    :return:
    """
    assert bra.grid == ket.grid
    assert bra.grid.N == operator.shape[0]

    ket = ket.Psi.astype(np.complex128)
    ket[:, 0] *= 1j
    new_ket = matmul_pointwise(operator, ket)
    bra_conj = bra.Psi.astype(np.complex128)
    bra_conj[:, 0] *= -1j
    return integrate_on_grid(np.sum(bra_conj * new_ket, axis=1), grid=bra.grid)


def mp_matrix_element(bra: RadialWaveFunction,
                      operator: np.ndarray,
                      ket: RadialWaveFunction) -> complex:
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

    integrand = np.zeros(bra.grid.N, dtype=np.complex128)
    ket = ket.Psi.astype(np.complex128)
    ket[:, 0] *= 1j
    bra_conj = bra.Psi.astype(np.complex128)
    bra_conj[:, 0] *= -1j
    for i in range(bra.grid.N):
        integrand[i] = mp.fdot(bra_conj[i], mp.matrix(operator[i, :, :]) * mp.matrix(ket[i]))

    return integrate_on_grid(integrand, grid=bra.grid)
