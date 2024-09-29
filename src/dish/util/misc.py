import numpy as np

from dish.schrodinger.coulomb.analytical import radial_function as radial_f_schrodinger
from dish.dirac.coulomb.analytical import radial_function as radial_f_dirac

from dish.util.atom import Nucleus, QuantumNumberSet
from dish.util.radial.grid import DistanceGrid
from dish.util.radial.wave_function import RadialWaveFunction

from dataclasses import dataclass


def find_suitable_number_of_integration_points_schrodinger(Z, M, m_particle, n, l, r_0, h):
    """
    Approximate the wave function by the solution of in a coulomb potential
    and find the last point r_max where the wave function is not close to zero.
    Use this point to find the N_max which is related to r_max via r_max = r_0 * (np.exp((N+1)*h) - 1)
    :param Z: nuclear charge in e
    :param M: nuclear mass in m_e
    :param m_particle: mass of the particle in m_e
    :param n: principal quantum number
    :param l: orbital angular momentum quantum number
    :param r_0: parameter for the construction of r
    :param h: parameter for the construction of r
    :return: N_max
    """
    r = np.logspace(-4, 5, num=500)
    R = radial_f_schrodinger(n, l, r, Z, M, m_particle)
    last_significant_r = r[np.max(np.argwhere(~np.isclose(R.R, 0, atol=1e-10)).reshape(-1))]

    N_max = int(np.floor(np.log(last_significant_r/r_0 + 1)/h)) - 1

    return int(N_max * 1.05)  # *1.05 is just an arbitrary number to hold extra space


def find_suitable_number_of_integration_points_dirac(Z, m_particle, n, kappa, r_0, h):
    """
    Approximate the wave function by the solution of in a coulomb potential
    and find the last point r_max where the wave function is not close to zero.
    Use this point to find the N_max which is related to r_max via r_max = r_0 * (np.exp((N+1)*h) - 1)
    :param Z: nuclear charge in e
    :param m_particle: mass of the particle in m_e
    :param n: principal quantum number
    :param kappa: ? quantum number
    :param r_0: parameter for the construction of r
    :param h: parameter for the construction of r
    :return: N_max
    """
    r = np.logspace(-4, 5, num=500)
    R = radial_f_dirac(n, kappa, r, Z, m_particle=m_particle).f
    last_significant_r = r[np.max(np.argwhere(~np.isclose(R, 0, atol=1e-10)).reshape(-1))]

    N_max = int(np.floor(np.log(last_significant_r/r_0 + 1)/h)) - 1

    return N_max + 10  # +10 is just an arbitrary number from experience to hold extra space


@dataclass(frozen=True)
class SolvingParameters:
    order_AM: int
    order_in: int
    max_number_of_iterations: int


@dataclass(frozen=True)
class SolvingResult:
    state: QuantumNumberSet
    nucleus: Nucleus
    potential_model: str
    m: float
    r_grid: DistanceGrid
    wave_function: RadialWaveFunction
    energy: float
    energy_convergence: list
    solving_parameters: SolvingParameters
    number_of_iterations: int
    solving_time: float


