import numpy as np
import math

from dish.util.math_util.special_functions import confluent_hypergeometric_f, spherical_harmonic
from dish.util.radial.wave_function import RadialSchrodingerWaveFunction
from dish.util.atom import QuantumNumberSet
from dish.util.radial.grid.grid import DistanceGrid

from typing import Union


def radial_function(n: int, l: int, r: Union[DistanceGrid, np.ndarray], Z: int, M: float = np.inf, m_particle: float = 1):
    """
    Radial component R_{n,l} of the analytical solution for the Schrödinger equation for a Coulomb-potential in atomic units.
    :param n: principal quantum number
    :param l: orbital angular momentum quantum number
    :param r: distance from nucleus in a_0
    :param Z: nuclear charge in e
    :param M: nuclear mass in m_e. The default is np.inf
    :param m_particle: mass of the particle for which the SE is solved in m_e. The default is 1 (for an electron).
    """
    r_grid = r
    if isinstance(r, DistanceGrid):
        r = r_grid.r

    if n <= 0:
        raise ValueError(f"Principal quantum number's allowed values are n=1,2,3,... but is '{n}'.")
    if not (0 <= l < n):
        raise ValueError(f"Orbital angular momentum quantum number's allowed values are l=0,1,...,n-1 but is '{l}'.")
    if any(r < 0):
        raise ValueError("Values passed to 'r' must be positive.")

    normalization_factor = np.sqrt(Z*math.factorial(n+l)/math.factorial(n-l-1))/(n*math.factorial(2*l+1))

    # calculate the reduced mass
    mu = m_particle/(m_particle/M + 1)

    r_ = Z*mu*r/n
    R = normalization_factor * (2*r_)**(l+1) * np.exp(-r_) * confluent_hypergeometric_f(-n+l+1, 2*l+2, 2*r_)
    return RadialSchrodingerWaveFunction(r_grid=r_grid, state=QuantumNumberSet(n=n, l=l), Psi=R)


def R(n: int, l: int, r: np.ndarray, Z: int, M: float, m_particle: float):
    """
    Alias for radial_function.
    """
    return radial_function(n, l, r, Z, M, m_particle)


def spherical_function(l: int, m: int, theta, phi):
    """
    Alias for util.math.special_functions.spherical_harmonic
    """
    return spherical_harmonic(l, m, theta, phi)

def Y(l: int, m: int, theta, phi):
    """
    Alias for spherical_function
    """
    return spherical_function(l, m, theta, phi)


def wave_function(n: int, l: int, m: int, r, theta, phi, Z: int, M: float = np.inf, m_particle: float = 1):
    """
    Return the wave function with quantum numbers n,l,m evaluated at points (r,theta,phi) for a hydrogen-like atom with charge
    :param n: principal quantum number
    :param l: orbital angular momentum quantum number
    :param m: magnetic momentum quantum number
    :param r: distance from nucleus in a_0
    :param theta: polar angle in radian
    :param phi: azimuthal angle in radian
    :param Z: nuclear charge in e
    :param M: nuclear mass in m_e
    :param m_particle: mass of the particle of which the wave function is returned in m_e
    """

    return 1 / r * R(n, l, r, Z, M, m_particle).R * Y(l, m, theta, phi)

def Psi(n: int, l: int, m: int, r, theta, phi, Z: int, M: float, m_particle: float = 1):
    """
    Alias for wave_function.
    """
    return wave_function(n, l, m, r, theta, phi, Z, M, m_particle)


def energy(n: int, Z: int, M: float = np.inf, m_particle: float = 1):
    r"""
    Calculate the energy of state with quantum number n for hydrogen-like atom with charge Z and nuclear mass M.
    :param n: principal quantum number
    :param Z: nuclear charge in e
    :param M: nuclear mass in m_e
    :param m_particle: the particles mass in m_e
    :return:
    """
    # calculate the reduced mass
    mu = m_particle/(m_particle/M + 1)
    return - Z**2 * mu / (2 * n**2)

def E(n: int, Z: int, M: float = np.inf, m_particle: float = 1):
    """
    Alias for energy.
    """
    return energy(n, Z, M, m_particle)
