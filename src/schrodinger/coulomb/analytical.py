import numpy as np

from util.math.special_functions import confluent_hypergeometric_f, spherical_harmonic


def radial_function(n: int, l: int, r: np.array, Z: int, M: float = np.inf):
    """
    Radial component R_{n,l} of the analytical solution for the Schrödinger equation for a Coulomb-potential in atomic units.
    :param n: principal quantum number
    :param l: orbital angular momentum quantum number
    :param r: distance from nucleus in a_0
    :param Z: nuclear charge in e
    :param M: nuclear mass in m_e. The default is np.inf
    """
    if n <= 0:
        raise ValueError(f"Principal quantum number's allowed values are n=1,2,3,... but is '{n}'.")
    if not (0 <= l < n):
        raise ValueError(f"Orbital angular momentum quantum number's allowed values are l=0,1,...,n-1 but is '{l}'.")
    if any(r < 0):
        raise ValueError("Values passed to 'r' must be positive.")

    normalization_factor = np.sqrt(Z*np.math.factorial(n+l)/np.math.factorial(n-l-1))/(n*np.math.factorial(2*l+1))

    # calculate the reduced mass
    mu = 1/(1/M + 1)

    r_ = Z*mu*r/n
    return normalization_factor * (2*r_)**(l+1) * np.exp(-r_) * confluent_hypergeometric_f(-n+l+1, 2*l+2, 2*r_)


def R(n: int, l: int, r: np.array, Z: int, M):
    """
    Alias for radial_function.
    """
    return radial_function(n, l, r, Z, M)


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


def wave_function(n: int, l: int, m: int, r, theta, phi, Z: int, M: float = np.inf):
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
    """

    return 1 / r * R(n, l, r, Z, M) * Y(l, m, theta, phi)

def Psi(n: int, l: int, m: int, r, theta, phi, Z: int, M: float):
    """
    Alias for wave_function.
    """
    return wave_function(n, l, m, r, theta, phi, Z, M)


def energy(n: int, Z: int, M: float = np.inf):
    r"""
    Calculate the energy of state with quantum number n for hydrogen-like atom with charge Z and nuclear mass M.
    :param n: principal quantum number
    :param Z: nuclear charge in e
    :param M: nuclear mass in m_e
    :return:
    """
    # calculate the reduced mass
    mu = 1/(1/M + 1)
    return - Z**2 * mu / (2 * n**2)

def E(n: int, Z: int, M: float = np.inf):
    """
    Alias for energy.
    """
    return energy(n, Z, M)
