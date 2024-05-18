import numpy as np
from scipy.special import gamma as gamma_f
from math import factorial

from dish.util.math_util.special_functions import confluent_hypergeometric_f
from dish.util.atomic_units import alpha, c
from dish.util.atom import QuantumNumberSet
from dish.util.radial.wave_function import RadialDiracWaveFunction
from dish.util.radial.grid.grid import DistanceGrid

from typing import Union


def energy(n: int, kappa: int, Z: int, m_particle: float = 1):
    r"""
    Calculate the energy of state with quantum number n for hydrogen-like atom with charge Z and the particles mass m_particle.
    :param n: principal quantum number
    :param kappa: ? quantum number
    :param Z: nuclear charge in e
    :param m_particle: mass of the particle in m_e
    :return:
    """
    gamma = np.sqrt(kappa**2-(alpha*Z)**2)
    return m_particle*c**2/np.sqrt(1+(alpha*Z)**2/(gamma+n-abs(kappa))**2)

def reduced_energy(n: int, kappa: int, Z: int, m_particle: float = 1):
    r"""
    Calculate the energy of state with quantum number n for hydrogen-like atom with charge Z and nuclear mass M.
    The electron rest-energy is subtracted
    :param n: principal quantum number
    :param kappa: ? quantum number
    :param Z: nuclear charge in e
    :param m_particle: mass of the particle in m_e
    :return:
    """
    return energy(n,kappa,Z, m_particle) - m_particle* c**2

def radial_function(n: int, kappa: int, r: Union[DistanceGrid, np.ndarray], Z: int, m_particle: float = 1):
    """
    Radial component R_{n,l} of the analytical solution for the Dirac equation for a Coulomb-potential in atomic units.
    :param n: principal quantum number
    :param kappa: dirac quantum number
    :param r: distance from nucleus in a_0
    :param Z: nuclear charge in e
    :param m_particle: mass of the particle in m_e
    """

    r_grid = r
    if isinstance(r, DistanceGrid):
        r = r_grid.r

    if n <= 0:
        raise ValueError(f"Principal quantum number's allowed values are n=1,2,3,... but is '{n}'.")
    if abs(kappa) > n:
        raise ValueError(f"|kappa| <= n is required but n={n} and kappa={kappa}")
    if abs(kappa) == n and kappa > 0:
        raise ValueError(f"eigenfunction for |kappa| = n only exists for kappa < 0 ")
    if kappa == 0:
        raise ValueError("kappa can't be zero")
    if any(r < 0):
        raise ValueError("Values passed to 'r' must be positive.")

    E = energy(n, kappa, Z, m_particle)

    gamma = np.sqrt(kappa**2-(alpha*Z)**2)

    k = abs(kappa)
    # generalized principal quantum number
    N = np.sqrt(n**2-2*(n-k)*(k-gamma))

    #lambda_ = np.sqrt((m_particle*c)**2-(E/c)**2)
    lambda_ = Z*E/((n-k+gamma)*c**2)
    x = 2 * lambda_ * r

    # normalization factor
    N_nkappa = np.sqrt(m_particle*(Z*gamma_f(2*gamma+1+n-k))/(2*factorial(n-k)*(N-kappa))) / (N*gamma_f(2*gamma+1))

    F1 = (N-kappa) * confluent_hypergeometric_f(k-n, 2*gamma+1, x)
    F2 = 0 if k-n == 0 else (k-n) * confluent_hypergeometric_f(k-n+1, 2*gamma+1, x)
    P = N_nkappa * np.sqrt(1+E/(m_particle*c**2)) * np.exp(-x/2) * np.power(x, gamma) * (F1 + F2)
    Q = N_nkappa * np.sqrt(1-E/(m_particle*c**2)) * np.exp(-x/2) * np.power(x, gamma) * (F1 - F2)
    return RadialDiracWaveFunction(r_grid=r_grid, Psi=np.stack((P, Q)).T, state=QuantumNumberSet(n=n, kappa=kappa))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dish.util.radial import DistanceGrid

    Z = 137
    n = 1
    kappa = -1
    r = DistanceGrid(r0=1e-15, h=1e-2, r_max=1e-3)

    P, Q = radial_function(n=n, kappa=kappa, r=r.r, Z=Z, m_particle=200)

    fig, ax = plt.subplots(2)
    ax[0].plot(r.r, P)
    ax[1].plot(r.r, Q)

    plt.show()
    print(np.trapz(P**2+Q**2, x=r.r))
