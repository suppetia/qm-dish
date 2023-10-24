import numpy as np
from scipy.special import gamma as gamma_f
from math import factorial

from util.math.special_functions import spherical_harmonic, confluent_hypergeometric_f
from util.atomic_units import alpha, c


def energy(n: int, kappa: int, Z: int, M: float = np.inf):
    r"""
    Calculate the energy of state with quantum number n for hydrogen-like atom with charge Z and nuclear mass M.
    :param n: principal quantum number
    :param kappa: ? quantum number
    :param Z: nuclear charge in e
    :param M: nuclear mass in m_e
    :return:
    """
    # calculate the reduced mass
    mu = 1/(1/M + 1)

    gamma = np.sqrt(kappa**2-(alpha*Z)**2)
    return c**2/np.sqrt(1+(alpha*Z*mu)**2/(gamma+n-abs(kappa))**2)


def radial_function(n: int, kappa: int, r: np.array, Z: int, M: float = np.inf):
    """
    Radial component R_{n,l} of the analytical solution for the Schrödinger equation for a Coulomb-potential in atomic units.
    :param n: principal quantum number
    :param kappa: ? quantum number
    :param r: distance from nucleus in a_0
    :param Z: nuclear charge in e
    :param M: nuclear mass in m_e. The default is np.inf
    """

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

    # calculate the reduced mass
    mu = 1/(1/M + 1)

    E = energy(n, kappa, Z, M)

    gamma = np.sqrt(kappa**2-(alpha*Z*mu)**2)

    k = abs(kappa)
    # generalized principal quantum number
    N = np.sqrt(n**2-2*(n-k)*(k-gamma))

    #lambda_ = np.sqrt(c**2-(E/c)**2)
    lambda_ = Z*mu*E/((n-k+gamma)*c**2)
    x = 2 * lambda_ * r

    # normalization factor
    N_nkappa = np.sqrt((Z*mu*gamma_f(2*gamma+1+n-k))/(2*factorial(n-k)*(N-kappa))) / (N*gamma_f(2*gamma+1))

    F1 = (N-kappa) * confluent_hypergeometric_f(k-n, 2*gamma+1, x)
    F2 = 0 if k-n == 0 else (k-n) * confluent_hypergeometric_f(k-n+1, 2*gamma+1, x)
    P = N_nkappa * np.sqrt(1+E/c**2) * np.exp(-x/2) * np.power(x, gamma) * (F1 + F2)
    Q = N_nkappa * np.sqrt(1-E/c**2) * np.exp(-x/2) * np.power(x, gamma) * (F1 - F2)
    return np.stack((P, Q))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    Z = 2
    n = 2
    kappa = -1
    r = 1.2*np.logspace(-2, 1, num=5000)

    P, Q = radial_function(n=n, kappa=kappa, r=r, Z=Z, M=np.inf)

    fig, ax = plt.subplots(2)
    ax[0].plot(r, P)
    ax[1].plot(r, Q)

    plt.show()
    print(np.trapz(P**2+Q**2, x=r))
