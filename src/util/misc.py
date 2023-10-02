import numpy as np
from schrodinger.coulomb.analytical import radial_function as radial_f_schrodinger
from dirac.coulomb.analytical import radial_function as radial_f_dirac

import re


def find_suitable_number_of_integration_points_schrodinger(Z, M, n, l, r_0, h):
    """
    Approximate the wave function by the solution of in a coulomb potential
    and find the last point r_max where the wave function is not close to zero.
    Use this point to find the N_max which is related to r_max via r_max = r_0 * (np.exp((N+1)*h) - 1)
    :param Z: nuclear charge in e
    :param M: nuclear mass in m_e
    :param n: principal quantum number
    :param l: orbital angular momentum quantum number
    :param r_0: parameter for the construction of r
    :param h: parameter for the construction of r
    :return: N_max
    """
    r = np.logspace(-2, 5, num=100)
    R = radial_f_schrodinger(n, l, r, Z, M)
    last_significant_r = r[np.max(np.argwhere(~np.isclose(R, 0, atol=1e-5)).reshape(-1))]

    N_max = int(np.floor(np.log(last_significant_r/r_0 + 1)/h)) - 1

    return int(N_max * 1.05)  # *1.05 is just an arbitrary number to hold extra space


def find_suitable_number_of_integration_points_dirac(Z, M, n, kappa, r_0, h):
    """
    Approximate the wave function by the solution of in a coulomb potential
    and find the last point r_max where the wave function is not close to zero.
    Use this point to find the N_max which is related to r_max via r_max = r_0 * (np.exp((N+1)*h) - 1)
    :param Z: nuclear charge in e
    :param M: nuclear mass in m_e
    :param n: principal quantum number
    :param kappa: ? quantum number
    :param r_0: parameter for the construction of r
    :param h: parameter for the construction of r
    :return: N_max
    """
    r = np.logspace(-2, 5, num=100)
    R = radial_f_dirac(n, kappa, r, Z, M)
    last_significant_r = r[np.max(np.argwhere(~np.isclose(R, 0, atol=1e-5)).reshape(-1))]

    N_max = int(np.floor(np.log(last_significant_r/r_0 + 1)/h)) - 1

    return int(N_max * 1.05)  # *1.05 is just an arbitrary number to hold extra space


def parse_atomic_term_symbol(symbol: str):
    l_values = list("spdfghilmnopqrtuvwxyz")

    pattern = re.compile(rf"(?P<n>[0-9]+)(?P<l>[{''.join(l_values)}])(?P<j>[0-9]+(\/2)?)", re.IGNORECASE)

    match = re.match(pattern, symbol)
    if match is None:
        raise ValueError(f"invalid atomic term symbol '{symbol}")

    n = int(match.group("n"))
    try:
        l = l_values.index(match.group("l"))
    except ValueError:
        raise ValueError(f"invalid symbol '{l}' for orbital quantum number")
    j = float(eval(match.group("j")))
    # check if the state is valid
    if l >= n:
        raise ValueError(f"invalid value for orbital momentum quantum number l: '{l}'. Valid range: 0-{n-1}")
    if not (np.isclose(j, l+1/2) or np.isclose(j, l-1/2)):
        raise ValueError(f"invalid value for total angular momentum number j: '{j}.\nValid values are {l-1/2} and {l+1/2}.")
    return n, l, j


if __name__ == "__main__":

    symbol = "2p7/2"
    n, l, j = parse_atomic_term_symbol(symbol)

    print(f"symbol '{symbol}': n={n}, l={l}, j={j}")


