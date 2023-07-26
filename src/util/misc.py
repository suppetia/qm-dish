import numpy as np
from schrodinger.coulomb.analytical import radial_function


def find_suitable_number_of_integration_points(Z, M, n, l,
                                               r_0=0.0005, h=0.02):
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
    r = np.logspace(-2, 5, num=50)
    R = radial_function(n, l, r, Z, M)
    last_significant_r = r[np.max(np.argwhere(~np.isclose(R, 0, atol=1e-5)).reshape(-1))]

    N_max = np.log(last_significant_r/r_0 + 1)/h - 1

    return N_max + 10  # + 10 is just an arbitrary number to hold extra space

