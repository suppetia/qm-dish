import numpy as np

from dish.schrodinger.coulomb.analytical import radial_function as radial_f_schrodinger
from dish.dirac.coulomb.analytical import radial_function as radial_f_dirac

import re

from typing import Union, List


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
    r = np.logspace(-4, 4, num=300)
    R = radial_f_dirac(n, kappa, r, Z, M).f
    last_significant_r = r[np.max(np.argwhere(~np.isclose(R, 0, atol=1e-5)).reshape(-1))]

    N_max = int(np.floor(np.log(last_significant_r/r_0 + 1)/h)) - 1

    return int(N_max * 1.00) + 10  # *1.005 is just an arbitrary number to hold extra space


class QuantumNumberSet:

    def __init__(self, n, l, j):
        self._n = n
        self._l = l
        self._j = j

    @property
    def n(self):
        return self._n

    @property
    def l(self):
        return self._l

    @property
    def j(self):
        return self._j

    def __iter__(self):
        yield from [self.n, self.l, self.j]

    @property
    def kappa(self):
        return int(pow(-1, int(self.j-self.l+1/2)) * (self.j+1/2))

    def term_symbol(self):
        l_values = "spdfghiklmnoqrtuvwxyz"

        j = str(self.j) if int(self.j) == self.j else f"{int(self.j*2)}/2"
        if self.l >= len(l_values):
            raise ValueError(f"symbol undefined for l = {self.l}")

        return f"{self.n}{l_values[self.l]}{j}"

    def __repr__(self):
        return f"QuantumNumberSet(n={self.n}, l={self.l}, j={self.j}; kappa={self.kappa} => state={self.term_symbol()})"


def parse_atomic_term_symbol(symbol: str) -> QuantumNumberSet:
    """
    Parse the atomic term symbol into quantum numbers.
    :param symbol:
        format '<n><l><j>',
            where <n> is the integer value,
            <l> can be either the spectroscopic symbol (s,p,d,...) or an integer in brackets ([1], [2], ...)
            and <j> can be either an explicit integer/2 or +/- .
        examples: "2s1/2" == "2[0]+", "15d3/2" == "15[2]-"

    :return:
    """
    l_values = list("spdfghiklmnoqrtuvwxyz")

    pattern = re.compile(rf"(?P<n>[0-9]+)(?P<l>[{''.join(l_values)}]|(\[\d+\]))\_?(?P<j>([0-9]+(\/2)?)|([\+\-]))", re.IGNORECASE)

    match = re.match(pattern, symbol)
    if match is None:
        raise ValueError(f"invalid atomic term symbol '{symbol}")

    n = int(match.group("n"))
    if match.group("l").startswith("[") and match.group("l").endswith("]"):
        try:
            l = int(match.group("l")[1:-1])
        except Exception:
            raise ValueError(f"invalid value '{match.group('l')}' for 'l' - must be an integer")
    else:
        try:
            l = l_values.index(match.group("l"))
        except ValueError:
            raise ValueError(f"invalid symbol '{l}' for orbital quantum number")
    if match.group("j") in ["+", "-"]:
        j = l + eval(f"{match.group('j')}.5")
    else:
        j = float(eval(match.group("j")))
    # check if the state is valid
    if l >= n:
        raise ValueError(f"invalid value for orbital momentum quantum number l: '{l}'. Valid range: 0-{n-1}")
    if not (np.isclose(j, l+1/2) or np.isclose(j, l-1/2)):
        raise ValueError(f"invalid value for total angular momentum number j: '{j}.\nValid values are {l-1/2} and {l+1/2}.")
    return QuantumNumberSet(n, l, j)





if __name__ == "__main__":

    symbol = "2[1]1/2"
    n, l, j = parse_atomic_term_symbol(symbol)

    print(f"symbol '{symbol}': n={n}, l={l}, j={j}")
    print(f"generated symbol: {QuantumNumberSet(n,l,j).term_symbol()}")


