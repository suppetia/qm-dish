import numpy as np
# import yaml
import re
from typing import Union

from dataclasses import dataclass

from dish.util.atomic_units import a_0
from dish.util.potential import FermiPotential, CoulombPotential, UniformBallPotential
from dish.util.potential import FermiChargeDistribution, CoulombChargeDistribution, UniformBallChargeDistribution


@dataclass
class Nucleus:
    """
    | Dataclass to store the properties of the nucleus.
    | The potential for point-like, ball-like and Fermi charge distributions is provided through the :meth:`potential` method.

    :param Z: number of protons in the nucleus == nuclear charge
    :param M: mass of the nucleus.
    :param R0: radius of the nucleus. For a Fermi charge distribution this is to the parameter *c*.
    :param a: (optional) diffuseness parameter for a Fermi distribution. Defaults to :math:`2.3 \\text{fm} /a_0 / (4\cdot\ln(3))` as described in *Parpia and Mohanty, Phys.Rev.A, 46 (1992), Number 7*.
    :param charge: (optional) charge of the hydrogenic atom. Defaults to *Z-1*.
    """
    Z: float
    M: float
    R0: float
    a: float = 2.3e-15/a_0 / (4*np.log(3))  # default value from Parpia and Mohanty, Phys.Rev.A, 46 (1992), Number 7
    charge: float = None  # usually just required by non-relativistic calculations, defaults to Z-1

    def __post_init__(self):
        if self.charge is None:
            self.charge = self.Z - 1

    @property
    def mu(self):
        return 1/(1+1/self.M)

    @classmethod
    def construct_from_name(cls, name):
        # load from yaml file
        ...

    def potential(self, r: Union[float, np.ndarray], model="Fermi"):
        """
        :param r: point(s) where the value of the potential should be evaluated
        :param model: (optional) alias for the type of the potential model. Choose from "Fermi", "ball-like", "point-like"=="Coulomb". Defaults to "Fermi".
        :return: Values of the nucleus' potential in distance *r*
        :rtype: float or np.ndarray, same as *r*
        """
        if model.lower() in ["f", "fermi"]:
            return FermiPotential(self)(r)
        elif model.lower() in ["u", "uniform", "ball", "ball-like", "balllike", "uniformball"]:
            return UniformBallPotential(self)(r)
        elif model.lower() in ["point", "point-like", "pointlike", "p", "coulomb", "c"]:
            return CoulombPotential(self)(r)
        else:
            raise ValueError(
                f"'model' must be either 'Fermi', 'uniform' or 'point-like' but is {model}.")

    @property
    def CoulombPotential(self):
        return CoulombPotential(self)
    @property
    def FermiPotential(self):
        return FermiPotential(self)
    @property
    def UniformBallPotential(self):
        return UniformBallPotential(self)

    def charge_density(self, r, model="Fermi"):
        if model.lower() in ["f", "fermi"]:
            return FermiChargeDistribution(self)(r)
        elif model.lower() in ["u", "uniform", "ball", "uniformball"]:
            return UniformBallChargeDistribution(self)(r)
        elif model.lower() in ["point", "point-like", "pointlike", "p", "coulomb", "c"]:
            return CoulombChargeDistribution(self)(r)
        else:
            raise ValueError(
                f"'model' must be either 'Fermi', 'uniform' or 'point-like' but is {model}.")

    @property
    def FermiChargeDistribution(self):
        return FermiChargeDistribution(self)

    @property
    def CoulombChargeDistribution(self):
        return CoulombChargeDistribution(self)

    @property
    def UniformBallChargeDistribution(self):
        return UniformBallChargeDistribution(self)


class QuantumNumberSet:
    """
    A container class to store the quantum numbers n, l and j.
    :math:`\kappa` is calculated for convenience.

    :param n: principal quantum number
    :param l: orbital angular momentum quantum number
    :param j: (optional) total angular momentum quantum number. For non-relativistic calculations not defined.

    """

    def __init__(self, n, l=None, j=None, *, kappa=None):
        self._n = n
        if l is not None:
            self._l = l
        self._j = j

        if kappa is not None:
            if kappa < 0:
                j_ = abs(kappa) - .5
                l_ = j_ - .5
            else:
                j_ = kappa - .5
                l_ = j_ + .5
            if l is not None and l != l_:
                raise ValueError(f"'kappa' was specified and does not match with value passed to 'l'")
            if j is not None and j != j_:
                raise ValueError(f"'kappa' was specified and does not match with value passed to 'j'")
            self._l = l_
            self._j = j_

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
        if self.j is not None:
            yield from [self.n, self.l, self.j]
        else:
            yield from [self.n, self.l]

    @property
    def kappa(self):
        if self.j is not None:
            return int(pow(-1, int(self.j-self.l+1/2)) * (self.j+1/2))
        return None

    def term_symbol(self):
        l_values = "spdfghiklmnoqrtuvwxyz"
        if self.j is not None:
            j = str(self.j) if int(self.j) == self.j else f"{int(self.j*2)}/2"
        else:
            j = ""
        if self.l >= len(l_values):
            raise ValueError(f"symbol undefined for l = {self.l}")

        return f"{self.n}{l_values[self.l]}{j}"

    def __repr__(self):
        if self.j is None:
            return f"QuantumNumberSet(n={self.n}, l={self.l} => state={self.term_symbol()})"
        return f"QuantumNumberSet(n={self.n}, l={self.l}, j={self.j}; kappa={self.kappa} => state={self.term_symbol()})"

    def __eq__(self, other):
        if isinstance(other, str):
            other = parse_atomic_term_symbol(other)
        elif isinstance(other, tuple):
            other = QuantumNumberSet(*other)
        elif not isinstance(other, QuantumNumberSet):
            return False
        if self.j is None:
            return self.n == other.n and self.l == other.l
        return self.n == other.n and self.l == other.l and self.j == other.j


def parse_atomic_term_symbol(symbol: str) -> QuantumNumberSet:
    """
    Parse the atomic term symbol into quantum numbers.

    :param symbol: spectroscopic representation of the electronic state

        format: '<n><l><j>',
        where <n> is the integer value,
        <l> can be either the spectroscopic symbol (s,p,d,...) or an integer in brackets ([1], [2], ...)
        and <j> can be either an explicit integer/2 or +/- .

        :Example: "2s1/2" == "2[0]+", "15d3/2" == "15[2]-"

    :rtype: QuantumNumberSet
    """
    l_values = list("spdfghiklmnoqrtuvwxyz")

    pattern = re.compile(rf"(?P<n>[0-9]+)(?P<l>[{''.join(l_values)}]|(\[\d+\]))\_?(?P<j>([0-9]+(\/2)?)|([\+\-]))?", re.IGNORECASE)

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
    elif match.group("j") is None:
        j = None
    else:
        j = float(eval(match.group("j")))
    # check if the state is valid
    if l >= n:
        raise ValueError(f"invalid value for orbital momentum quantum number l: '{l}'. Valid range: 0-{n-1}")
    if j is not None and not (np.isclose(j, l+1/2) or np.isclose(j, l-1/2)):
        raise ValueError(f"invalid value for total angular momentum number j: '{j}.\nValid values are {l-1/2} and {l+1/2}.")
    return QuantumNumberSet(n, l, j)



if __name__ == "__main__":

    symbol = "2[1]1/2"
    n, l, j = parse_atomic_term_symbol(symbol)

    print(f"symbol '{symbol}': n={n}, l={l}, j={j}")
    print(f"generated symbol: {QuantumNumberSet(n, l, j).term_symbol()}")

    print(parse_atomic_term_symbol(symbol) == symbol)
    print(QuantumNumberSet(n, l, j) == "1"+symbol)

    print(parse_atomic_term_symbol("1s"))

