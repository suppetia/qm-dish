"""
factors to convert from atomic units to SI units.
source: 2018 CODATA recommended values
"""

import re
from typing import Union

import numpy as np
from mpmath import mp


class _MultiprecisionUnits:  # multiprecision units
    # ------------------ (auxiliary) natural constants ------------------------------
    # speed of light
    c_0 = mp.mpf("299792458")  # in m s^-1
    # vacuum permittivity
    epsilon_0 = mp.mpf("8.8541878128e-12")  # in F m^-1 = A s V^-1 m^-1
    # atomic mass unit
    u = mp.mpf("1.66053906660e-27")  # in kg

    # ------------------ coherent units ------------------------------
    # reduced Planck constant / atomic unit of action
    hbar = mp.mpf("1.054571817e-34")  # in J s = kg m^2 s^-1
    # elementary charge / atomic unit of charge
    e = mp.mpf("1.602176634e-19")  # in C = A s
    # electron rest mass / atomic unit of mass
    m_e = mp.mpf("9.1093837915e-31")  # in kg

    @property
    def k_e(self):
        # Coulomb constant
        return 1/(4 * mp.pi * self.epsilon_0)  # in kg m^3 s^-4 A^-2

    # ------------------ derived units ------------------------------

    @property
    def alpha(self):
        # Sommerfeld's finestructure constant
        return self.e**2 * self.k_e / (self.hbar * self.c_0)

    @property
    def c(self):
        # speed of light
        return 1/self.alpha

    @property
    def a_0(self):
        # Bohr radius / atomic unit of length
        return self.hbar**2 / (self.m_e * self.e**2 * self.k_e)

    @property
    def E_h(self):
        # hartree / atomic unit of energy
        return self.m_e * self.e**4 * self.k_e**2 / self.hbar**2


mpu = _MultiprecisionUnits()

# define float representations of mp units

# ------------------ (auxiliary) natural constants ------------------------------

# speed of light
c_0 = float(mpu.c_0)  # in m s^-1

# vacuum permittivity
epsilon_0 = float(mpu.epsilon_0)  # in F m^-1 = A s V^-1 m^-1

# ------------------ coherent units ------------------------------

# reduced Planck constant / atomic unit of action
hbar = float(mpu.hbar)  # in J s = kg m^2 s^-1

# elementary charge / atomic unit of charge
e = float(mpu.e)  # in C = A s

# electron rest mass / atomic unit of mass
m_e = float(mpu.m_e)  # in kg

# Coulomb constant
k_e = float(mpu.k_e)  # in kg m^3 s^-4 A^-2

# ------------------ derived units ------------------------------

# Sommerfeld's finestructure constant
alpha = float(mpu.alpha)

# speed of light
c = float(mpu.c)

# Bohr radius / atomic unit of length
a_0 = float(mpu.a_0)

# hartree / atomic unit of energy
E_h = float(mpu.E_h)


def convert_units(old_unit: Union[str, float],
                  new_unit: Union[str, float],
                  value=1.,
                  old_unit_exp=1,
                  new_unit_exp=1):
    """
    | Convert 'value' in unit 'old_unit' to 'new_unit'.
    | ! You need to assure that these values are of the same dimension !

    :param old_unit: either the old units name or the conversion value into SI units
    :param new_unit: either the new units name or the conversion value into SI units
    :param value: value to convert. The default is 1 to just obtain the conversion factor.
    :param old_unit_exp: exponent of the old unit. The default is 1.
    :param new_unit_exp: exponent of the new unit. The default is 1.

    :return: converted value
    """
    old_precision = mp.dps
    value_type = type(value)
    # set precision to 50 places
    mp.dps = max(50, old_precision)
    units = {"E_h": mpu.E_h, "a_0": mpu.a_0, "hbar": mpu.hbar, "e": mpu.e, "m_e": mpu.m_e,
             # other units
             "eV": mpu.e,  # in J
             "u": mpu.u,
             # SI units
             "m": 1, "kg": 1, "J": 1, "C": 1,
             }
    exponent_pattern = re.compile(r"(?P<u>\w+)(\^\(?(?P<e>-?\d+(\.\d+)?(/\d+)?)\)?)?", re.IGNORECASE)
    if isinstance(old_unit, str):
        match = re.match(exponent_pattern, old_unit)
        old_unit = match.group("u")
        if match.group("e") is not None:
            old_unit_exp *= eval(match.group("e"))
    if isinstance(new_unit, str):
        match = re.match(exponent_pattern, new_unit)
        new_unit = match.group("u")
        if match.group("e") is not None:
            new_unit_exp *= eval(match.group("e"))
    if old_unit in units:
        old_unit = units[old_unit]
    if new_unit in units:
        new_unit = units[new_unit]

    old_unit = mp.power(mp.mpf(old_unit), old_unit_exp)
    new_unit = mp.power(mp.mpf(new_unit), new_unit_exp)

    new_value = mp.power(mp.mpf(value), old_unit_exp)*old_unit/new_unit
    mp.dps = old_precision
    return value_type(new_value)


if __name__ == "__main__":
    val = 1.00782503223
    print(convert_units("u", "m_e", val))
