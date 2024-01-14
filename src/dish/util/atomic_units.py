"""
factors to convert from atomic units to SI units.
source: 2018 CODATA recommended values
"""

import re
from typing import Union

import numpy as np
from mpmath import mp

# ------------------ (auxiliary) natural constants ------------------------------

# speed of light
_c_0 = 299_792_458  # in m s^-1

# vacuum permittivity
_epsilon_0 = 8.854_187_812_8e-12  # in F m^-1 = A s V^-1 m^-1

# ------------------ coherent units ------------------------------

# reduced Planck constant / atomic unit of action
hbar = 1.054_571_817e-34  # in J s = kg m^2 s^-1

# elementary charge / atomic unit of charge
e = 1.602_176_634e-19  # in C = A s

# electron rest mass / atomic unit of mass
m_e = 9.109_383_7915e-31  # in kg

# Coulomb constant
k_e = 1/(4 * np.pi * _epsilon_0)  # in kg m^3 s^-4 A^-2

# ------------------ derived units ------------------------------

# Sommerfeld's finestructure constant
alpha = e**2 * k_e / (hbar * _c_0)

# speed of light
c = 1/alpha

# Bohr radius / atomic unit of length
a_0 = hbar**2 / (m_e * e**2 * k_e)

# hartree / atomic unit of energy
E_h = m_e * e**4 * k_e**2 / hbar**2


def convert_u2m_e(mass):  # convert mass from u to m_e
    u = 1.660_539_066_60e-27
    return mass*u/m_e


def convert_units(old_unit: Union[str, float],
                  new_unit: Union[str, float],
                  value=1.,
                  old_unit_exp=1,
                  new_unit_exp=1):
    """
    Convert 'value' in unit 'old_unit' to 'new_unit'.
    ! YOU NEED TO ASSURE THAT THESE VALUES ARE OF THE SAME DIMENSION !
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
    units = {"E_h": E_h, "a_0": a_0, "hbar": hbar, "e": e, "m_e": m_e,
             # other units
             "eV": e,  # in J
             "u": 1.660_539_066_60e-27,  # in kg
             # SI units
             "m": 1, "kg": 1, "J": 1,
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

    new_value = mp.mpf(value)*old_unit/new_unit
    mp.dps = old_precision
    return value_type(new_value)


if __name__ == "__main__":
    val = 1.00782503223
    print(convert_u2m_e(val))
    print(convert_units("u", "m_e", val))
