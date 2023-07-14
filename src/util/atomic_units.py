"""
factors to convert from atomic units to SI units.
source: 2018 CODATA recommended values
"""

import numpy as np

# ------------------ (auxiliary) natural constants ------------------------------

# speed of light
c = 299_792_458  # in m s^-1

# vacuum permittivity
epsilon_0 = 8.854_187_8128  # in F m^-1 = A s V^-1 m^-1)

# ------------------ coherent units ------------------------------

# reduced Planck constant / atomic unit of action
hbar = 1.054_571_817e-34  # in J s = kg m^2 s^-1

# elementary charge / atomic unit of charge
e = 1.602_176_634e-19  # in C = A s

# electron rest mass / atomic unit of mass
m_e = 9.109_383_7915e-31  # in kg

# Coulomb constant
k_e = 1/(4 * np.pi * epsilon_0)  # in kg m^3 s^-4 A^-2

# ------------------ derived units ------------------------------

# Sommerfeld's finestructure constant
alpha = e**2 * k_e / (hbar * c)

# Bohr radius / atomic unit of length
a_0 = hbar**2 / (m_e * e**2 + k_e)

# hartree / atomic unit of energy
E_h = m_e * e**4 * k_e**2 / hbar**2
