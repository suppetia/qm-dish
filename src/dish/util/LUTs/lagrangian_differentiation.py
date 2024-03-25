import sympy as sp
import numpy as np
import os

from dish.util.math_util.taylor import taylor_coefficients, sym_x

LUT_filename = os.path.dirname(os.path.abspath(__file__)) + os.sep + "lagrangian_differentiation.LUT"


def compute_LUT(order=10):
    # lagrangian differentiation formula coefficients up to the order^th order
    sym_j = sp.symbols("sym_j")
    _operator = - sp.log(1-sym_x) * (1-sym_x) ** sym_j  # where x is the finite differential operator
    _taylor_expansion = taylor_coefficients(_operator, order=order, x0=0)

    taylor_coeffs_2D = np.empty((order, order+1), dtype=np.float64)
    for j in range(order):
        taylor_coeffs_2D[j, :] = [t.subs({sym_j: j}) for t in _taylor_expansion]

    np.savetxt(LUT_filename, taylor_coeffs_2D, fmt="%.15e", delimiter="\t")


if not os.path.exists(LUT_filename):
    compute_LUT()

taylor_coeffs = np.loadtxt(LUT_filename, dtype=np.float64, delimiter="\t")



if __name__ == "__main__":
    compute_LUT(order=15)


