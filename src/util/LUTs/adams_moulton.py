import sympy as sp
import numpy as np
import os

from util.math.taylor import taylor_coefficients, sym_x

LUT_filename = os.path.dirname(os.path.abspath(__file__)) + os.sep + "adams_moulton.LUT"


def compute_LUT(order=10):
    # Adams moulton coefficients up to the 10th order
    _AM_operator = - sym_x/sp.log(1-sym_x)  # where x is the finite differential operator
    _taylor_expansion = taylor_coefficients(_AM_operator, order=order, x0=0)
    AM_taylor_coeffs = np.array([list(sp.fraction(t)) for t in _taylor_expansion], dtype=np.int64)

    np.savetxt(LUT_filename, AM_taylor_coeffs, fmt="%d", delimiter="\t")


if not os.path.exists(LUT_filename):
    compute_LUT()

taylor_coeffs = np.loadtxt(LUT_filename, dtype=np.int64, delimiter="\t")



if __name__ == "__main__":
    compute_LUT(order=15)


