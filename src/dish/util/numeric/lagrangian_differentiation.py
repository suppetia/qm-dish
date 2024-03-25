import math
import numpy as np

from dish.util.LUTs.lagrangian_differentiation import taylor_coeffs


coefficients_storage = {}  # if differentiation coefficients have been computed once, store them in a dict


def differentiation_coefficients(order):
    # load taylor coefficients from the taylor expansion of - sym_x/sp.log(1-sym_x) where sym_x(f[x]) = f[x] - f[x-1]
    if order > len(taylor_coeffs):
        raise ValueError("order exceeded the stored taylor coefficients. Create a larger LUT to work with this order.")

    k = order # alias to agree with naming in Johnson lecture
    m = np.zeros((k+1, k+1), dtype=np.float64)
    a = taylor_coeffs

    for j in range(k+1):
        # calculate the coeffs m[., .] of y in
        # dy/dt[order-j] = sum_{i=0}^k a[j,i] del^i y[order] = sum_{i=0}^k m[order-j, i] y[i]

        j_ = k - j  # shortcut

        for i in range(k+1):
            for l in range(i+1):
                m[j_, k-l] += a[j, i] * (-1)**l * math.factorial(i)/(math.factorial(l)*math.factorial(i-l))

    return m


def differentiation_coefficients_efficient(order):
    if order in coefficients_storage:  # if the coefficients have been computed once, load the values
        return coefficients_storage[order]

    # load taylor coefficients from the taylor expansion of - sym_x/sp.log(1-sym_x) where sym_x(f[x]) = f[x] - f[x-1]
    if order > len(taylor_coeffs):
        raise ValueError("order exceeded the stored taylor coefficients. Create a larger LUT to work with this order.")

    k = order  # alias to agree with naming in Johnson lecture
    m = np.zeros((k+1, k+1), dtype=np.float64)
    a = taylor_coeffs

    for j in range(math.ceil((k+1)/2)):
        # calculate the coeffs m[., .] of y in
        # dy/dt[order-j] = sum_{i=0}^k a[j,i] del^i y[order] = sum_{i=0}^k m[order-j, i] y[i]

        j_ = k - j  # shortcut

        for i in range(k+1):
            for l in range(i+1):
                m[j_, k - l] += a[j, i] * (-1)**l * math.factorial(i)/(math.factorial(l)*math.factorial(i-l))

        if j == j_:
            continue
        m[j] = - m[j_, ::-1]  # use symmetry

    coefficients_storage[order] = m
    return m
