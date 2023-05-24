import numpy as np
import sympy as sp

from util.math.taylor import taylor_coefficients, sym_x
from util.math.linear_algebra import matmul_pointwise
from util.LUTs import AM_taylor_coeffs


def adams_moulton_coefficients(order):
    # taylor series coefficients of -d/log(1-d) as tuples containing (numerator, denominator)
    # taylor_coeffs = np.array([(1,1), (-1, 2), (-1, 12), (-1, 24), (-19, 720), (-3, 160),
    #                           (-863, 60_480), (-275, 24_192), (-33_953, 3_628_500), (-8_183, 1_036_800)],
    #                          dtype=np.int64)
    # if order > len(taylor_coeffs) - 1:
    #     raise ValueError(f"Adams-Moulton not implemented for orders larger than {len(taylor_coeffs) - 1}")
    
    D = np.lcm.reduce(AM_taylor_coeffs[:order+1, 1])
    AM_coeffs = np.zeros(order+1, dtype=np.int64)

    AM_coeffs[0] = D  # k=0
    for n in range(1, order + 1):
        # coefficients for f_{n-k} for k>0
        f_nk_coeffs = np.array([1] + [(-1)**k * np.math.factorial(n) // (np.math.factorial(n-k)*np.math.factorial(k))
                                      for k in range(1, n+1)], dtype=np.int64)
        AM_coeffs[:n+1] += AM_taylor_coeffs[n, 0]*D // AM_taylor_coeffs[n, 1] * f_nk_coeffs

    return AM_coeffs[::-1], D


def adams_schrodinger(k: int, direction: str, y_start: np.array, b: np.array, c: np.array, h: float):
    if direction.lower() not in ["in", "out"]:
        raise ValueError(f"integration direction must be 'in' or 'out'")

    assert len(y_start.shape) == 2  # assure y_start is an 2-dim-array
    assert y_start.shape[0] == k  # with the length equals to te order k since the k + 1 order Adams-Moulton needs k start values
    assert y_start.shape[1] == 2  # and with two entries (R, Q) in the second dimension
    assert b.shape == c.shape  # assure b and c got the same length

    AM_coeffs, D = adams_moulton_coefficients(k)

    # the naming of the variables follows Johnson (Lectures 2006) chapter 2.3.1

    lambda_ = h * AM_coeffs[k]/D

    y = np.empty((len(b), 2), np.float64)
    y[:k] = y_start

    for n in range(k, y.shape[0]):
        M_inv = np.array([[1, lambda_*b[n]],
                          [lambda_*c[n], 1]
                          ]) / (1-lambda_*lambda_*b[n]*c[n])
        # print(np.array([c[n-k:n], b[n-k:n]]).T * y[n-k:n])
        G = np.zeros((k, 2, 2))
        G[:, 0, 1] = b[n-k:n]
        G[:, 1, 0] = c[n-k:n]
        y[n] = np.matmul(M_inv,
                         (y[n-1] + h/D * np.sum(np.array([AM_coeffs[:-1], AM_coeffs[:-1]]).T
                                                * matmul_pointwise(G, y[n-k:n]),
                                                axis=0
                                                )
                          )
                         )
    if direction.lower() == "in":
        return y[::-1]
    else:
        return y

def adams(k: int, direction: str, y_start: np.array, G: np.array, h: float):
    if direction.lower() not in ["in", "out"]:
        raise ValueError(f"integration direction must be 'in' or 'out'")

    assert len(y_start.shape) == 2  # assure y_start is an 2-dim-array
    assert y_start.shape[0] == k  # with the length equals to the order k since the k + 1 order Adams-Moulton needs k start values
    assert y_start.shape[1] == 2  # and with two entries (R, Q) in the second dimension
    assert G.shape[1:] == (2,2)

    AM_coeffs, D = adams_moulton_coefficients(k)

    # the naming of the variables follows Johnson (Lectures 2006) chapter 2.3.1
    y = np.empty((len(G), 2), dtype=np.float64)
    y[:k] = y_start

    for n in range(k, y.shape[0]):
        M_inv = np.linalg.inv(np.identity(2) - h * AM_coeffs[k]/D * G[n])

        f = matmul_pointwise(G[n-k:n], y[n-k:n])
        y[n] = np.matmul(M_inv,
                         (y[n-1] + h/D * np.sum(np.array([AM_coeffs[:-1], AM_coeffs[:-1]]).T * f,
                                                axis=0
                                                )
                          )
                         )
    if direction.lower() == "in":
        return y[::-1]
    else:
        return y








if __name__ == "__main__":
    # for k in range(1,9):
    #     print(adams_moulton_coefficients(k))

    N = 100
    h = 1  # 0.02
    r0 = 1  # 0.0005
    t = np.arange(N)/h + 1
    r = r0*(np.exp(t)-1)

    k = 10

    b = np.append(np.diff(r)/np.diff(t), [np.nan])
    c = -2 * b * (1-1/r-1/(2*r**2))

    y_start = np.arange(k*2).reshape((k, 2))

    y1 = adams_schrodinger(k, "out", y_start, b, c, h)

    G = np.zeros((len(b), 2, 2))
    G[:, 0, 1] = b
    G[:, 1, 0] = c
    y2 = adams(k, "out", y_start, G, h)

    print(y1)
    print(y2)
    print(np.allclose(y1[:-1], y2[:-1]))

    # print(timeit.timeit(lambda: adams_schrodinger(k, "out", y_start, b, c, h), number=10_000))
    #
    # G = np.zeros((len(b), 2, 2))
    # G[:, 0, 1] = b
    # G[:, 1, 0] = c
    # print(timeit.timeit(lambda: adams(k, "out", y_start, G, h), number=10_000))




