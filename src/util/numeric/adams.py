import numpy as np
import scipy as sc

from util.math.linear_algebra import matmul_pointwise
from util.LUTs import AM_taylor_coeffs
try:
    from util.numeric.adams_f import adams as adams_f_subroutine
    _NO_FORTRAN = False
except ModuleNotFoundError:
    _NO_FORTRAN = True
    print("for improved speeds compile the Fortran-version of adams")
print(_NO_FORTRAN)


def adams_moulton_coefficients(order) -> np.ndarray:
    # load taylor coefficients from the taylor expansion of - sym_x/sp.log(1-sym_x) where sym_x(f[x]) = f[x] - f[x-1]
    if order > len(AM_taylor_coeffs):
        raise ValueError("order exceeded the stored taylor coefficients. Create a larger LUT to work with this order.")
    D = np.lcm.reduce(AM_taylor_coeffs[:order+1, 1])  # calculate D as the least common multiple of the taylor coefficients denominators
    AM_coeffs = np.zeros(order+1, dtype=np.int64)

    AM_coeffs[0] = D  # k=0
    for n in range(1, order + 1):
        # coefficients for f_{n-k} for k>0
        f_nk_coeffs = np.array([1] + [(-1)**k * np.math.factorial(n) // (np.math.factorial(n-k)*np.math.factorial(k))
                                      for k in range(1, n+1)], dtype=np.int64)
        AM_coeffs[:n+1] += AM_taylor_coeffs[n, 0]*(D // AM_taylor_coeffs[n, 1]) * f_nk_coeffs

    return np.append([D], AM_coeffs[::-1])


def adams_schrodinger(k: int, direction: str, y_start: np.array, b: np.array, c: np.array, h: float):
    if direction.lower() not in ["in", "out"]:
        raise ValueError(f"integration direction must be 'in' or 'out'")

    assert len(y_start.shape) == 2  # assure y_start_out is an 2-dim-array
    assert y_start.shape[0] == k  # with the length equals to te order k since the k + 1 order Adams-Moulton needs k start values
    assert y_start.shape[1] == 2  # and with two entries (R, Q) in the second dimension
    assert b.shape == c.shape  # assure b and c got the same length

    AM_coeffs = adams_moulton_coefficients(k)
    D = AM_coeffs[0]
    AM_coeffs = AM_coeffs[1:]

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

    dim = G.shape[1]  # get the dimension of y from G as each element in G is in \mathbb{R}^{dim \times dim}

    assert len(y_start.shape) == 2  # assure y_start_out is an 2-dim-array
    assert y_start.shape[0] == k  # with the length equals to the order k since the k + 1 order Adams-Moulton needs k start values
    assert y_start.shape[1] == dim
    assert G.shape[1:] == (dim, dim)

    AM_coeffs = adams_moulton_coefficients(k)
    D = AM_coeffs[0]
    AM_coeffs = AM_coeffs[1:]

    if direction.lower() == "in":
        G = G[::-1]

    # the naming of the variables follows Johnson (Lectures 2006) chapter 2.3.1
    y = np.empty((len(G), dim), dtype=np.float64)
    y[:k] = y_start if direction.lower() == "out" else y_start[::-1]

    f = np.empty_like(y)  # f = G*y
    f[:k] = matmul_pointwise(G[:k], y[:k])
    # a = np.array([AM_coeffs[:-1]]*dim).T  # coefficient matrix used to multiply elementwise with f, generate here for increased perfomance
    a = np.ones((k, dim), dtype=np.int64)
    for i in range(k):
        a[i] *= AM_coeffs[i]
    M = (np.ones_like(G) * np.eye(dim)) - h * (AM_coeffs[k] / D) * G
    M_inv = np.linalg.inv(M)
    for n in range(k, y.shape[0]):
        y[n] = M_inv[n] @ (y[n-1] + h/D * np.sum(a * f[n-k:n], axis=0))
        f[n] = G[n] @ y[n]
    if direction.lower() == "in":
        return y[::-1]
    else:
        return y

if not _NO_FORTRAN:
    def adams_f(k: int, direction: str, y_start: np.array, G: np.array, h: float):
        if direction.lower() not in ["in", "out"]:
            raise ValueError(f"integration direction must be 'in' or 'out'")

        dim = G.shape[1]  # get the dimension of y from G as each element in G is in \mathbb{R}^{dim \times dim}

        assert len(y_start.shape) == 2  # assure y_start_out is an 2-dim-array
        assert y_start.shape[0] == k  # with the length equals to the order k since the k + 1 order Adams-Moulton needs k start values
        assert y_start.shape[1] == dim
        assert G.shape[1:] == (dim, dim)

        AM_coeffs = np.asfortranarray(adams_moulton_coefficients(k), dtype=np.int64)

        if direction.lower() == "in":
            G = G[::-1]
        G = np.asfortranarray(G, dtype=np.float64)

        # the naming of the variables follows Johnson (Lectures 2006) chapter 2.3.1
        y = np.empty((len(G), dim), dtype=np.float64, order="F")
        y[:k] = y_start if direction.lower() == "out" else y_start[::-1]
        adams_f_subroutine(y, G, AM_coeffs, h, k, G.shape[0], dim)
        if direction.lower() == "in":
            return y[::-1]
        else:
            return y








if __name__ == "__main__":
    N = 10000
    h = 0.0001  # 0.02
    r0 = 1  # 0.0005
    t = (np.arange(N) + 1) * h
    r = r0*(np.exp(t)-1)

    k = 13

    b = np.append(np.diff(r)/np.diff(t), [np.nan])
    c = -2 * b * (1-1/r-1/(2*r**2))

    y_start = np.arange(k*2).reshape((k, 2))
    #
    # y1 = adams_schrodinger(k, "out", y_start, b, c, h)
    #
    G = np.zeros((len(b), 2, 2))
    G[:, 0, 1] = b
    G[:, 1, 0] = c

    import time

    AM_coeffs = adams_moulton_coefficients(k)

    t_start = time.perf_counter()
    y = adams(k, "out", y_start, G, h)
    t_end = time.perf_counter()
    print(f"ellapsed time (adams): {t_end-t_start}s")

    t_start = time.perf_counter()
    y2 = adams_f(k, "out", y_start, G, h)
    t_end = time.perf_counter()
    print(f"ellapsed time (adams_f): {t_end-t_start}s")
    print(np.allclose(y[~np.isnan(y)], y2[~np.isnan(y2)]))

    # print(y1)
    # print(y2)
    # print(np.allclose(y1[:-1], y2[:-1]))
    #
    # import timeit
    # # print(timeit.timeit(lambda: adams_schrodinger(k, "out", y_start_out, b, c, h), number=10_000))
    # # print(timeit.timeit(lambda: adams(k, "out", y_start, G, h), number=1_000))
    # for k_ in range(1, k+1):
    #     print(f"k = {k_}:", timeit.timeit(lambda: adams(k_, "out", y_start[:k_], G, h), number=1_000))

    # # test how big the performance increase using adams on larger matrices than multiple times is
    # import timeit
    # def multiple_adams(num):
    #     for _ in range(num):
    #         y_start = np.arange(k * 2).reshape((k, 2))
    #         G = np.zeros((len(b), 2, 2))
    #         G[:, 0, 1] = b
    #         G[:, 1, 0] = c
    #         adams(k, "out", y_start, G, h)
    # def bigger_adams(num):
    #     y_start = np.concatenate([np.arange(k * 2).reshape((k, 2)) for _ in range(num)], axis=1)
    #     G = np.zeros((len(b), 2*num, 2*num))
    #     for i in range(num):
    #         G[:, 2*i, 2*i+1] = b
    #         G[:, 2*i+1, 2*i] = c
    #     y = adams(k, "out", y_start, G, h)
    #     # print(y)
    #
    # # bigger_adams(2)
    #
    # print(timeit.timeit(lambda: multiple_adams(10), number=1_000))
    # print(timeit.timeit(lambda: bigger_adams(10), number=1_000))




