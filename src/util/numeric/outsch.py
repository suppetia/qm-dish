import numpy as np

from util.math.taylor import sym_x, taylor_coefficients
from util.LUTs.outsch import taylor_coeffs


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
                m[j_, k-l] += a[j, i] * (-1)**l * np.math.factorial(i)/(np.math.factorial(l)*np.math.factorial(i-l))

    return m

def differentiation_coefficients_efficient(order):
    # load taylor coefficients from the taylor expansion of - sym_x/sp.log(1-sym_x) where sym_x(f[x]) = f[x] - f[x-1]
    if order > len(taylor_coeffs):
        raise ValueError("order exceeded the stored taylor coefficients. Create a larger LUT to work with this order.")

    k = order  # alias to agree with naming in Johnson lecture
    m = np.zeros((k+1, k+1), dtype=np.float64)
    a = taylor_coeffs

    for j in range(np.math.ceil((k+1)/2)):
        # calculate the coeffs m[., .] of y in
        # dy/dt[order-j] = sum_{i=0}^k a[j,i] del^i y[order] = sum_{i=0}^k m[order-j, i] y[i]

        j_ = k - j  # shortcut

        for i in range(k+1):
            for l in range(i+1):
                m[j_, k - l] += a[j, i] * (-1)**l * np.math.factorial(i)/(np.math.factorial(l)*np.math.factorial(i-l))

        if j == j_:
            continue
        m[j] = - m[j_, ::-1]  # use symmetry

    return m


def outsch(order, p0, q0, l, E, V, r, t):
    k = order
    m = differentiation_coefficients(k)

    b = (np.diff(r)/np.diff(t))[:k]
    c = -2 * b * (E-V[:k])
    d = -2 * b * (l+1)/r[:k]

    # construct a system of linear ODEs using matrix notation A*x=B where x = (p,q)
    p0 = np.array(p0)
    q0 = np.array(q0)
    B = np.empty(2*k)
    B[:k] = -m[1:, 0]*p0
    B[k:] = -m[1:, 0]*q0

    A = np.empty((2*k, 2*k))
    A[:k, :k] = m[1:, 1:]
    A[:k, k:] = - np.identity(k)*b  # TODO: check if this needs to be b or b[1:]
    # print(A)
    A[k:, k:] = m[1:, 1:] - np.identity(k)*d
    A[k:, :k] = - np.identity(k)*c
    # print(A)

    result = np.linalg.solve(A, B)
    p = result[:k]
    q = result[k:]

    R = r[:k]**(l+1)*p
    Q = r[:k]**(l+1)*(q+(l+1)/r[:k]*p)

    return R, Q



if __name__ == "__main__":
    from timeit import timeit

    order = 10
    number = 10000

    # m = differentiation_coefficients_efficient(order)
    # m2 = differentiation_coefficients(order)
    # print(np.allclose(m, m2))
    # print(m)
    # print(m2)

    # t_original = timeit(lambda: differentiation_coefficients(order), number=number)
    # print(t_original)
    # t_efficient = timeit(lambda: differentiation_coefficients_efficient(order), number=number)
    # print(t_efficient)

    from src.schrodinger.coulomb.analytical import energy, radial_function

    N = 600
    h = 0.02
    r0 = 0.0005  # 0.0005
    t = (np.arange(N) + 1) * h
    r = r0 * (np.exp(t) - 1)

    Z = 1
    mu = 1
    l = 0
    n = 1
    E = energy(n, Z)
    V = - Z*mu/r

    R, Q = outsch(order, p0=1, q0=-Z*mu/(l+1), l=l, E=E, V=V, r=r, t=t)

    R_ = radial_function(n, l, r, Z, np.inf)

    print(R)
    print(R_[:order])


