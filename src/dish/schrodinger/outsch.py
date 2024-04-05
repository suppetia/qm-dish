import numpy as np

from dish.util.numeric.lagrangian_differentiation import differentiation_coefficients_efficient


def outsch(order, p0, q0, l, E, V, r_grid):
    k = order
    m = differentiation_coefficients_efficient(k)

    # b = (np.diff(r)/np.diff(t))[:k]
    # b = r[-1] / (np.exp(t[-1]) - 1) * np.exp(t[:k])
    r = r_grid.r[1:k+1]
    b = r_grid.rp[1:k+1]
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
    A[:k, k:] = - np.identity(k)*b
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

    from dish.schrodinger.coulomb.analytical import energy, radial_function

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

    print(R*R_[0]/R[0])
    print(R_[:order])


