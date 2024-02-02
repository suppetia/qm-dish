import numpy as np

from dish.schrödinger.outsch import differentiation_coefficients_efficient
from dish.util.atomic_units import alpha
from dish.util.radial.grid import DistanceGrid


def outdir(order, Z, kappa, W, V, r_grid: DistanceGrid):
    k = order
    m = differentiation_coefficients_efficient(k)
    gamma = np.sqrt(kappa**2 - (alpha*Z)**2)

    r = r_grid.r[1:]
    r_prime = r_grid.rp[1:k+1]

    r_ = r[:k]

    a = -(gamma+kappa)*r_prime/r_
    b = -alpha*(W-V[:k]+2/alpha**2)*r_prime
    c = alpha*(W-V[:k])*r_prime
    d = -(gamma-kappa)*r_prime/r_

    u0 = 1
    v0 = -(kappa+gamma)/(alpha*Z) if kappa > 0 else alpha*Z/(gamma-kappa)

    # construct a system of linear equations using matrix notation A*x=B where x = (u,v)
    u0 = np.array(u0)
    v0 = -np.array(v0)
    B = np.empty(2 * k)
    B[:k] = -m[1:, 0] * u0
    B[k:] = -m[1:, 0] * v0

    A = np.empty((2 * k, 2 * k))
    A[:k, :k] = m[1:, 1:] - np.identity(k) * a
    A[:k, k:] = -np.identity(k) * b
    A[k:, :k] = -np.identity(k) * c
    A[k:, k:] = m[1:, 1:] - np.identity(k) * d

    result = np.linalg.solve(A,B)
    scaled_result = result * np.ravel([r_**gamma]*2)

    P = scaled_result[:k]
    Q = scaled_result[k:]

    return P, Q



if __name__ == "__main__":
    from dirac.coulomb.analytical import energy, radial_function
    from util.atomic_units import c
    print(c)

    M = np.inf
    Z = 2
    n = 3
    kappa = 1

    N = 570
    h = 0.0005  # 0005 #0.00001
    r0 = 0.0005
    t = np.arange(N) * h
    r = r0 * (np.exp(t) - 1)
    r[np.isclose(r, 0, atol=1e-15)] = 1e-15

    V = - Z * 1/(1+1/M)/r

    W = energy(n, kappa, Z, M) - c**2
    print(W)

    order = 13
    P_start, Q_start = outdir(order, Z, kappa, W, V, r, t)

    P, Q = radial_function(n=n, kappa=kappa, r=r, Z=Z, M=np.inf)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2)
    ax[0].plot(r[:order], P[:order])
    ax[1].plot(r[:order], Q[:order])

    # ax[0].scatter(r[:order], P_start)
    # ax[1].scatter(r[:order], Q_start)

    ax[0].scatter(r[:order], P_start[:order]*P[order-1]/P_start[-1])
    ax[1].scatter(r[:order], Q_start[:order] * Q[order - 1] / Q_start[-1])

    plt.show()

