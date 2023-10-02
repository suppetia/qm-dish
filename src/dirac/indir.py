import numpy as np

from util.atomic_units import c


def generalized_prinzipal_quantum_number(n, k, gamma):
    return np.sqrt(n**2-2*(n-k)*(k-gamma))


def indir(order, r, E, kappa, effective_charge):

    if order < 1:
        raise ValueError("expansion order must be at least 1")

    lambda_ = np.sqrt(c**2 - E**2/c**2)
    sigma = E*effective_charge/(c**2*lambda_)

    def bk_list(k: int):
        if k == 1:
            return [1, 1/(2*c) * (kappa * effective_charge/lambda_)]
        bks = bk_list(k-1)
        bk = 1/(2+k*lambda_) * (kappa**2 - (k - sigma)**2 - effective_charge**2/c**2) * bks[-1]
        bks.append(bk)
        return bks

    def ak_list(k: int, bks):
        return [0] + [c/(k_*lambda_) * (kappa + (k_-sigma)*E/c**2 - effective_charge*lambda_/c**2) * bks[k_]
                      for k_ in range(1, k+1)]

    b = bk_list(order)
    a = ak_list(order, b)

    P = np.power(r, sigma) * np.exp(-lambda_ * r) * (np.sqrt((c**2+E)/(2*c**2)) * np.sum([a[k]/r**k for k in range(order+1)])
                                                     + np.sqrt((c**2-E)/(2*c**2)) * np.sum([b[k]/r**k for k in range(1, order+1)]))
    Q = np.power(r, sigma) * np.exp(-lambda_ * r) * (np.sqrt((c**2+E)/(2*c**2)) * np.sum([a[k]/r**k for k in range(order+1)])
                                                     - np.sqrt((c**2-E)/(2*c**2)) * np.sum([b[k]/r**k for k in range(1, order+1)]))

    return P, Q


if __name__ == "__main__":
    from dirac.coulomb.analytical import energy, radial_function
    from util.atomic_units import c, alpha

    print(c)

    M = np.inf
    Z = 2
    n = 3
    kappa = 1

    N = 21000
    h = 0.0005  # 0005 #0.00001
    r0 = 0.0005
    t = np.arange(N) * h
    r = r0 * (np.exp(t) - 1)
    r[np.isclose(r, 0, atol=1e-15)] = 1e-15

    V = - Z * 1 / (1 + 1 / M) / r

    E = energy(n, kappa, Z, M)
    print(E)
    gamma = np.sqrt(kappa**2-(alpha*Z)**2)
    N = np.sqrt(n**2-2*(n-abs(kappa))*(abs(kappa)-gamma))
    zeta = Z - N + 1

    order = 13
    P_start, Q_start = indir(order, r[-order:], E-c**2, kappa, effective_charge=zeta)

    P, Q = radial_function(n=n, kappa=kappa, r=r, Z=Z, M=np.inf)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2)
    ax[0].plot(r[-order:], P[-order:])
    ax[1].plot(r[-order:], Q[-order:])

    # ax[0].scatter(r[:order], P_start)
    # ax[1].scatter(r[:order], Q_start)

    ax[0].scatter(r[-order:], P_start * P[-order:] / (P_start[0] if not np.isclose(P_start[0], 0) else 1))
    ax[1].scatter(r[-order:], Q_start * Q[-order:] / (Q_start[0] if not np.isclose(Q_start[0], 0) else 1))

    plt.show()
