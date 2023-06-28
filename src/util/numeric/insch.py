import numpy as np


def insch(order, r, E, l, mu, effective_charge):

    lambda_ = np.sqrt(-2*E*mu)
    sigma = effective_charge*mu/lambda_

    def ak_list(k: int):
        if k == 0:
            return [1]
        aks = ak_list(k-1)
        ak = (l*(l+1)-(sigma-k)*(sigma-k+1))/(2*k*lambda_)*aks[-1]
        aks.append(ak)
        return aks

    def bk_list(k: int):
        if k == 0:
            return [-lambda_]
        bks = bk_list(k-1)
        bk = ((sigma+k)*(sigma-k+1)-l*(l+1))/(2*k)*bks[-1]
        bks.append(bk)
        return bks

    a = ak_list(order)
    b = bk_list(order)
    # b = [-lambda_] + [((sigma+k_)*(sigma-k_+1)-l*(l+1))/(2*k_)*a[k_-1] for k_ in range(1, order+1)]

    R = np.power(r, sigma)*np.exp(-lambda_*r) * np.sum(np.array([a[k]/r**k for k in range(order+1)]), axis=0)
    Q = np.power(r, sigma)*np.exp(-lambda_*r) * np.sum(np.array([b[k]/r**k for k in range(order+1)]), axis=0)

    return R, Q


if __name__ == "__main__":
    order = 10

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
    V = - Z * mu / r

    R, Q = insch(order, r=r[-order:], mu=mu, l=l, E=E, effective_charge=1)

    R_ = radial_function(n, l, r, Z, np.inf)

    print(R)
    print(R_[-order:])
