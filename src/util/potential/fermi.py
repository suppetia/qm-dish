import numpy as np
from scipy.interpolate import PchipInterpolator


def S(k, mu, order=50):
    return np.sum([pow(-1, (n-1) % 2)/pow(n, k)*np.exp(-n*mu) for n in range(1, order+1)], axis=0)

def P(k, r, c, a, order=50):
    return np.sum([pow(-1, (n-1) % 2) / pow(n, k) * np.exp(-n*np.abs(r-c)/a) for n in range(1, order + 1)],
                  axis=0)


def fermi_charge_distribution(Z, c, a, r):
    mu = c/a
    if mu > 1:
        N = 1 + (np.pi/mu)**2 + 6/mu**3*S(3, mu)
    else:
        N = 1 + (a*np.pi/c)**2 + 6*(a/c)**3*S(3, mu)

    rho0 = 3/(4*np.pi*pow(c,3))*Z/N
    rho = rho0/(1+np.exp(r/a-mu))
    return np.nan_to_num(rho, nan=0.)  # handle overflow for large distances


def potential(Z, c, a, r):
    # # evaluate the potential on an evenly spaced grid and inter-/extrapolate afterwards on the given points
    # r_ = np.linspace(start=r[0] if r[0] > 0 else 1e-10,
    #                  stop=r[-1],
    #                  num=5000)
    r_ = r

    mu = c/a
    if mu > 1:
        N = 1 + (np.pi/mu)**2 + 6/mu**3*S(3, mu)
    else:
        N = 1 + (a*np.pi/c)**2 + 6*(a/c)**3*S(3, mu)

    P2 = P(2, r_, c, a)
    P3 = P(3, r_, c, a)
    S3 = S(3, mu)

    V = np.select(
        condlist=[r_ < c, r_ >= c],
        choicelist=[
            Z / (c*N) * (3 / 2 - np.power(r_, 2) / (2 * c ** 2) + (np.pi / mu) ** 2 / 2 + 3 / mu ** 2 * P2 + 6 * a ** 3 / (c ** 2 * r_) * (S3 - P3)),
            Z / (N * r_) * (1 + (np.pi / mu) ** 2 - 3 * (a ** 2 * r_ / c ** 3) * P2 + 6 / mu ** 3 * (S3 - P3))
        ]
    )

    # return PchipInterpolator(r_, V, extrapolate=True)(r)
    return np.interp(r, r_, V)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from util.misc import parse_atomic_term_symbol, find_suitable_number_of_integration_points_dirac
    from util.potential import poisson

    Z = 2
    c=1e0
    a=1/4

    n, l, j = parse_atomic_term_symbol("2p3/2")
    M = np.inf

    mu = 1 / (1 + 1 / M)
    kappa = -l - 1 if np.isclose(j, l + 1 / 2) else l

    # N = 570
    h = 0.0005
    r0 = 0.0005
    N = find_suitable_number_of_integration_points_dirac(Z, M, n, kappa, r0, h)
    print(f"number of integration points: {N}")
    t = np.arange(N) * h
    r = r0 * (np.exp(t) - 1)

    # r = np.linspace(0,10, num=500)

    V_fermi = potential(Z, c, a, r)
    # N = np.trapz(r**2*V_fermi, x=r)
    # V_fermi = V_fermi/N*Z
    V_poisson = poisson.solve_radial_poisson_equation(Z, r, rho=lambda r_: fermi_charge_distribution(Z, c, a, r_),
                                                      num_it=100000)
    print(V_fermi[0]/V_poisson[0])
    # V_poisson *= V_fermi[0]/V_poisson[0]

    print(np.sum((V_poisson-V_fermi)**2))


    plt.figure()
    plt.plot(r, V_fermi, "-", label="using direct method")
    plt.plot(r, V_poisson, "-", label="using general method")
    plt.plot(r[r>0.3], Z/r[r>.3], "-", label="Coulomb")
    plt.legend()
    # plt.xlim(0,4e-0)
    plt.show()