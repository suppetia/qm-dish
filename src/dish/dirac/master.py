import numpy as np
from scipy.integrate import simpson

from typing import Union

from dish.dirac.coulomb.analytical import energy, radial_function

from dish.dirac.outdir import outdir
from dish.dirac.indir import indir
from dish.util.numeric.adams import adams

from dish.util.atomic_units import c, alpha
from dish.util.math_util import count_nodes
from dish.util.wave_function import RadialDiracWaveFunction
from dish.util.misc import DistanceGrid


def outer_classical_turning_point(V, W) -> int:
    return int(np.argmin(np.abs(W-V)))


def master(n, l, j, Z, M, V,
           r: Union[DistanceGrid, np.array],
           t: np.array = None,
           h: float = None,
           order_adams: int = 7,
           order_indir: int = 7,
           E_guess: Union[float, str] = "auto",
           max_number_of_iterations: int = 50):

    mu = 1/(1+1/M)

    kappa = -l - 1 if np.isclose(j, l+1/2) else l
    k = abs(kappa)

    gamma = np.sqrt(kappa**2-(alpha*Z*mu)**2)

    if E_guess == "auto":
        E_guess = energy(n, kappa, Z, M)
        # E_guess = energy_schrodinger(n, Z, M) + c**2

    if isinstance(r, DistanceGrid):
        r_grid = r
    else:
        # TODO: refactor properly
        if t is None or h is None:
            raise ValueError("values for 'r', 't' and 'h' must be provided if r isn't a DistanceGrid")
        # create a new Grid as a workaround
        r_grid = DistanceGrid(h, r[-1]/(np.exp(t[-1])-1), len(r))

    r = r_grid.r
    r_prime = r_grid.rp
    h = r_grid.h

    a_mat = -r_prime * (kappa / r)
    b_mat = lambda W: -alpha * r_prime * (W - V + 2 * c**2)
    c_mat = lambda W: alpha * r_prime * (W - V)
    d_mat = -a_mat

    G = np.zeros((len(r), 2, 2))
    G[:, 0, 0] = a_mat
    G[:, 1, 1] = d_mat

    # number of radial nodes for P_nkappa
    n_r = n - l - 1
    # generalized principal quantum number
    N = np.sqrt(n**2-2*(n-k)*(k-gamma))
    # effective charge
    zeta = Z - N + 1

    # bounds of eigenenergy
    W_u = np.inf  # highest energy with n_r nodes
    W_l = -np.inf  # lowest energy with n_r nodes

    it = 0
    W_guess = E_guess - c ** 2

    while it < max_number_of_iterations:
        it += 1

        # print(W_guess)
        E_guess = W_guess + c ** 2

        a_c = outer_classical_turning_point(V, W_guess)
        # print(a_c)

        # y_start_out = radial_function(n, kappa, r[:order_adams], Z, M).T

        y_start_out = np.array(
            outdir(order=order_adams, Z=Z, kappa=kappa, W=W_guess, V=-Z * mu / r,
                   r_grid=r_grid)
        ).T
        y_start_out[:, :] *= N  # the Q part is overestimated, to gain better assumption lower the initial values

        y_start_in = np.array(
            indir(order=order_indir, r=r[-order_adams:], E=E_guess, kappa=kappa,
                  effective_charge=zeta)
        ).T
        # if all start values are zero, also the following integration using adams scheme will yield zero values
        # to handle this artificially create a small offset from zero
        if np.allclose(y_start_in, 0, atol=1e-18):
            y_start_in[0, 0] = 1e-15
            y_start_in[0, 1] = 1e-18

        # for inward integration the sign of r_prime and therefore G changes!
        G[:, 0, 1] = b_mat(W_guess)
        G[:, 1, 0] = c_mat(W_guess)

        y_out = adams(order_adams, "out", y_start_out, G[:a_c + 1], h)
        y_in = adams(order_adams, "in", y_start_in, -G[a_c:], h)

        # make P=y[:,0] continuous
        y_in *= y_out[-1, 0] / y_in[0, 0]

        y = np.append(y_out, y_in[1:], axis=0)

        num_nodes = count_nodes(y[:, 0])
        # print("number of nodes:", num_nodes)

        if num_nodes < n_r:
            W_l = max(W_l, W_guess)  # keep track of the lower bound of energy (greatest energy with too few nodes)
            W_guess_new = W_guess * 0.9
        elif num_nodes > n_r:
            W_u = min(W_u, W_guess)  # keep track of the upper bound of energy (lowest energy with too many nodes)
            W_guess_new = W_guess * 1.1
        else:
            # check if Q is continuous
            # if this is fulfilled, we found the eigenfunction and therefore the eigenenergy
            if np.isclose(y_out[-1, 1] - y_in[0, 1], 0, atol=1e-15):
                break

            W_guess_new = W_guess + c*(y_in[0, 1]-y_out[-1, 1])*y_out[-1, 0] / np.trapz((y[:, 0]**2 + y[:, 1]**2)*r_prime, dx=h)
            # W_guess_new = W_guess + c*(y_in[0, 1]-y_out[-1, 1])*y_out[-1, 0] / simpson((y[:, 0] ** 2 + y[:, 1] ** 2) * r_prime,dx=h)

            # if the new guess does not differ from the old guess the eigenfunction is as good as it gets
            if np.isclose(W_guess-W_guess_new, 0, atol=1e-15):
                break
        if W_guess_new < W_l:
            W_guess_new = (W_guess+W_l)/2
        elif W_guess_new > W_u:
            W_guess_new = (W_guess+W_u)/2

        energy_convergence = abs(W_guess - W_guess_new)
        W_guess = W_guess_new

        N = 1 / np.sqrt(np.trapz((y[:, 0] ** 2 + y[:, 1] ** 2) * r_prime, dx=h))
    else:
        it = -1  # set it to -1 as a flag that the algorithm didn't converge

    # the initial guesses from asymptotics may be not accurate and therefore lead to a bump in the Q component,
    # use the analytical solution for the first points as an estimator to smoothen this component
    # y_start = radial_function(n, kappa, r[:5 * order_adams], Z, M).T
    # y_start *= y[5 * order_adams - 1] / y_start[-1]
    # y[:len(y_start)] = y_start

    N = 1/np.sqrt(np.trapz((y[:, 0]**2+y[:, 1]**2) * r_prime, dx=h))
    # N = 1/np.sqrt(simpson((y[:, 0] ** 2 + y[:, 1] ** 2) * r_prime,dx=h))

    y *= N

    return RadialDiracWaveFunction(r, y), W_guess, energy_convergence, a_c, it


if __name__ == "__main__":
    from util.misc import find_suitable_number_of_integration_points_dirac, parse_atomic_term_symbol
    from util.atomic_units import convert_units

    Z = 20
    # n, l, j = parse_atomic_term_symbol("25k17/2")
    n, l, j = parse_atomic_term_symbol("2p1/2")
    # n, l, j = parse_atomic_term_symbol("1s1/2")

    M = convert_units("u", "m_e", 42) #np.inf

    mu = 1 / (1 + 1 / M)
    kappa = -l - 1 if np.isclose(j, l+1/2) else l

    # N = 570
    h = 0.005
    r0 = 2e-6#0.0005
    h = 0.005
    r0 = 1e-6#0.0005
    N = find_suitable_number_of_integration_points_dirac(Z, M, n, kappa, r0, h)
    print(f"number of integration points: {N}")
    t = (np.arange(N)+1) * h
    r = r0 * (np.exp(t) - 1)
    # print(r[-1])
    # r[np.isclose(r, 0, atol=1e-15)] = 1e-15

    from util.potential import fermi
    from util.atomic_units import a_0

    core = convert_units("m", "a_0", 3.4759582751250715e-15)
    a = 2.3e-15/a_0 / (4*np.log(3))

    V = - fermi.potential(Z, core, a, r)

    V = - Z * mu/r

    import time

    t_start = time.time()
    Psi, E, dE, a_c, num_iteration = master(n, l, j, Z, M, V, r, t, h,
                                      order_adams=11,
                                      order_indir=11,
                                      E_guess="auto",
                                      max_number_of_iterations=20
                                      )
    print(f"finding eigenenergy and -function took {time.time() - t_start:.3f}s and {num_iteration} iterations")
    P,Q = Psi.Psi.T

    import matplotlib.pyplot as plt
    from dirac.coulomb.analytical import radial_function, energy
    P_func, Q_func = radial_function(n, kappa, r, Z, M)
    E_analytical = energy(n, kappa, Z, M) - c**2
    print(f"analytical value of eigenenergy: {E_analytical}")
    print(f"numerical value of eigenenergy: {E}")

    print(f"absolute error in eigenenergy: {E_analytical-E:.3e} a.u.")
    print(f"relative error in eigenenergy: {abs((E_analytical - E)/E):.3e}")

    print(f"absolute error in wave function: {np.sum(np.abs(P-P_func))}")
    print(f"mean squared error in wave function: {np.sum((P-P_func)**2)/N}")

    fig1, ax = plt.subplots(2)
    # fig1.tight_layout()
    ax[0].set_title("large component")
    ax[1].set_title("small component")
    ax[0].plot(r, P_func, label="analytical")
    ax[1].plot(r, Q_func, label="analytical")
    ax[0].plot(r, P, label="numerical")
    ax[1].plot(r, Q, label="numerical")
    ax[0].scatter([r[a_c]], [P[a_c]], c="k", marker="x", label=r"$a_c$")
    ax[1].scatter([r[a_c]], [Q[a_c]], c="k", marker="x", label=r"$a_c$")

    # ax[0].plot(r[:a_c], P[:a_c], "-", label="numerical out")
    # ax[1].plot(r[:a_c], Q[:a_c], "-", label="numerical out")
    # ax[0].plot(r[a_c:], P[a_c:], "-", label="numerical in")
    # ax[1].plot(r[a_c:], Q[a_c:], "-", label="numerical in")

    ax[0].legend()
    # ax[0].set_xscale("log")
    ax[1].legend()
    # ax[1].set_xscale("log")
    fig1.suptitle(f"Z={Z}, M={M}, n={n}, l={l}, j={j}, h={h}")
    fig1.savefig("recent_calculation.png", dpi=300)

    plt.show()
