import numpy as np

from typing import Union

from dirac.coulomb.analytical import energy
from schrodinger.coulomb.analytical import energy as energy_schrodinger

from dirac.outdir import outdir
from dirac.indir import indir
from util.numeric.adams import adams

from util.atomic_units import c, alpha
from util.math import count_nodes


def outer_classical_turning_point(V, W) -> int:
    return int(np.argmin(np.abs(W-V)))


def master(n, l, j, Z, M, V, r, t, h,
           order_adams: int = 7,
           order_indir: int = 7,
           E_guess: Union[float, str] = "auto",
           max_number_of_iterations: int = 50):

    num_integration_points = len(t)
    mu = 1/(1+1/M)

    kappa = -l - 1 if np.isclose(j, l+1/2) else l
    k = abs(kappa)

    gamma = np.sqrt(kappa**2-(alpha*Z*mu)**2)

    if E_guess == "auto":
        # E_guess = energy(n, kappa, Z, M) #* 0.99995
        E_guess = energy_schrodinger(n, Z, M) + c**2

    r_prime = r[-1] / (np.exp(t[-1]) - 1) * np.exp(t)

    a_mat = -r_prime * (kappa / r)
    b_mat = lambda W: -alpha * r_prime * (W - V + 2 * c**2)
    c_mat = lambda W: alpha * r_prime * (W - V)
    d_mat = -a_mat

    G = np.zeros((num_integration_points, 2, 2))
    G[:, 0, 0] = a_mat
    G[:, 1, 1] = d_mat

    # number of radial nodes
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

        print(W_guess)
        E_guess = W_guess + c ** 2

        a_c = outer_classical_turning_point(V, W_guess)
        # print(a_c)

        y_start_out = np.array(
            outdir(order=order_adams, Z=Z, kappa=kappa, W=W_guess, V=-Z * mu / r,
                   r=r[:a_c + 1], t=t[:a_c + 1])
        ).T

        y_start_in = np.array(
            indir(order=order_indir, r=r[-order_adams:], E=E_guess, kappa=kappa,
                  effective_charge=zeta)
        ).T
        #
        # offset = 0
        # while np.allclose(y_start_in, 0, atol=1e-15):
        #     offset += order_adams
        #     print(r[-order_adams-offset:-offset])
        #     y_start_in = np.array(
        #         indir(order=order_indir, r=r[-order_adams-offset:-offset], E=E_guess, kappa=kappa,
        #               effective_charge=zeta)
        #     ).T
        #     print(y_start_in)

        # for inward integration the sign of r_prime and therefore G changes!
        G[:, 0, 1] = b_mat(W_guess)
        G[:, 1, 0] = c_mat(W_guess)

        y_out = adams(order_adams, "out", y_start_out, G[:a_c + 1], h)
        y_in = adams(order_adams, "in", y_start_in, -G[a_c:], h)
        # y_in[np.isclose(y_in, 0, atol=1e-15)] = 0

        # make P=y[:,0] continuous
        y_in *= y_out[-1, 0] / y_in[0, 0]
        # print("y_in:", y_in[0,0])
        # print(y_start_in[:,0])

        y = np.append(y_out, y_in[1:], axis=0)

        num_nodes = count_nodes(y[:, 0])
        print("number of nodes:", num_nodes)

        if num_nodes < n_r:
            W_l = max(W_l, W_guess)  # keep track of the lower bound of energy (greatest energy with too few nodes)
            W_guess_new = W_guess * 0.9
        elif num_nodes > n_r:
            W_u = min(W_u, W_guess)  # keep track of the upper bound of energy (lowest energy with too many nodes)
            W_guess_new = W_guess * 1.1
        else:
            # check if Q is continuous
            # if this is fulfilled, we found the eigenfunction and therefore the eigenenergy
            # print(y_out[-1, 1] - y_in[0, 1])
            if np.isclose(y_out[-1, 1] - y_in[0, 1], 0, atol=1e-10):
                break

            W_guess_new = W_guess + c*(y_in[0, 1]-y_out[-1, 1])*y_out[-1, 0] / np.trapz(y[:, 0]**2 + y[:, 1]**2, x=r)

            # # if the new guess does not differ from the old guess the eigenfunction is as good as it gets
            # print(W_guess-W_guess_new)
            # if np.isclose(W_guess-W_guess_new, 0, atol=1e-15):
            #     break
        if W_guess_new < W_l:
            W_guess_new = (W_guess+W_l)/2
        elif W_guess_new > W_u:
            W_guess_new = (E_guess+W_u)/2

        W_guess = W_guess_new
    else:
        it = -1  # set it to -1 as a flag that the algorithm didn't converge

    N = 1/np.sqrt(np.trapz(y[:, 0]**2+y[:, 1]**2, x=r))

    y *= N
    P, Q = y.T

    return (P, Q), W_guess, a_c, it


if __name__ == "__main__":
    from util.misc import find_suitable_number_of_integration_points_dirac, parse_atomic_term_symbol

    Z = 1
    # n, l, j = 2, 0, 1/2
    n, l, j = parse_atomic_term_symbol("2p3/2")  # the algorithm does currently fail for this state
    n, l, j = parse_atomic_term_symbol("2s1/2")
    M = np.inf

    mu = 1 / (1 + 1 / M)
    kappa = -l - 1 if np.isclose(j, l+1/2) else l

    # N = 570
    h = 0.005
    r0 = 0.0005
    N = find_suitable_number_of_integration_points_dirac(Z, M, n, kappa, r0, h)
    print(f"number of integration points: {N}")
    t = np.arange(N) * h
    r = r0 * (np.exp(t) - 1)
    print(r[-1])
    r[np.isclose(r, 0, atol=1e-15)] = 1e-15


    V = - Z * mu/r

    import time

    t_start = time.time()
    (P, Q), E, a_c, num_iteration = master(n, l, j, Z, M, V, r, t, h,
                                      order_adams=11,
                                      order_indir=7,
                                      E_guess="auto",
                                      max_number_of_iterations=20
                                      )
    print(f"finding eigenenergy and -function took {time.time() - t_start:.3f}s and {num_iteration} iterations")

    import matplotlib.pyplot as plt
    from dirac.coulomb.analytical import radial_function, energy
    P_func, Q_func = radial_function(n, kappa, r, Z, M)
    E_analytical = energy(n, kappa, Z, M) - c**2
    print(f"analytical value of eigenenergy: {E_analytical}")
    print(f"numerical value of eigenenergy: {E}")

    print(f"absolute error in eigenenergy: {E_analytical-E:.3e} a.u.")
    print(f"relative error in eigenenergy: {abs((E_analytical - E)/E):.3e}")

    print(f"absolute error in wave function: {np.sum(np.abs(P-P_func))}")
    print(f"absolute error in wave function normalized by number of integration points: {np.sum(np.abs(P-P_func))/N}")

    fig1, ax = plt.subplots(2)
    # fig1.tight_layout()
    ax[0].set_title("large component")
    ax[1].set_title("small component")
    ax[0].plot(r, P_func, label="analytical")
    ax[1].plot(r, Q_func, label="analytical")
    ax[0].plot(r, P, label="numerical")
    ax[1].plot(r, Q, label="numerical")
    ax[0].scatter([r[a_c]], [P[a_c]], c="k", marker="x", label="outer classical turning point")
    ax[1].scatter([r[a_c]], [Q[a_c]], c="k", marker="x", label="outer classical turning point")

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
