import numpy as np

from schrodinger import insch, outsch
from schrodinger.coulomb.analytical import energy
# handle the case if no compiled Fortran module is present
from util.numeric.adams import _NO_FORTRAN
if _NO_FORTRAN:
    from util.numeric.adams import adams
else:
    from util.numeric.adams import adams_f as adams
from util.math import count_nodes

from typing import Union


def outer_classical_turning_point(V, E, l, r) -> int:
    return int(np.argmin(np.abs(E-(V+l*(l+1)/(2*r**2)))))


def master(n, l, Z, M, V, r, t, h, charge=0,
           order_adams: int = 7,
           order_insch: int = 7,
           E_guess: Union[float, str] = "auto",
           max_number_of_iterations: int = 50):

    mu = 1/(1+1/M)
    effective_charge = charge + 1  # an e- in distance feels the effective charge

    if E_guess == "auto":
        E_guess = energy(n, Z, M) * .95  # inject artificial error

    # calculate/prepare outside the loop for speed improvements
    #b = np.append([0], np.diff(r) / np.diff(t))
    b = r[-1]/(np.exp(t[-1])-1)*np.exp(t)
    # c = lambda E: -2 * b * (E - V - l * (l+1) / (2 * r ** 2))
    def c(E):
        return (-2 * b * (E - V - l * (l+1) / (2 * r ** 2)))
    G = np.zeros((len(b), 2, 2))
    G[:, 0, 1] = b

    n_r = n-l-1  # number of radial nodes

    # bounds of eigenenergy
    E_u = np.inf  # highest energy with n_r nodes
    E_l = -np.inf   # lowest energy with n_r nodes

    it = 0
    while it < max_number_of_iterations:
        it += 1
        # print("Iteration:", it)

        a_c = outer_classical_turning_point(V, E_guess, l, r)

        y_start_out = np.array(
            outsch.outsch(order=order_adams, p0=1, q0=-Z * mu / (l + 1), l=l, E=E_guess, V=-Z * mu / r,
                          r=r[:a_c+1],
                          t=t[:a_c+1])
            ).T
        y_start_in = np.array(insch.insch(order=order_insch, r=r[-order_adams:],
                                          mu=mu, l=l, E=E_guess, effective_charge=effective_charge)
                              ).T


        G[:, 1, 0] = c(E_guess)  # for inward integration the sign of b and therefore G changes!

        # # calculate in and out integration in the same step
        # G_ = np.zeros((max(a_c+1, len(b)-a_c), 4, 4))
        # G_[:a_c+1, :2, :2] = G[:a_c + 1]
        # G_[-(len(G)-a_c):, 2:, 2:] = -G[a_c:]
        # y_start = np.append(y_start_out, y_start_in, axis=1)

        # with Pool(processes=2) as pool:
        #     adams_out = pool.apply_async(adams, (order_adams, "out", y_start_out, G[:a_c+1], h))
        #     adams_in = pool.apply_async(adams, (order_adams, "in", y_start_in, -G[a_c:], h))
        #     pool.close()
        #     pool.join()
        #     y_out = adams_out.get()
        #     y_in = adams_in.get()

        # y_out = adams.adams_schrodinger(order_adams, "out", y_start_out, b[:a_c+1], c(E_guess)[:a_c+1], h)
        # y_in = adams.adams_schrodinger(order_adams, "in", y_start_in, -b[a_c:], -c(E_guess)[a_c:], h)
        y_out = adams(order_adams, "out", y_start_out, G[:a_c+1], h)
        y_in = adams(order_adams, "in", y_start_in, -G[a_c:], h)
        y_in[np.isclose(y_in, 0, atol=1e-15)] = 0

        y_in *= y_out[-1, 0]/y_in[0, 0]  # make R=y[:,0] continuous
        # y_out *= y_in[0, 0] / y_out[-1, 0]

        y = np.append(y_out, y_in[1:], axis=0)

        num_nodes = count_nodes(y[:, 0])
        # print(num_nodes)

        # fig, ax = plt.subplots(nrows=2, num=0)
        # ax[0].clear()
        # ax[1].clear()
        # ax[0].plot(r[:a_c+1], y_out[:, 0])
        # ax[0].plot(r[a_c:], y_in[:, 0])
        # ax[1].plot(r[:a_c+1], y_out[:, 1])
        # ax[1].plot(r[a_c:], y_in[:, 1])
        # y_out2 = adams.adams(order_adams, "out", y_start_out, G[:], h)
        # y_in2 = adams.adams(order_adams, "out", y_start_in[::-1], G[::-1], h)[::-1]
        # y_in2 *= y_out2[a_c, 0] / y_in2[a_c, 0]  # make R=y[:,0] continues
        # ax[0].plot(r, y_out2[:, 0])
        # ax[0].plot(r, y_in2[:, 0])
        # ax[1].plot(r, y_out2[:, 1])
        # ax[1].plot(r, y_in2[:, 1])
        # ax[0].plot([r[a_c], r[a_c]], [-1, 1], "k")
        # ax[1].plot([r[a_c], r[a_c]], [-1, 1], "k")
        # ax[0].set_ylim(-1, 1)
        # ax[1].set_ylim(-1, 1)
        # fig.show()
        # while not plt.waitforbuttonpress():
        #     continue

        if num_nodes < n_r:
            E_l = max(E_l, E_guess)  # keep track of the lower bound of energy (greatest energy with too few nodes)

            # if n_r - num_nodes > 5:  # if the guess is far off, increase the adjustment steps in energy
            #     E_guess_new = E_guess * 0.5
            # else:
            E_guess_new = E_guess * 0.9
        elif num_nodes > n_r:
            E_u = min(E_u, E_guess)  # keep track of the upper bound of energy (lowest energy with too many nodes)
            # if num_nodes - n_r > 5:  # if the guess is far off, increase the adjustment steps in energy
            #     E_guess_new = E_guess * 2
            # else:
            E_guess_new = E_guess * 1.1
        else:
            # check if the first derivative is continuous
            # if this is fulfilled, we found the eigenfunction and therefore the eigenenergy
            # print(y_out[-1, 1] - y_in[0, 1])
            if np.isclose(y_out[-1, 1] - y_in[0, 1], 0, atol=1e-10):
                break

            E_guess_new = E_guess + (y_out[-1, 1]-y_in[0, 1])*y_out[-1, 0] / (2*np.trapz(y[:,0]**2, x=r))

        if E_guess_new < E_l:
            E_guess_new = (E_guess+E_l)/2
        elif E_guess_new > E_u:
            E_guess_new = (E_guess+E_u)/2

        E_guess = E_guess_new
        # print("new E_guess:", E_guess)

    N = 1/np.sqrt(np.trapz(y[:,0]**2, x=r))

    return N * y, E_guess, a_c, it


if __name__ == "__main__":
    from util.misc import find_suitable_number_of_integration_points_schrodinger
    Z = 1
    n, l = 5,0#10, 3
    M = np.inf
    mu = 1 / (1 + 1 / M)

    # N = 570
    h = 0.0005#0005 #0.00001
    r0 = 0.0005
    N = find_suitable_number_of_integration_points_schrodinger(Z, M, n, l, r0, h)
    print(N)
    t = np.arange(N) * h
    r = r0 * (np.exp(t) - 1)
    r[np.isclose(r, 0, atol=1e-15)] = 1e-15


    V = - Z * 1/(1+1/M)/r

    import time
    t_start = time.time()
    R, E, a_c, num_iteration = master(n, l, Z, M, V, r, t, h,
                       order_adams = 11,
                       order_insch = 7,
                       E_guess = "auto",
                       max_number_of_iterations = 50
                       )
    print(f"finding eigenenergy and -function took {time.time()-t_start:.3f}s and {num_iteration} iterations")

    import matplotlib.pyplot as plt
    from schrodinger.coulomb.analytical import radial_function, energy
    R_func = radial_function(n, l, r, Z, M)
    E_analytical = energy(n, Z, M)
    print(f"analytical value of eigenenergy: {E_analytical}")
    print(f"numerical value of eigenenergy: {E}")

    print(f"absolute error: {E_analytical-E:.3e} a.u.")
    print(f"relative error: {abs((E_analytical - E)/E):.3e}")

    fig1, ax = plt.subplots(2)
    ax[0].plot(r, R_func, label="analytical")
    ax[1].plot(r[:-1], np.diff(R_func) / np.diff(r), label="analytical")

    ax[0].plot(r[:a_c], R[:a_c, 0], "-", label="numerical out")
    ax[1].plot(r[:a_c], R[:a_c, 1], "-", label="numerical out")
    ax[0].plot(r[a_c:], R[a_c:, 0], "-", label="numerical in")
    ax[1].plot(r[a_c:], R[a_c:, 1], "-", label="numerical in")

    ax[0].legend()
    # ax[0].set_xscale("log")
    ax[1].legend()
    # ax[1].set_xscale("log")
    fig1.suptitle(f"Z={Z}, M={M}, n={n}, l={l}, h={h}")
    fig1.savefig("recent_calculation.png", dpi=300)

    plt.show()
