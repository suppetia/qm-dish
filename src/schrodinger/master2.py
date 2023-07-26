import numpy as np

from util.numeric import insch, outsch
# handle the case if no compiled Fortran module is present
from util.numeric.adams import _NO_FORTRAN
if _NO_FORTRAN:
    from util.numeric.adams import adams
else:
    from util.numeric.adams import adams_f as adams

from util.math import count_nodes
from schrodinger.coulomb.analytical import energy

import matplotlib.pyplot as plt
from typing import Union


def outer_classical_turning_point(V, E, l, r) -> int:
    return int(np.argmin(np.abs(E-(V+l*(l+1)/(2*r**2)))))


def master(n, l, Z, M, V, r, t, charge=0,
           order_adams: int = 7,
           order_insch: int = 7,
           E_guess: Union[float, str] = "auto",
           max_number_of_iterations: int = 50):

    mu = 1/(1+1/M)
    effective_charge = charge + 1  # an e- in distance feels the effective charge

    if E_guess == "auto":
        E_guess = energy(n, Z, M)

    # calculate/prepare outside the loop for speed improvements
    b = np.diff(r, append=r[-1]) / np.diff(t, append=t[-1] + (t[-1] - t[-2]))
    c = lambda E: -2 * b * (E - V - l * (l + 1) / (2 * r ** 2))

    n_r = n-l-1  # number of radial nodes

    # bounds of eigenenergy
    E_u = np.inf  # highest energy with n_r nodes
    E_l = -np.inf   # lowest energy with n_r nodes

    # init values
    E_best = E_guess
    E_guess_new = E_guess

    num_parallel_guesses = 3

    it = 1
    while it <= max_number_of_iterations:
        print("Iteration:", it)
        energies_with_correct_num_nodes = []
        energies = np.empty(num_parallel_guesses, dtype=np.float64)
        discontinuities = np.empty(num_parallel_guesses, dtype=np.float64)
        a_cs = np.empty(num_parallel_guesses, dtype=np.uint32)
        y = np.empty((num_parallel_guesses, len(r), 2), dtype=np.float64)
        num_nodes = np.empty(num_parallel_guesses, dtype=np.uint8)
        G = np.zeros((len(b), 2*num_parallel_guesses, 2*num_parallel_guesses))
        y_start_out = np.empty((order_adams, 2 * num_parallel_guesses), dtype=np.float64)
        y_start_in = np.empty((order_adams, 2 * num_parallel_guesses), dtype=np.float64)
        for i in range(num_parallel_guesses):
            E_ = E_guess * (1+(i-num_parallel_guesses//2)*10**(-max(1, it/3)))
            energies[i] = E_

            a_cs[i] = outer_classical_turning_point(V, E_, l, r)
            # print(a_cs[i])

            y_start_out[:, 2*i:2*(i+1)] = np.array(outsch.outsch(order=order_adams, p0=1, q0=-Z*mu/(l+1), l=l, E=E_, V=-Z*mu/r,
                                                 r=r[:a_cs[i]+1],
                                                 t=t[:a_cs[i]+1])
                                   ).T
            y_start_in[:, 2*i:2*(i+1)] = np.array(insch.insch(order=order_insch, r=r[-order_adams:],
                                              mu=mu, l=l, E=E_, effective_charge=effective_charge)
                                  ).T

            G[:, 2*i, 2*i+1] = b
            G[:, 2*i+1, 2*i] = c(E_)  # for inward integration the sign of b and therefore G changes!

        # compute adams outside the loop for speed improvements
        y_out = adams(order_adams, "out", y_start_out, G[:np.max(a_cs)+1], h)
        y_in = adams(order_adams, "in", y_start_in, -G[np.min(a_cs)-1:], h)
        # y_in[np.isclose(y_in, 0, atol=1e-10)] = 0


        y_in_scaling = [[y_out[a_cs[i], 2*i]/y_in[-(len(r)-a_cs[i]), 2*i]]*2 for i in range(num_parallel_guesses)]
        y_in_scaling = np.ravel(y_in_scaling)
        y_in *= y_in_scaling
        for i in range(num_parallel_guesses):
            discontinuities[i] = y_out[a_cs[i], 2*i+1] - y_in[-(len(r)-a_cs[i]), 2*i+1]
            y[i] = np.append(y_out[:a_cs[i]+1, 2*i:2*(i+1)], y_in[-(len(r)-a_cs[i])+1:, 2*i:2*(i+1)], axis=0)

            num_nodes[i] = count_nodes(y[i, :, 0])

            if num_nodes[i] < n_r:
                E_l = max(E_l, energies[i])  # keep track of the lower bound of energy (greatest energy with too few nodes)

            elif num_nodes[i] > n_r:
                E_u = min(E_u, energies[i])  # keep track of the upper bound of energy (lowest energy with too many nodes)

            else:
                energies_with_correct_num_nodes.append(i)

        if energies_with_correct_num_nodes:
            min_discontinuity = np.min(np.abs(discontinuities[energies_with_correct_num_nodes]))
            idx_best = np.argwhere(abs(discontinuities) == min_discontinuity)[0,0]
            E_best = energies[idx_best]
            # check if the first derivative is continuous
            # if this is fulfilled, we found the eigenfunction and therefore the eigenenergy
            if np.isclose(discontinuities[idx_best]/y[idx_best, a_cs[idx_best], 1], 0, atol=1e-10):
                break


            E_guess_new = E_best + discontinuities[idx_best]*(y[idx_best, a_cs[idx_best], 0] / (2 * np.trapz(y[idx_best, :, 0]**2, x=r)))
            # if np.isclose(E_guess_new,E_best, atol=1e-8, rtol=1e-10):
            #     fig1, ax = plt.subplots(2)
            #     for ii in range(num_parallel_guesses):
            #
            #         ax[0].plot(r, y[ii, :, 0], ".")
            #         ax[1].plot(r, y[ii, :, 1], ".")
            #     break
        else:
            if num_nodes[-1] < n_r:  # if the last (highest) energy hasn't enough nodes all other energies don't fulfill the criterium either
                E_guess_new = E_l * 0.9
            elif num_nodes[0] > n_r:
                E_guess_new = E_u * 1.1

        if E_guess_new < E_l:
            E_guess_new = (E_best+E_l)/2
        elif E_guess_new > E_u:
            E_guess_new = (E_best+E_u)/2

        # if np.isclose(E_guess_new - E_guess, 0, atol=1e-12):
        #     print(discontinuities)
        #     break

        E_guess = E_guess_new
        print("new E_guess:", E_guess)
        it += 1

    N = 1/np.sqrt(np.trapz(y[idx_best, :, 0]**2, x=r))

    return N * y[idx_best], E_guess, a_cs[idx_best]


if __name__ == "__main__":
    from util.misc import find_suitable_number_of_integration_points
    Z = 2
    M = np.inf
    mu = 1 / (1 + 1 / M)

    n, l = 4,1

    #N = 400
    h = 0.02
    r0 = 0.0005  # 0.0005
    N = find_suitable_number_of_integration_points(Z, M, n, l, r0, h)
    t = (np.arange(N) + 1) * h
    r = r0 * (np.exp(t) - 1)

    V = - Z * 1/(1+1/M)/r

    import time
    t_start = time.perf_counter()
    R, E, a_c = master(n, l, Z, M, V, r, t,
                       order_adams = 9,
                       order_insch = 5,
                       E_guess = -.1,#"auto",
                       max_number_of_iterations = 1000
                       )
    print(f"finding eigenenergy and -function took {time.perf_counter()-t_start:.3f}s")

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

    ax[0].plot(r[:a_c], R[:a_c, 0], ".", label="numerical out")
    ax[1].plot(r[:a_c], R[:a_c, 1], ".", label="numerical out")
    ax[0].plot(r[a_c:], R[a_c:, 0], ".", label="numerical in")
    ax[1].plot(r[a_c:], R[a_c:, 1], ".", label="numerical in")
    # ax[0].plot(r[1:a_c], R[:a_c-1, 0], ".", label="numerical out shifted