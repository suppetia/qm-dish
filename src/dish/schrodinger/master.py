import numpy as np

from dish.schrodinger import insch, outsch
from dish.schrodinger.coulomb.analytical import energy
from dish.util.numeric.adams import adams

from dish.util.radial.wave_function import RadialSchrodingerWaveFunction
from dish.util.radial.grid.grid import DistanceGrid
from dish.util.radial.integration import integrate_on_grid

from dish.util.math_util import count_nodes
from dish.util.atom import QuantumNumberSet

from typing import Union


def outer_classical_turning_point(V, E, l, r) -> int:
    return int(np.argmin(np.abs(E-(V+l*(l+1)/(2*r**2)))))


def master(n, l, Z, M, V, r,
           t: np.ndarray = None, h: float = None,
           charge: float = 0,
           order_adams: int = 7,
           order_insch: int = 7,
           E_guess: Union[float, str] = "auto",
           max_number_of_iterations: int = 50):

    mu = 1/(1+1/M)
    V *= mu  # account for nuclear recoil
    effective_charge = charge + 1  # an e- in distance feels the effective charge

    if E_guess == "auto":
        E_guess = energy(n, Z, M)

    if isinstance(r, DistanceGrid):
        r_grid = r
    else:
        # TODO: refactor properly
        if t is None or h is None:
            raise ValueError("values for 'r', 't' and 'h' must be provided if r isn't a DistanceGrid")
        # create a new Grid as a workaround
        r_grid = DistanceGrid(h, r[-1] / (np.exp(t[-1]) - 1), len(r))

    # leave out the first point as there will be divide by zero errors
    V_ = V[1:]
    r = r_grid.r[1:]
    r_prime = r_grid.rp[1:]
    h = r_grid.h

    # calculate/prepare outside the loop for speed improvements
    b = r_prime
    c = lambda E: -2 * b * (E - V_ - l * (l+1) / (2 * r ** 2))
    G = np.zeros((len(b), 2, 2))
    G[:, 0, 1] = b

    n_r = n-l-1  # number of radial nodes

    # bounds of eigenenergy
    E_u = np.inf  # highest energy with n_r nodes
    E_l = -np.inf   # lowest energy with n_r nodes

    energy_convergence = []

    it = 0
    while it < max_number_of_iterations:
        it += 1
        # print("Iteration:", it)

        a_c = outer_classical_turning_point(V_, E_guess, l, r)

        y_start_out = np.array(
            outsch.outsch(order=order_adams, p0=1, q0=-Z * mu / (l + 1), l=l, E=E_guess, V=-Z * mu / r,
                          r_grid=r_grid)
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

        if num_nodes < n_r:
            E_l = max(E_l, E_guess)  # keep track of the lower bound of energy (greatest energy with too few nodes)

            E_guess_new = E_guess * 0.9
        elif num_nodes > n_r:
            E_u = min(E_u, E_guess)  # keep track of the upper bound of energy (lowest energy with too many nodes)

            E_guess_new = E_guess * 1.1
        else:
            # check if the first derivative is continuous
            # if this is fulfilled, we found the eigenfunction and therefore the eigenenergy
            if np.isclose(y_out[-1, 1] - y_in[0, 1], 0, atol=1e-10):
                break

            E_guess_new = E_guess + (y_out[-1, 1]-y_in[0, 1])*y_out[-1, 0] / (2*np.trapz(y[:,0]**2, x=r))

        if E_guess_new < E_l:
            E_guess_new = (E_guess+E_l)/2
        elif E_guess_new > E_u:
            E_guess_new = (E_guess+E_u)/2

        energy_convergence.append(abs(E_guess - E_guess_new))
        E_guess = E_guess_new
    else:
        it = -1

    # all radial wavefunctions must be zero in the origin
    y = np.insert(y, obj=0, values=0, axis=0)

    # N = 1/np.sqrt(np.trapz(y[:,0]**2, x=r))
    N = 1 / np.sqrt(integrate_on_grid(y[:, 0] ** 2 + y[:, 1] ** 2, grid=r_grid, suppress_warning=True))

    y *= N

    return RadialSchrodingerWaveFunction(r_grid, y[:, 0], state=QuantumNumberSet(n,l), Psi_prime=y[:, 1]), E_guess, energy_convergence, a_c, it


if __name__ == "__main__":
    from dish.util.misc import find_suitable_number_of_integration_points_schrodinger
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
    R, E, E_convergence, a_c, num_iteration = master(n, l, Z, M, V, r, t, h,
                       order_adams = 11,
                       order_insch = 7,
                       E_guess = energy(n, Z, M) * 0.95,
                       max_number_of_iterations = 50
                       )
    print(f"finding eigenenergy and -function took {time.time()-t_start:.3f}s and {num_iteration} iterations")

    R_ = R
    R = np.stack([R_.Psi, R_.Psi_prime], axis=-1)

    import matplotlib.pyplot as plt
    from dish.schrodinger.coulomb.analytical import radial_function
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
