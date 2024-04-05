import numpy as np
from scipy.optimize import curve_fit

# from dish.util.atom import Nucleus, QuantumNumberSet
from dish.util.radial.grid.grid import DistanceGrid, RombergIntegrationGrid


def construct_grid_from_dict(r_grid: dict,
                             nucleus: "dish.util.atom.Nucleus",
                             state: "dish.util.atom.QuantumNumberSet",
                             *, relativistic=True):
    """
    Construct a DistanceGrid (or RombergIntegrationGrid if possible) from the parameters of the grid given in the dict ``r_grid``.
    If not provided the values default to ``h = 5e-3``, ``r0 = 2e-6``, ``N = "auto"``.
    If ``N = "auto"`` the number is derived from the ``state`` as a suited value from the analytic solution.
    :param r_grid: parameter dict to construct the grid from
    :param nucleus: The nucleus for which the radial SE/DE should be solved. Only actually required if ``N = "auto"``.
    :param state:
    :param relativistic:
    :return:
    """
    h = r_grid.get("h", 0.005)
    r0 = r_grid.get("r0", 2e-6)
    if r_grid.get("r_max") is not None:
        return DistanceGrid(h, r0, r_max=r_grid.get("r_max"))
    elif r_grid.get("N", "auto") == "auto":
        if relativistic:
            from dish.util.misc import find_suitable_number_of_integration_points_dirac

            N = find_suitable_number_of_integration_points_dirac(Z=nucleus.Z,
                                                                 M=nucleus.M,
                                                                 n=state.n,
                                                                 kappa=state.kappa,
                                                                 r_0=r0,
                                                                 h=h)
        else:
            from dish.util.misc import find_suitable_number_of_integration_points_schrodinger

            N = find_suitable_number_of_integration_points_schrodinger(Z=nucleus.Z,
                                                                       M=nucleus.M,
                                                                       n=state.n,
                                                                       l=state.l,
                                                                       r_0=r0,
                                                                       h=h)
    else:
        try:
            N = int(r_grid['N'])
        except Exception:
            raise ValueError(f"Number of grid points 'N' must be an integer but is {type(r_grid['N'])}")

    k = np.log2(N - 1)
    if np.isclose(k - np.floor(k), 0):
        return RombergIntegrationGrid(h, r0, N)
    else:
        return DistanceGrid(h, r0, N)


def construct_grid_from_points(r: np.array) -> DistanceGrid:
    N = len(r)

    def grid_func(x, r0, h):
        return r0*(np.exp(x*h)-1)

    # construct the grid parameters from fitting a few points
    num_fit_points = min(30, len(r))
    fit_pts = np.arange(num_fit_points, dtype=np.int64)*N//num_fit_points
    popt, pcov = curve_fit(grid_func, fit_pts, r[fit_pts], p0=[1e-5, 1e-8])
    threshold = 1e-10
    if (pcov > threshold).any():
        for r0 in [1e-5, 1e-7, 1e-9, 1e-11]:
            for h in [1e-9, 1e-7, 1e-5, 1e-3, 1e-1]:
                popt, pcov = curve_fit(grid_func, fit_pts, r[fit_pts], p0=[r0, h])
                if not (pcov > threshold).any():
                    break
    if (pcov > threshold).any():
        raise ValueError("failed to construct grid from given points")
    h = popt[1]
    r0 = popt[0]

    k = np.log2(N - 1)
    if np.isclose(k - np.floor(k), 0):
        return RombergIntegrationGrid(h=h, r0=r0, k=k)
    else:
        return DistanceGrid(h, r0, N)


if __name__ == "__main__":

    r0 = 1e-8
    h = 2.9198e-7
    r_max = 3

    d_grid = DistanceGrid(h, r0, r_max=r_max)
    print(d_grid.N)
    # print(d_grid.r[-1])

    ri_grid = RombergIntegrationGrid.construct_similar_grid_from_distance_grid(d_grid)
    print(ri_grid.N)
    print(ri_grid.k)
    # print(ri_grid.r[-1])

    new_grid = construct_grid_from_points(ri_grid.r)
    print(type(new_grid))
    print(new_grid.h)
    # print(new_grid.r[-1])

    print(d_grid == new_grid)

