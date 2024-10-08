import time
from os import cpu_count

from dish.schrodinger.master import master
from dish.util.atom import Nucleus, QuantumNumberSet, parse_atomic_term_symbol
from dish.util.radial.grid.grid import DistanceGrid
from dish.util.radial.grid.construct_grid import construct_grid_from_dict
from dish.util.misc import SolvingResult, SolvingParameters

from typing import Union, Tuple, List

from multiprocessing import Pool

import logging
log = logging.getLogger(__name__)


def solve(nucleus: Nucleus,
          state: Union[str, QuantumNumberSet, Tuple[int, int, float]],
          r_grid: Union[DistanceGrid, dict] = {"h": 0.005, "r0": 2e-6},
          potential_model: str = "Fermi",
          m: float = 1,
          E_guess: Union[float, str] = "auto",
          order_AM: int = 9,  # order of the Adams-Moulton procedure
          order_insch: int = 7,  # order of the procedure tu
          max_number_of_iterations=20
          ) -> SolvingResult:
    """
    Solve the radial Dirac equation for Hydrogen-like atoms for a particle in state 'state'.

    :param nucleus: parameters of the nucleus
    :param state: electron state to find the wave function
    :param r_grid: grid on which the wave function is to be evaluated.
            Can be constructed from dict.
    :param potential_model: model of the charge distribution of the nucleus.
            Either 'Fermi', 'uniform' (charged ball) or 'point-like'.
            The default is 'Fermi'.
    :param m: mass of the particle for which the SE is solved in m_e.
            The default is 1 (for an electron).
    :param E_guess: initial guess for the energy of the state.
            Can be 'auto' which will use the analytical value for a point-like nucleus.
            The default is 'auto'.
    :param order_AM: order of the Adams-Moulton procedure which is used to solve the differential equation.
            The default is 9.
    :param order_indir: order of the procedure which is used to find the most inner points of the wave function.
            The default is 7.
    :param max_number_of_iterations: number of iterations after which the solving routine will stop.
            The default is 20.
    :return:
        The result of the solving routine 'master' as a 'SolvingResult'.
    :rtype: SolvingResult
    """

    if isinstance(state, str):
        state = parse_atomic_term_symbol(state)
    elif isinstance(state, Tuple):
        state = QuantumNumberSet(*state)

    # construct DistanceGrid from parameter dict
    if isinstance(r_grid, dict):
        r_grid = construct_grid_from_dict(r_grid, nucleus, m, state, relativistic=False)

    # evaluate the given potential model of the nucleus on the given grid
    if potential_model.lower() in ["f", "fermi"]:
        potential_model = "Fermi"
    elif potential_model.lower() in ["u", "uniform", "ball", "uniformball"]:
        potential_model = "uniform"
    elif potential_model.lower() in ["point", "point-like", "pointlike", "p", "coulomb", "c"]:
        potential_model = "point-like"
    else:
        raise ValueError(f"'potential_model' must be either 'Fermi', 'uniform' or 'point-like' but is {potential_model}.")
    V = nucleus.potential(r_grid.r, model=potential_model)

    t_start = time.perf_counter()
    Psi, E, dE, a_c, num_iteration = master(n=state.n,
                                        l=state.l,
                                        Z=nucleus.Z,
                                        M=nucleus.M,
                                        V=V,
                                        r=r_grid,
                                        m_particle=m,
                                        charge=nucleus.system_charge,
                                        order_adams=order_AM,
                                        order_insch=order_insch,
                                        E_guess=E_guess,
                                        max_number_of_iterations=max_number_of_iterations,
                                        )
    solving_time = time.perf_counter()-t_start

    result = SolvingResult(
        state=state,
        nucleus=nucleus,
        potential_model=potential_model,
        m=m,
        r_grid=r_grid,
        wave_function=Psi,
        energy=E,
        energy_convergence=dE,
        solving_parameters=SolvingParameters(order_AM, order_insch, max_number_of_iterations),
        number_of_iterations=num_iteration,
        solving_time=solving_time,
    )

    return result


def multiple_solve(nucleus: Nucleus,
                   states: List[Union[str, QuantumNumberSet, Tuple[int, int, float]]],
                   r_grid: Union[DistanceGrid, dict] = {"h": 0.005, "r0": 2e-6},
                   potential_model: str = "Fermi",
                   m: float = 1,
                   E_guess: Union[float, str, List[float]] = "auto",
                   order_AM: int = 9,  # order of the Adams-Moulton procedure
                   order_insch: int = 7,  # order of the procedure tu
                   max_number_of_iterations=20,
                   num_processes: int = -1
                   ) -> List[SolvingResult]:

    # construct DistanceGrid from parameter dict
    if isinstance(r_grid, dict):
        r_grid = construct_grid_from_dict(r_grid, nucleus, states[0])

    if isinstance(E_guess, list):
        if not len(E_guess) == len(states):
            raise ValueError("'E_guess' must be a single entry of an array-like of the same length as 'states'.")
    else:
        E_guess = [E_guess] * len(states)

    if num_processes < 1:
        logging.info("automatically choose the optimal cpu count")
        num_processes = cpu_count()
    if num_processes > cpu_count():
        logging.warning(f"Requested too many parallel processes. Limit to the available core count {cpu_count()}")
        num_processes = cpu_count()

    arguments = [(nucleus, states[i], r_grid, potential_model, m, E_guess[i],
                  order_AM, order_insch, max_number_of_iterations)
                 for i in range(len(states))]
    with Pool(num_processes) as pool:

        results: List[SolvingResult] = pool.starmap(solve, arguments)

    return results
