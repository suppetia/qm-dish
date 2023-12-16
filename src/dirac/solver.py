import time

from dirac.master import master
from util.atom import Nucleus
from util.misc import \
    DistanceGrid, \
    QuantumNumberSet, \
    find_suitable_number_of_integration_points_dirac, \
    parse_atomic_term_symbol
from util.atomic_units import a_0, convert_u2m_e
from util.wave_function import RadialDiracWaveFunction

from typing import Union
from dataclasses import dataclass

@dataclass
class SolvingParameters:
    order_AM: int
    order_indir: int
    max_number_of_iterations: int

@dataclass(frozen=True)
class SolvingResult:
    state: QuantumNumberSet
    nucleus: Nucleus
    potential_model: str
    r_grid: DistanceGrid
    wave_function: RadialDiracWaveFunction
    energy: float
    energy_convergence: float
    solving_parameters: SolvingParameters
    number_of_iterations: int
    solving_time: float


def solve(nucleus: Nucleus,
          state: Union[str, QuantumNumberSet],
          r_grid: Union[DistanceGrid, dict] = {"h": 0.005, "r0": 2e-6},
          potential_model: str = "Fermi",
          E_guess: Union[float, str] = "auto",
          order_AM: int = 9,  # order of the Adams-Moulton procedure
          order_indir: int = 7,  # order of the procedure tu
          max_number_of_iterations=20
          ) -> SolvingResult:
    """
    Solve the radial Dirac equation for Hydrogen-like atoms in state 'state'.
    :param nucleus: parameters of the nucleus
    :param state: electron state to find the wave function
    :param r_grid: grid on which the wave function is to be evaluated.
            Can be constructed from dict.
    :param potential_model: model of the charge distribution of the nucleus.
            Either 'Fermi', 'uniform' (charged ball) or 'point-like'.
            The default is 'Fermi'.
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
    """

    if isinstance(state, str):
        state = parse_atomic_term_symbol(state)

    # construct DistanceGrid from parameter dict
    if isinstance(r_grid, dict):
        h = r_grid.get("h", 0.005)
        r0 = r_grid.get("r0", 2e-6)
        if r_grid.get("r_max") is not None:
            r_grid = DistanceGrid(h, r0, r_max=r_grid.get("r_max"))
        elif r_grid.get("N", "auto") == "auto":
            N = find_suitable_number_of_integration_points_dirac(Z=nucleus.Z,
                                                                 M=nucleus.M,
                                                                 n=state.n,
                                                                 kappa=state.kappa,
                                                                 r_0=r0,
                                                                 h=h)
            r_grid = DistanceGrid(h, r0, N)
        else:
            try:
                N = int(r_grid['N'])
            except Exception:
                raise ValueError(f"Number of grid points 'N' must be an integer but is {type(r_grid['N'])}")
            r_grid = DistanceGrid(h, r0, N)

    if potential_model.lower() in ["f", "fermi"]:
        potential_model = "Fermi"
        V = nucleus.FermiPotential(r_grid.r)
    elif potential_model.lower() in ["u", "uniform", "ball", "uniformball"]:
        potential_model = "uniform"
        V = nucleus.UniformBallPotential(r_grid.r)
    elif potential_model.lower() in ["point", "point-like", "pointlike", "p", "coulomb", "c"]:
        potential_model = "point-like"
        V = nucleus.CoulombPotential(r_grid.r)
    else:
        raise ValueError(f"'potential_model' must be either 'Fermi', 'uniform' or 'point-like' but is {potential_model}.")

    t_start = time.perf_counter()
    Psi, E, dE, a_c, num_iteration = master(n=state.n,
                                        l=state.l,
                                        j=state.j,
                                        Z=nucleus.Z,
                                        M=nucleus.M,
                                        V=V,
                                        r=r_grid,
                                        order_adams=order_AM,
                                        order_indir=order_indir,
                                        E_guess=E_guess,
                                        max_number_of_iterations=max_number_of_iterations,
                                        )
    solving_time = time.perf_counter()-t_start

    result = SolvingResult(
        state=state,
        nucleus=nucleus,
        potential_model=potential_model,
        r_grid=r_grid,
        wave_function=Psi,
        energy=E,
        energy_convergence=dE,
        solving_parameters=SolvingParameters(order_AM, order_indir, max_number_of_iterations),
        number_of_iterations=num_iteration,
        solving_time=solving_time,
    )

    return result

# TODO: CLI using click
# def cli():
