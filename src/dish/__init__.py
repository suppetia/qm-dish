import importlib.metadata
try:
    __version__ = importlib.metadata.version("qm-dish")
except importlib.metadata.PackageNotFoundError:
    __version__ = "failed to fetch version"


import dish.util
import dish.dirac
import dish.schrodinger

from dish.util.atom import Nucleus, parse_atomic_term_symbol, QuantumNumberSet
from dish.util.radial.grid import DistanceGrid, RombergIntegrationGrid
from dish.util.atomic_units import convert_units

from dish.dirac.solver import solve
from dish.util.radial.integration import (
    integrate_on_grid,
    radial_integral,
    matrix_element,
    mp_matrix_element
)

import logging
# logging.getLogger('dish').addHandler(logging.NullHandler())

