import dish.util
import dish.dirac
import dish.schrodinger

from dish.util.atom import Nucleus, parse_atomic_term_symbol, QuantumNumberSet
from dish.util.radial.grid import DistanceGrid
from dish.util.atomic_units import convert_units

from dish.dirac.solver import solve

import logging
# logging.getLogger('dish').addHandler(logging.NullHandler())

