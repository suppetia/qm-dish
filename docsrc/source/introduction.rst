Introduction into dish
======================

*dish* is a lightweight solver the Schrödinger and Dirac equation for hydrogenic systems.
It allows to find the energy and the wave function for states in these systems and provides a framework to calculate some basic matrix elements.

While *dish* is capable of both relativistic and non-relativistic calculations, the main use case are computations in the relativistic framework and therefore the focus of this documentation will be on these.
Working with non-relativistic wave functions will in the most cases work as a drop-in replacement (except for the mismatch in dimensions), but nevertheless the api is discussed in the :ref:`last section <label-NonRelativisticCalculations>`.


Hartree Atomic Units
--------------------

All calculations are performed internally dimensionless in `Hartree atomic units`_.
All classes and functions therefore expect values in atomic units.
*dish* provides a utility function :func:`dish.util.atomic_units.convert_units` to convert from and to atomic units.
More information :ref:`below <label-convertUnits>`.



Defining the Hydrogen-like System
---------------------------------

The hydrogenic system is defined by the properties of the nucleus and the model used to describe its potential.
There are three potential models implemented:

1. A point-like nucleus, which results in a pure Coulomb potential.

.. math:: \rho(r) = Z\delta(r)

2. A ball-like nucleus, which is modeled by a homogeneously charged sphere.

.. math:: \rho(r) = \begin{cases}\rho_0 &, r \le R_0\\ 0 &, r > R_0\end{cases}

3. A nucleus with a charge density distribution described by a Fermi distribution:

.. math:: \rho(r) = \frac{\rho_0}{1+\exp((r-c)/a)}

To store the information about the nucleus a :class:`dish.util.atom.Nucleus`-object is used:

.. code-block:: python

   from dish.util.atom import Nucleus

   nuc = Nucleus(Z:float, M:float=<optional>,
                 R_rms:float=<optional>,
                 R0:float=<optional>,
                 c:float=<optional>, a:float=<optional>,
                 system_charge:float=<optional>)


The parameters are:

- ``Z``: The number of protons which is the nuclear charge.
- ``M``: The mass of the nucleus. This is used to take in account for nuclear recoil for non-relativistic calculations. Use *numpy.inf* to have a static nucleus.
- The radius of the charge distribution (not defined for point-like model :math:`\rho(r) = Z\delta(r)` and therefore optional):
  It can be passed either via the root-mean-square radius :math:`R_\text{rms} = \sqrt{\langle r^2\rangle}` via the parameter ``R_rms`` or via model specific parameters:

  2. For a homogeneously charged sphere the radius :math:`R_0` can be passed via the parameter ``R0``.
  3. For a Fermi charge distribution the Fermi parameter :math:`c` be be passed by the correspondingly named parameter ``c``.

  Either ``R_rms``, ``R0`` or ``c`` can be given (but not more than one) and the other parameters are calculated.

  .. note::

     For small values of :math:`R_{\text{rms}}` value the Fermi model is not applicable with the default value of :math:`a`.
     If :math:`c` can't be calculated from :math:`R_0` or :math:`R_rms` an error will be thrown if a Fermi-like model is requested.

- (optional) ``a``: To specify the Fermi distribution this *diffuseness* parameter is required.
  It defaults to :math:`2.3 \text{fm} /a_0 / (4\cdot\ln(3))` as described in *Parpia and Mohanty, Phys.Rev.A, 46 (1992), Number 7*
- (optional) ``system_charge``: The charge of the hydrogenic system. This defaults to ``Z-1``.

The charge density and the potential can be evaluated at specific points by calling

.. code-block:: python

   rho = nuc.charge_density(r: float|numpy.ndarray|Distancegrid, model:str=<optional>)

   V = nuc.potential(r: float|numpy.ndarray|Distancegrid, model:str=<optional>))

To the argument ``r`` a *float*, *numpy.ndarray* of floats or a :class:`dish.util.grid.DistanceGrid` should be passed to evaluate the charge density or potential at the given points.
The argument ``model`` is optional and expects a *string* that specifies, which model for the nucleus should be used.
Valid options are (case-insensitive):

1. For a point-like nuclear model: "point", "point-like", "pointlike", "p", "coulomb", "c"
2. For a homogeneously charged sphere: "u", "uniform", "ball", "uniformball"
3. For a Fermi charge distribution: "f", "fermi"

The potential models and their charge distributions are implemented in :mod:`dish.util.potential`.
To each model there are corresponding classes for the potential and the charge density distribution,
which are subclasses of :class:`dish.util.potential.PotentialModel` and :class:`dish.util.potential.ChargeDistribution` respectively.

.. note::

   It is easily extendable to a custom potential by implementing a subclass of :class:`dish.util.atom.Nucleus`
   and overriding the :meth:`potential` (and optionally the :meth:`charge_density`) method:

   .. code-block:: python

      from dish.util.atom import Nucleus

      class MyCustomNucleus(Nucleus):

          def potential(self, r, model):
              if model.lower() == "my-custom-model-name":
                  # my implementation
                  return result
              else:
                  return super().potential(r, model)

          def charge_density(self, r, model):
              if model.lower() == "my-custom-model-name":
                  # my implementation
                  return result
              else:
                  return super().charge_density(r, model)

   If your nuclear model might be useful for others, you are very welcome to submit a pull-request and implement subclasses
   of :class:`dish.util.potential.PotentialModel` and :class:`dish.util.potential.ChargeDistribution`
   and their respective calls in :meth:`Nucleus.potential()`/:meth:`Nucleus.charge_density()`.


Defining the Grid
-----------------

All operations are performed on a finite grid of the form

.. math::

   r(t) = r_0 \cdot (\exp(h\cdot t)-1)

where :math:`t` is a linear grid

.. math::

   t(i) = i \cdot h\,, \quad i = 0,\dots,N-1 \,.

The grid is therefore defined by the parameters :math:`r_0, h, N`,
and is implemented as the class :class:`dish.util.radial.grid.grid.DistanceGrid`.

.. code-block:: python

   from dish.util.radial.grid.grid import DistanceGrid

   grid = DistanceGrid(r0:float, h:float, N:int)
   # or
   grid = DistanceGrid(r0:float, h:float, r_max:float = <value>)

Using the latter method and passing a maximal radial distance :math:`N` is calculated automatically.


A special grid that is better suited if you want to :ref:`calculate matrix elements <label-MatrixElements>` of later
or calculate any other integral using wave functions from *dish* is a :class:`dish.util.radial.grid.grid.RombergIntegrationGrid`.
This has essentially the same structure but the number of grid points is :math:`N = 2^k + 1` for a positive integer :math:`k`.
This makes integration using `Romberg's method`_ possible which yields higher precession most of the times.
(See chapter ??? of the underlying thesis for more details.)

It can be constructed by passing ``N`` or ``k`` or from a :class:`DistanceGrid` by increasing the number of points by lowering ``h`` until the condition is met.

.. code-block:: python

   from dish.util.radial.grid.grid import RombergIntegrationGrid

   r_grid = RombergIntegrationGrid(r0:float, h:float, N:int)
   r_grid = RombergIntegrationGrid(r0:float, h:float, k:int = <value>)
   # or
   r_grid = RombergIntegrationGrid.construct_similar_grid_from_distance_grid(grid:DistanceGrid)


The actual values of the exponential grid :math:`r[i]` and the linear grid :math:`t[i]` can be accessed using

.. code-block:: python

   r_grid.r  # exponential grid
   r_grid.t  # linear grid
   r_grid.rp # == r_prime = dr/dt


Defining Electronic States
--------------------------

An electronic state in a hydrogenic system is defined by the four quantum numbers :math:`n,l,j` and :math:`m`.
The orbital angular projection quantum number :math:`m` is important only for the spherical part of the wave function.
The radial part of the wave function is therefore defined by :math:`n,l,j`.
To store these values use an instance of :class:`dish.util.atom.QuantumNumberSet`.

.. code-block:: python

   from dish.util.atom import QuantumNumberSet

   state = QuantumNumberSet(n:int, l:int, j:float)
   state = QuantumNumberSet(n:int, kappa:float = <value>)

It is common to combine :math:`l` and :math:`j` into the Dirac quantum number :math:`\kappa = \mp (j+\frac{1}{2}), \text{for } j = l \pm \frac{1}{2}`.
One can access this as an attribute ``state.kappa``.

Usually a state is written not in terms of :math:`n,l,j` but in spectroscopic notation using a term symbol,
e.g. :math:`1s_{\frac{1}{2}}` instead of :math:`(n,l,j) = (1,0,\frac{1}{2})` or :math:`4d_{\frac{3}{2}}` instead of :math:`(4,2,\frac{3}{2})`.
Since for a hydrogenic system just :math:`j = l \pm \frac{1}{2}` are possible also a :math:`+` or :math:`-` are common,
e.g. :math:`1s+ = 1s_{\frac{1}{2}}` or :math:`4d- = 4d_{\frac{3}{2}}`.

Both versions can be parsed into a :class:`QuantumNumberSet` using :func:`dish.util.atom.parse_atomic_term_symbol`.
Alternatively the notation ``"n[l]j"`` can be used, which is especially useful for higher :math:`l`, or any combination of these versions.

.. code-block:: python

   from dish.util.atom import parse_atomic_term_symbol

   state: QuantumNumberSet = parse_atomic_term_symbol(state_repr:str)
   # e.g.
   parse_atomic_term_symbol("1s+")
   parse_atomic_term_symbol("4d3/2")
   parse_atomic_term_symbol("4[2]-")


.. note::

   Even through for :math:`s`-states only :math:`\text{n}s_{\frac{1}{2}}` exists
   and :math:`\text{n}s_{-\frac{1}{2}}` does not, it is required to specify the states as
   ``"<n>s+"`` or ``"<n>s1/2"`` to distinguish it from :ref:`non-relativistic states <label-NonRelativisticCalculations>`.


Having defined the system, the grid and the state one can perform the calculation to find the electronic wave function and the states energy.


Finding the states energy and wave function
-------------------------------------------


.. code-block:: python

   from dish.schrodinger.solver import solve
   from dish.dirac.solver import solve

   result = solve(nucleus: dish.util.atom.Nucleus,
                  state: dish.util.atom.QuantumNumberSet,
                  r_grid: dish.util.radial.grid.grid.DistanceGrid = <optional:dict(r0=1e-6, h=1e-4)>,
                  potential_model: str = <optional:"Fermi">,
                  E_guess: float = <optional:"auto">,
                  order_AM: int = <optional:9>,
                  order_indir: int = <optional:7>,
                  max_number_of_iterations: int = <optional:20>
                 )

   E = result.energy  # actually E-c^2
   wf = result.wave_function  # RadialDiracWaveFunction

   wf.f        # large component
   wf.g        # small component
   wf.r_grid   # grid


   from dish.dirac.solver import solve

   result = solve(nucleus: dish.util.atom.Nucleus,
                  state: dish.util.atom.QuantumNumberSet,
                  r_grid: dish.util.radial.grid.grid.DistanceGrid = <optional:dict(r0=1e-6, h=1e-4)>,
                  potential_model: str = <optional:"Fermi">,
                  E_guess: float = <optional:"auto">,
                  order_AM: int = <optional:9>,
                  order_indir: int = <optional:7>,
                  max_number_of_iterations: int = <optional:20>
                 )

   E = result.energy  # actually E-c^2
   wf = result.wave_function  # RadialSchrodingerWaveFunction

   wf.R        # radial function
   wf.Q        # dR/dr
   wf.r_grid   # grid


.. _label-convertUnits:

Converting values to atomic units
---------------------------------

.. code-block:: python

   from dish.util.atomic_units import convert_units

   convert_units(old_unit: str|float,
                 new_unit: str|float,
                 value: float = <optional:1.>,
                 old_unit_exp: float = 1,
                 new_unit_exp: float = 1
                )

   # e.g.
   convert_units("u", "m_e")
   convert_units("J", "E_h")
   convert_units("eV", "E_h")
   convert_units("m^2", "a_0^2")
   convert_units("m", "a_0", old_unit_exp=2, new_unit_exp=2)

.. math::

   \mathrm{new\_value} = \frac{\left(\mathrm{value} \cdot \mathrm{new\_unit}\right)^\mathrm{old\_unit\_exp}}{\mathrm{new\_unit}^\mathrm{new\_unit\_exp}}



.. _label-MatrixElements:

Calculation of Matrix Elements
------------------------------

.. code-block:: python

   from dish.util.radial.integration import integrate_on_grid

   integrate_on_grid(y:numpy.ndarray, grid:DistanceGrid)


.. math::

   \langle n_1 \kappa_1\mid O\mid n_2 \kappa_2\rangle = \int_0^\infty y(r) dr


   \langle n_1 \kappa_1\mid \gamma_5 \mid n_2\kappa_2 \rangle = i \int_0^\infty (-f_1 g_2 + g_1 f_2) d r


.. code-block:: python

   wf1 = solve(...).wave_function
   wf2 = solve(...).wave_function

   # assure wf1 and wf2 are evaluated on the same grid
   integrate_on_grid(-wf1.f*wf2.g + wf1.g * wf2.f, grid=wf1.grid) * 1j


.. code-block:: python

   from dish.util.radial.operator import BraOperator, SymbolicMatrixOperator

   y_5 = SymbolicMatrixOperator([[0,1], [1,0]])

   BraOperator(wf1) * y_5 * wf2


.. code-block:: python

   from dish.util.radial.operator import BraOperator, RadialOperator

   r_hat = RadialOperator(lambda r: r)

   BraOperator(wf1) * r_hat * wf2


.. code-block:: python

   from dish.util.radial.operator import BraOperator, RadialOperator, SymbolicMatrixOperator

   r_hat = RadialOperator(lambda r: r)

   O = SymbolicMatrixOperator([[1, 1+r_hat], [-5*r_hat, r_hat*r_hat]])

   BraOperator(wf1) * O * wf2



A High-level Interface for Operators
------------------------------------



.. _label-NonRelativisticCalculations:

Non-relativistic Calculations
-----------------------------

- electronic state


Functions/Classes used in this introduction
-------------------------------------------

.. autoclass:: dish.util.atom.Nucleus
   :noindex:

.. automodule:: dish.util.potential
   :members:

.. autoclass:: dish.util.radial.wave_function.RadialDiracWaveFunction



.. _Hartree atomic units: https://en.wikipedia.org/wiki/Hartree_atomic_units

.. _Romberg's method: https://en.wikipedia.org/wiki/Romberg's_method