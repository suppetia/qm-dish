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
- ``M``: The mass of the nucleus. This is used to take in account for nuclear recoil for non-relativistic calculations. Use *numpy.inf* to have a static nucleus. (This is the default value.)
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

To the argument ``r`` a *float*, *numpy.ndarray* of floats or a :class:`dish.util.radial.grid.grid.DistanceGrid` should be passed to evaluate the charge density or potential at the given points.
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

   An example implementation of a Yukawa potential is shown in the :ref:`examples section <label-examplesYukawa>`.


Defining the Grid
-----------------

All operations are performed on a finite grid of the form

.. math::

   r(t) = r_0 \cdot (\mathrm{e}^{t}-1)\,,

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
It has essentially the same structure but the number of grid points is :math:`N = 2^k + 1` for a positive integer :math:`k`.
This makes integration using `Romberg's method`_ possible which yields higher precession most of the times.
(See chapter 3.7 of the underlying thesis for more details.)

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
Havin defined the state, the nucleus and the grid, one can solve the Dirac equation to find the energy and the wave function of the given state by calling the function
:func:`dish.dirac.solver.solve`:

.. code-block:: python

   from dish.dirac.solver import solve

   result = solve(nucleus: dish.util.atom.Nucleus,
                  state: dish.util.atom.QuantumNumberSet,
                  r_grid: dish.util.radial.grid.grid.DistanceGrid = <optional:dict(r0=1e-6, h=1e-4)>,
                  potential_model: str = <optional:"Fermi">,
                  E_guess: float = <optional:"auto">,
                  order_AM: int = <optional:5>,
                  order_indir: int = <optional:7>,
                  max_number_of_iterations: int = <optional:20>
                 )

   E = result.energy  # actually E-c^2
   wf = result.wave_function  # RadialDiracWaveFunction


The objects constructed in the last few sections should be passed to the parameters ``nucleus``, ``state`` and ``r_grid``.
The model of the nuclear potential can be specified by passing a corresponding alias (see above) to the parameter ``potential_model``. By default a Fermi-like model is used as specified by "Fermi".
To ``m`` the mass of he particle for which Dirac's equation should be solved can be passed. The default value is :math:`1` (a.u.) which is the mass of an electron.
By passing the mass of a muon :math:`m_\mu \approx 207` (a.u.) one can also obtain muonic wave functions. See chapter 4.1.3 of the thesis for more details.
The value passed to ``E_guess`` is used as an initial guess for the states energy. This usually expects a *float* but the default *string* value "auto" uses the analytic solution for a Coulomb potential.
Changing the values of ``order_AM``, ``order_indir`` and ``max_number_of_iterations`` the solving procedure can be adjusted.
The default values have proven to be reliable and to yield good results. For more details refer to chapters 3 and 4 of the thesis.

The :func:`dish.dirac.solve`-function returns a :class:`dish.util.misc.SolvingResult`-object in which all important information about the solving process are stored.


Wave Functions
--------------

The wave function, calculated using :meth:`solve`-function from the previous section, is stored in a :class:`dish.util.radial.wave_function.RadialDiracWaveFunction`.
Bound solutions of Dirac's equation are of the form

.. math::

   \psi_{n\kappa m}(r,\theta,\varphi) = \frac{1}{r}\begin{pmatrix} if_{n\kappa}(r)\Omega_{\kappa,m}(\theta,\varphi)\\ g_{n\kappa}(r)\Omega_{-\kappa, m}(\theta,\varphi) \end{pmatrix} \,.

The radial parts :math:`f` and :math:`g` evaluated on the :class:`DistanceGrid` ``grid`` can be accessed using the corresponding fields:

.. code-block:: python

   wf.f     # large component values -> stored in a numpy array
   wf.g     # small component values -> stored in a numpy array

Additionally, information about the grid and the state to which the wave function belongs are stored:

.. code-block:: python

   wf.r     # grid points -> stored in a numpy array
   wf.grid  # the DistanceGrid instance itself
   wf.state # the QuantumNumberSet instance of the state

Having found the wave function on a particular grid, the wave function can be interpolated on an arbitrary grid or even at arbitrary distance using cubic spline interpolation:

.. code-block:: python

   wf.interpolate_at(r: DistanceGrid)       # new RadialDiracWaveFunction
   wf.interpolate_values(r: numpy.ndarray)  # values in an numpy array

To use the wave functions from another tool the values can be exported in a plain text file using the :meth:`write_to_file`-method:

.. code-block:: python

   wf.write_to_file(filename: str)




.. _label-convertUnits:

Converting values to atomic units
---------------------------------

To convert from and to atomic unit use the utility function

.. code-block:: python

   from dish import convert_units
   convert_units(old_unit: str|float,
                 new_unit: str|float,
                 value: float = <optional:1.>,
                 old_unit_exp: float = <optional:1>,
                 new_unit_exp: float = <optional:1>
                )

which calculates

.. math::

   \mathrm{new\_value} = \frac{\left(\mathrm{value} \cdot \mathrm{new\_unit}\right)^\mathrm{old\_unit\_exp}}{\mathrm{new\_unit}^\mathrm{new\_unit\_exp}}\,.

The fields ``old_unit`` and ``new_unit`` expect the alias of the old/new unit or alternatively a conversion factor to convert to the corresponding SI units of the same dimension.

.. note::

   There are no checks of the dimension performed! Make sure to only convert units of matching dimensions.



.. code-block:: python

   # examples of common conversions
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

Having the relativistic radial wave functions :math:`\mid n_1\kappa_1\rangle` and :math:`\mid n_2\kappa_2\rangle` of two states, one can calculate radial matrix elements

.. math::

   \langle n_1 \kappa_1\mid \hat{o}_r \mid n_2 \kappa_2\rangle = \int_0^\infty \begin{pmatrix} -if_{n_1\kappa_1}(r) & g_{n_1\kappa_1}(r) \end{pmatrix} \hat{o}_r(r) \begin{pmatrix} if_{n_2\kappa_2}(r)\\g_{n_2\kappa_2}(r) \end{pmatrix} dr \,,

where :math:`\hat{o}_r(r)` is a radial operator.
In this formula the Jacobian determinant $r^2$ cancels with the factor $1 / r$ of the wave functions. Because of this, the radial wave functions are stored without this factor to avoid unnecessary loss of precision due to numerical errors when dividing by numbers very close to zero.

To calculate these matrix elements two interfaces are provided. The low-level interface just performs the evaluation of a radial integral on a :class:`DistanceGrid` or its subclass :class:`RombergIntegrationGrid`:

.. code-block::

   from dish.util.radial.integration import integrate_on_grid

   integrate_on_grid(y:numpy.ndarray, grid:DistanceGrid)


This offers the most flexibility as :math:`y` can be any array of values at the grid points while taking the integration mathematics off the user.

.. code-block:: python

   from dish.util.radial.integration import integrate_on_grid

   integrate_on_grid(y:numpy.ndarray, grid:DistanceGrid)

Passing a :class:`RombergIntegrationGrid` to the parameter ``grid``, Romberg's method will be used for integration, and if a :class:`DistanceGrid` instance is passed, it will simply fall back to the trapezoidal rule.
As discussed above and in chapter 3.7 of the thesis a :class:`RombergIntegrationGrid` should be used if possible.

A High-level Interface for Operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively the high-level interface implemented in the module :mod:`dish.util.radial.operator` can be used. It enables a syntax very similar to the Bra-ket notation-

You can define operators which extend the base-class :class:`dish.util.radial.operator.AbstractOperator` and apply them on instances of subclasses of :class:`dish.util.radial.wave_function.RadialWaveFunction`.
In the case of scalar or matrix operators, a new wave function is returned with values modified by the effect of the operator.
Also, a :class:`dish.util.radial.operator.BraOperator` can be constructed from a :class:`RadialWaveFunction`-object which performs the integration when it is applied on a :class:`RadialWaveFunction`-instance, and therefore this returns a scalar value.
Applying an operator is done by multiplying the operator from the left to a wave function:

.. code-block:: python

   from dish.util.radial.operator import BraOperator, AbstractOperator

   # this is actually not valid but a subclass should be used
   op = AbstractOperator()

   op * wf1  # returns a new wave function
   BraOperator(wf1) * op * wf2  # returns a scalar value
   # e.g. calculating an expectation value becomes
   BraOperator(wf1) * op * wf2

Operators can be chained and algebraic rules hold. The evaluation is performed from the right to the left:

.. code-block:: python

   # let op1, op2, op3 be instances of subclasses of AbstractOperator

   op1 * op2 * wf1 # == op1 * (op2 * wf1)
   op1 * (op2 + op3) * wf1 # == op1 * (op2 * wf1 + op3 * wf1)


.. note::

   The discussed syntax can be used for both, relativistic and non-relativistic wave functions. Note, that in the case of non-relativistic case the wave function is scalar and hence no matrix operators can be used.
   If a scalar operator :math:`\hat{a}_r` is applied on a relativistic wave function, it is interpreted as :math:`\hat{a}_r \cdot \text{id}_2`, where :math:`\text{id}_2` is the unit matrix.

There are two types of operators that extend the base-class :class:`AbstractOperator`.
The first one are operators that are given as actual numbers at each grid point, i.e. the values are stored in an array and when applied on a wave function point-wise multiplied by it.
These are implemented in the classes :class:`dish.util.radial.operator.ScalarOperator` and :class:`dish.util.radial.operator.MatrixOperator`.

The more interesting kind are the symbolic operators which accept functions that take the wave function as an argument.
These are implemented as subclasses of :class:`dish.util.radial.operator.SymbolicScalarOperator` and :class:`dish.util.radial.operator.SymbolicMatrixOperator`.
This allows for very versatile usage like the implementation of the differential operator :math:`d / dr` using numeric differentiation:

.. code-block:: python

   from dish import DifferentialOperator
   DifferentialOperator() * wf

Also operators that are dependent of :math:`r` can be easily constructed using a :class:`dish.util.radial.operator.RadialOperator` which is a subclass of :class:`SymbolicScalarOperator`.
To calculate e.g. the expectation value :math:`\langle r \rangle` one can write:

.. code-block:: python

   # let wf_s be a RadialSchrodingerWaveFunction
   # and wf_d a RadialDiracWaveFunction
   from dish import RadialOperator

   BraOperator(wf_s) * RadialOperator(lambda r: r) * wf_s
   # the same syntax can be used for the relativistic wave functions
   # where actually the scalar operator is multiplied by a unity matrix
   BraOperator(wf_d) * RadialOperator(lambda r: r) * wf_d


Matrix versions can be constructed by passing scalar operators in a nested list:

.. code-block:: python

   from dish import SymbolicMatrixOperator

   op = SymbolicMatrixOperator([[RadialOperator(lambda r: r), 5],
                                 1j, RadialOperator(lambda r: r**2+50)]])
   # physically useless but here for demonstration purposes
   BraOperator(wf_d) * op * wf_d

Instances of :class:`RadialOperator` evaluate the function's argument :math:`r` on the grid of the wave function.
For better readability, it is useful to use anonymous functions using the ``lambda``-syntax, like shown in the example above, when constructing :class:`RadialOperator`-instances.



.. _label-NonRelativisticCalculations:

Non-relativistic Calculations
-----------------------------

For non-relativistic calculations Schrödinger's equation can be solved in the center of mass system to consider a finite mass of the nucleus and therefore nuclear recoil.
Construct a :class:`dish.util.atom.Nucleus` with a finite mass by passing the value to the parameter ``M`` as discussed above.

Since in the non-relativistic theory there is no coupling of orbital angular momentum and spin, the radial part of an electronic states just depends on the principal quantum number :math:`n` and the orbital angular momentum quantum number :math:`l`.
These can be specified either using directly a :class:`dish.util.atom.QuantumNumberSet`

.. code-block:: python

   from dish.util.atom import QuantumNumberSet

   state = QuantumNumberSet(n:int, l:int)

or by parsing the spectroscopic notation similar to the relativistic states discussed above:

.. code-block:: python

   from dish.util.atom import parse_atomic_term_symbol

   state = parse_atomic_term_symbol(state_repr:str)
   # e.g.
   parse_atomic_term_symbol("1s")
   parse_atomic_term_symbol("4d")
   parse_atomic_term_symbol("4[2]")

There is no difference between the grid for a relativistic and a non-relativistic wave function. Follow the instructions above.

Finally solving Schrödinger's equation is done via:

.. code-block:: python

   from dish.schrodinger.solver import solve

   result = solve(nucleus: dish.util.atom.Nucleus,
                  state: dish.util.atom.QuantumNumberSet,
                  r_grid: dish.util.radial.grid.grid.DistanceGrid = <optional:dict(r0=1e-6, h=1e-4)>,
                  potential_model: str = <optional:"Fermi">,
                  E_guess: float = <optional:"auto">,
                  order_AM: int = <optional:5>,
                  order_insch: int = <optional:7>,
                  max_number_of_iterations: int = <optional:20>
                 )

   E = result.energy  # actually E-c^2
   wf = result.wave_function  # RadialSchrodingerWaveFunction

In very close analogy to the relativistic wave functions the solutions of Schrödinger's equation which are of the form

.. math::

   \psi_{nlm}(r, \theta, \varphi) = \frac{1}{r}R_{nl}Y_{lm}(\theta, \varphi)

are stored in a :class:`dish.util.radial.wave_function.RadialSchrodingerWaveFunction`.
The radial component :math:`R` can be accessed as well as its derivative :math:`Q`:

.. code-block:: python

   wf.R        # radial function's values -> stored in a numpy array
   wf.Q        # dR/dr -> stored in an numpy array

The other features are the same as for the relativistic wave functions.

How to calculate matrix elements using non-relativistic wave functions is already covered in the :ref:`above section <label-MatrixElements>`.



Functions/Classes used in this introduction
-------------------------------------------

.. autofunction:: dish.dirac.solver.solve
   :no-index:

.. autofunction:: dish.schrodinger.solver.solve
   :no-index:

.. autofunction:: dish.util.atomic_units.convert_units
   :no-index:

.. autoclass:: dish.util.atom.Nucleus
   :members:
   :no-index:

.. automodule:: dish.util.potential
   :members:
   :no-index:

.. autoclass:: dish.util.atom.QuantumNumberSet
   :no-index:

.. autofunction:: dish.util.atom.parse_atomic_term_symbol
   :no-index:

.. autoclass:: dish.util.misc.SolvingResult
   :no-index:

.. autoclass:: dish.util.radial.grid.grid.DistanceGrid
   :no-index:
   :members:

.. autoclass:: dish.util.radial.grid.grid.RombergIntegrationGrid
   :no-index:
   :members:

.. autoclass:: dish.util.radial.wave_function.RadialDiracWaveFunction
   :no-index:
   :members:

.. autoclass:: dish.util.radial.wave_function.RadialSchrodingerWaveFunction
   :no-index:
   :members:


.. automodule:: dish.util.radial.operator
   :no-index:
   :members:



.. _Hartree atomic units: https://en.wikipedia.org/wiki/Hartree_atomic_units

.. _Romberg's method: https://en.wikipedia.org/wiki/Romberg's_method