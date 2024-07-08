Quickstart
==========

All calculations internally are performed dimensionless using `Hartree atomic units`_.
All classes and functions therefore expect values in atomic units.
To convert between units of the SI system and atomic units *dish* provides the method :func:`convert_units <dish.util.atomic_units.convert_units>`.

It returns the converted value of *value* (given in *old_unit*) in *new_unit*.
(Simplified this returns ``value * new_unit / old_unit``).
! You need to assure that *old_unit* and *new_unit* have the same dimension as there are no checks performed!
*old_unit* and *new_unit* can be either numerical values or strings for the most common units (e.g. "E_h", "eV", "J").


To calculate electronic states of a Hydrogen-like system first the properties of the :class:`Nucleus <dish.util.atom.Nucleus>` need to be specified.
*dish* has build-in support for either a point-like nucleus (a pure Coulomb potential), a homogeneously charged ball-like nucleus or a nucleus which charge is described by a Fermi distribution:

.. math:: \rho(r) = \frac{\rho_0}{1+\exp((r-c)/a)}

Therefore, to following properties need to be specified:

1. The nuclear charge *Ze* passed to the parameter ``Z``.
2. The mass of the nucleus ``M``. To perform calculations with a fixed core this can be set to *numpy.inf* .
3. Parameters of the charge distribution.
   The radius can be either given as the root mean squared radius :math:`R_\text{rms} \sqrt{\langle r^2 \rangle}` via the parameter ``R_rms``,
   for a ball-like model as the radius :math:`R_0` via the parameter ``R0``
   or for a Fermi charge density distribution as the parameter :math:`c` via the constructor parameter of the same name.

   If a value is passed to one of the three parameters the others are calculated if possible.
   For a point-like model this must not be passed.

   For a Fermi model also the *diffuseness* parameter :math:`a` is required.
   It can be passed explicitly to the constructor via the parameter ``a`` but defaults
   to :math:`2.3 \text{fm} /a_0 / (4\cdot\ln(3))` as this is a good approximation for most stable nuclei (*Parpia and Mohanty, Phys.Rev.A, 46 (1992), Number 7*).

   .. note:: For small nuclei the Fermi-parameter :math:`c` can not be derived from :math:`R_\text{rms}` as the model
      can't be applied with the default value of :math:`a`.


An example for Ca19+ looks like:

.. code-block:: python

    from dish import Nucleus, convert_units

    nuc = Nucleus(Z=20,
                  R_rms=convert_units("fm", "a_0", 3.4776),
                  M=convert_units("u", "m_e", 40.078)
                  )



Then the electronic state which should be calculated should be specified.
This can be either done be passing the quantum numbers a tuple ``(n, l, j)`` (e.g. *(2, 0, 1/2)*)
or by passing the atomic term symbol as a string ``"<n><l-alias><j>"`` (e.g. *"2s1/2"*).
For the latter one there is another possible notation: ``<n>[<l>]<+/->`` (e.g. *"2s1/2 == 2[0]+*) where the name for the angular-momentum number can be substituted by the value in brackets
and as only single electron systems are calculated *j* will always be :math:`l\pm1/2` which can be passed by a *+* or a *-*.
A mixture of the latter two variants is possible.

From the string a :class:`QuantumNumberSet <dish.util.atom.QuantumNumberSet>` object will be constructed
using the :func:`parse_atomic_term_symbol <dish.util.atom.parse_atomic_term_symbol>` function.
You can also pass a :class:`QuantumNumberSet <dish.util.atom.QuantumNumberSet>` object directly.

For example the :math:`3p_{1/2}` state can be specified the following way:

.. code-block:: python

   state = "3p-"
   # which is equivalent to
   state = "3[2]1/2


After specifying this properties one can run the solving algorithm. In a minimal setup the function call to :func:`solve <dish.dirac.solver.solve>` looks like:

.. code-block:: python

   from dish import solve
   result = solve(nucleus=nuc, state=state)

By default, for the nuclear potential the Fermi model is used.
To change this pass the alias of the potential model a string to the ``potential_model`` parameter:

.. code-block:: python

   result = solve(nucleus=nuc,
                  state=state,
                  potential_model="Fermi"  # other options are "ball"=="uniform" or "pointlike"=="coulomb"
                  )


After the solving process the energy and wave function of the ``state`` can be obtained from the :class:`SolvingResult <dish.util.misc.SolvingResult>` object which is stored in *result* via

.. code-block:: python

   E = result.energy

   Psi = result.wave_function

The wave function
-----------------

has the form

.. math::

   \psi_{n \kappa m}(\textbf{r})  = \frac{1}{r} \begin{pmatrix} i f_{n\kappa} (r) \Omega_{\kappa, m}(\theta, \varphi) \\ g_{n\kappa}(r) \Omega_{-\kappa,m}(\theta, \varphi) \end{pmatrix}

where :math:`(r, \theta, \varphi)` are spherical coordinates and :math:`\Omega_{\kappa, m}` is a spherical spinor.
It is stored in a :py:class:`RadialDiracWaveFunction <dish.util.radial.wave_function.RadialDiracWaveFunction>` object which stores the array of points where the wave function is evaluated in the field *r*, the large component :math:`f_\kappa` in the field *f* and the small component :math:`g_\kappa` in the field *g*:

.. code-block:: python

   Psi.r  # grid points -> stored in a numpy array
   Psi.f  # large component values at the grid points -> stored in a numpy array
   Psi.g  # small component values at the grid points -> stored in a numpy array

.. note::

   To allow further calculations (mainly the calculation of [matrix elements](#calculating-matrix-elements))
   :math:`f` and :math:`g` are stored like above (without the factor :math:`\frac{1}{r}`) to minimize loss of accuracy due to unnecessary rounding.



Configuring the solving parameters
----------------------------------

The solving algorithm uses a finite-differences approach to solve the coupled system of equations for the large and small component.
The energy is searched for which both components can be found continuous.
To numerically solve the Dirac equation an `Adams-Moulton method`_ is used.

The parameters of the solving algorithm are chosen based on the state by default and can be adjusted by the user:

The Grid
~~~~~~~~

The radial grid on which the wave function is evaluated on is an exponential grid of the form

.. math::

   r(t) = r_0 \cdot (\exp(h\cdot t)-1)

where :math:`t` is a linear grid from :math:`0` to :math:`N`.
This information is stored in a :class:`DistanceGrid <dish.util.radial.grid.DistanceGrid>` object which can be constructed from the parameters ``r0``, ``h`` and ``N`` or ``r_max`` which defines the region where the maximum :math:`r` value is evaluated.
It is recommended to regulate :math:`N` using the other three parameters and not passing it explicitly.
The grid can be passed to the solve function via the parameter ``r_grid``:

.. code-block:: python

   from dish import DistanceGrid
   result = solve(nucleus=nuc,
                  state=state,
                  r_grid=DistanceGrid(h=0.005, r0=1e-6, r_max=2)
                  # another possibility is to pass this using a dictionary which will be parsed internally:
                  # r_grid={"h": 0.005, "r0"=1e-6, "r_max"=2}
                  # passing N instead of r_max is also possible but more inconvenient
                  # r_grid=DistanceGrid(h=0.005, r0=1e-6, N=20000)
                  )

.. note::

   | The more points the grid contains and therefore the smaller ``r0`` and ``h`` are chosen the more computation intensive is the solving process.
   | For :math:`N \le 10^5` the Python version provides enough speed that the solving finishes in under a second of computation time on decent hardware but for larger ``N`` it is highly recommended to use the Fortran version which speeds up the main part of the solving process by one to two orders of magnitude.

A better grid for integration purposes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To calculate matrix elements using the wave functions one needs to solve a radial integral.
Since the grid on which the wave function is evaluated is given due to the method how the wave function is found,
one can either integrate on the exponential grid :math:`r` or on the linear grid :math:`t` where the latter one is better
suited as the numerical methods to integrate on equidistant points are well understood.
On equidistant grids Newton-Cotes-formulas are used, where the lowest order method, the so-called trapezoid rule,
is the most efficient but also the most inaccurate.

If possible use a :class:`RombergIntegrationGrid <dish.util.radial.grid.RombergIntegrationGrid>` instead of a :class:`DistanceGrid <dish.util.radial.grid.DistanceGrid>` since it enables to use Romberg's method
for approximating the integral which is an improvement of the trapezoidal rule using Richardson extrapolation.
But it requires an equidistant grid with a number of points :math:`N = 2^k + 1` where :math:`k` is a positive integer.

One can either just construct a :class:`RombergIntegrationGrid <dish.util.radial.grid.RombergIntegrationGrid>` (which is a subclass of :class:`DistanceGrid <dish.util.radial.grid.DistanceGrid>`) directly

.. code-block:: python

   from dish.util.radial.grid import RombergIntegrationGrid

   r_grid = RombergIntegrationGrid(h=1e-5,
                                   r0=1e-8,
                                   k=15,
                                   # or N can be passed but needs to fulfill the requirement N = 2^k+1
                                   # N=2**15+1
                                   )

or a similar grid can be derived from a given :class:`DistanceGrid <dish.util.radial.grid.DistanceGrid>` *grid*

.. code-block:: python

   r_grid = RombergIntegrationGrid.construct_similar_grid_from_distance_grid(grid)

where the ``h`` parameter is altered so that ``r_max`` and ``r0`` are kept constant but there is the right number of points for Romberg integration.
Note that this method will always create a grid where ``h`` is smaller (or equal) than in the original grid so that there are more or the same number of points.


The Adams-Moulton method
~~~~~~~~~~~~~~~~~~~~~~~~

| Adams-Moulton (AM) methods are a linear multistep method to solve systems of ordinary differential equations numerically on a finite grid.
| A multistep method of order :math:`k` uses the information if the previous :math:`k` points to determine the next point (for a detailed discussion see chapter 3.3 of the underlying thesis). Therefore, the AM method requires :math:`k` initial values.
| In the algorithm the AM method is used from the most inner points in the outward direction and inwards from the most outer points.
| The initial values are calculated from asymptotic considerations. The order of the method to retrieve the initial values for the inward integration can be specified by the parameter ``order_indir``.
| The order of the AM itself can be set via ``order_AM``.

.. code-block:: python

   result = solve(nucleus=nuc,
                  state=state,
                  order_AM=9,    # default value that has proven to be reliable
                  order_indir=7  # default value
                  )

Usually there is no need to modify this values but for some wave functions that oscillate quickly near the origin it might help to increase the order of the AM method.

.. note::

   The higher ``order_AM`` the more compute intensive is the solving process.

The ``indir`` function
~~~~~~~~~~~~~~~~~~~~~~

To use a :math:`k^\text{th}` order multistep method :math:`k` initial values are required
in each direction. The initial guesses for the inward direction are obtained from asymptotic assumptions
and the wave function is expanded in a converging series (see chapter 3 of the thesis).
This series converges very fast if :math:`r` is large enough in the most outward region and hence
only the first few terms need to be calculated.
The number of terms can be passed via the parameter ``order_indir``.
The default value of 7 should be sufficient in most cases.

Functions/Classes used in this Quickstart
-----------------------------------------

.. autofunction:: dish.dirac.solver.solve
   :no-index:

.. autofunction:: dish.util.atomic_units.convert_units
   :no-index:

.. autoclass:: dish.util.atom.Nucleus
   :members:
   :no-index:


.. autoclass:: dish.util.atom.QuantumNumberSet
   :no-index:

.. autofunction:: dish.util.atom.parse_atomic_term_symbol
   :no-index:

.. autoclass:: dish.util.misc.SolvingResult
   :no-index:

.. autoclass:: dish.util.radial.grid.DistanceGrid
   :no-index:

.. autoclass:: dish.util.radial.grid.RombergIntegrationGrid
   :no-index:



.. _Hartree atomic units: https://en.wikipedia.org/wiki/Hartree_atomic_units
.. _Adams-Moulton method: https://en.wikipedia.org/wiki/Linear_multistep_method#Adams%E2%80%93Moulton_methods
