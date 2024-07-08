Examples
========

Simple Plotting of wave functions
---------------------------------
Calculating two states and plotting both radial components using *matplotlib*.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   from dish.util.atom import Nucleus, parse_atomic_term_symbol
   from dish.util.atomic_units import convert_units
   from dish.util.radial.grid.grid import DistanceGrid
   from dish.dirac.solver import solve

   fig1, ax = plt.subplots(nrows=2, figsize=(6,4), sharex=True)

   nuc = Nucleus(Z=1,
               R0=convert_units("m", "a_0", .8783e-15),
               M=np.inf,
               )
   grid = DistanceGrid(r0=1e-3, h=1e-3, r_max=150)
   res1 = solve(nucleus=nuc, state=parse_atomic_term_symbol("5d-"), r_grid=grid)
   ax[0].plot(grid.r, res1.wave_function.f, label="$5d-$")
   ax[1].plot(grid.r, res1.wave_function.g)

   res2 = solve(nucleus=nuc, state=parse_atomic_term_symbol("3p-"), r_grid=grid)
   ax[0].plot(grid.r, res2.wave_function.f, label="$3p-$")
   ax[1].plot(grid.r, res2.wave_function.g)

   ax[1].set_xlabel("$r$ (in a.u.)")
   ax[0].set_ylabel("$f(r)$")
   ax[1].set_ylabel("$g(r)$")

   ax[0].legend()


Operator Interface
------------------

Some minor examples
~~~~~~~~~~~~~~~~~~~

Using the low-level interface to calculate

.. math::

   \langle n_1 \kappa_1\mid \gamma_5 \mid n_2\kappa_2 \rangle = i \int_0^\infty (-f_1 g_2 + g_1 f_2) d r \,:


.. code-block:: python

   wf1 = solve(...).wave_function
   wf2 = solve(...).wave_function

   # assure wf1 and wf2 are evaluated on the same grid
   integrate_on_grid(-wf1.f*wf2.g + wf1.g * wf2.f, grid=wf1.grid) * 1j

Doing the same using the high-level interface:

.. code-block:: python

   from dish.util.radial.operator import BraOperator, SymbolicMatrixOperator

   y_5 = SymbolicMatrixOperator([[0,1], [1,0]])

   BraOperator(wf1) * y_5 * wf2


A simple :class:`RadialOperator`:

.. code-block:: python

   from dish.util.radial.operator import BraOperator, RadialOperator

   r_hat = RadialOperator(lambda r: r)

   BraOperator(wf1) * r_hat * wf2

Instances of :math:`SymbolicScalarOperator` and :math:`SymbolicMatrixOperator` can be chained:

.. code-block:: python

   from dish.util.radial.operator import BraOperator, RadialOperator, SymbolicMatrixOperator

   r_hat = RadialOperator(lambda r: r)

   O = SymbolicMatrixOperator([[1, 1+r_hat], [-5*r_hat, r_hat*r_hat]])

   BraOperator(wf1) * O * wf2
   BraOperator(wf1) * (O + O) * wf2
   BraOperator(wf1) * O * O * wf2


A Full Example
~~~~~~~~~~~~~~

Calculating the energy expectation value for the :math:`1\text{s}_{1/2}` state of hydrogen.

.. code-block:: python

   from dish import (
       Nucleus,
       DistanceGrid,
       RombergIntegrationGrid,
       parse_atomic_term_symbol,
       convert_units,
       solve
   )
   from dish.util.radial.operator import (
       BraOperator,
       SymbolicMatrixOperator,
       DifferentialOperator
   )
   from dish.util.radial.operator import RadialOperator as RO
   from dish.util.atomic_units import c

   import numpy as np

   # define the hydrogenic system
   nuc = Nucleus(Z=1,
                 c=convert_units("m", "a_0", .69975e-15),
                 a=convert_units("m", "a_0", 1e-15)/(4*np.log(3))
                 )

   r_grid = DistanceGrid(r0=1e-6, h=1e-3, r_max=250)
   r_grid = RombergIntegrationGrid.construct_similar_grid_from_distance_grid(r_grid)

   # calculate the wave functions
   state_a = parse_atomic_term_symbol("1s1/2")
   r_a = solve(nucleus=nuc, state=state_a, r_grid=r_grid,
               potential_model="Fermi")
   a = r_a.wave_function

   # implement H_D
   H_D = SymbolicMatrixOperator([
        [RO(lambda r: nuc.potential(r, "f") + c**2),
         RO(c) * DifferentialOperator() - RO(lambda r: c*state_a.kappa/r)
         ],
        [RO(-c) * DifferentialOperator() + RO(lambda r: c*state_a.kappa/r),
         RO(lambda r: nuc.potential(r, "f") - c**2)]
       ])

   # calculate the energy expectation value <a|H_D|a>
   E = BraOperator(a) * H_D * a


.. _label-examplesYukawa:

Implementation of a Custom Potential
------------------------------------

Here an example implementation of a Yukawa potential

.. math::

   V_\text{Yukawa}(r) = -g^2 \frac{\mathrm{e}^{-mr}}{r}

is given.

.. code-block:: python

   from dish.util.atom import Nucleus
   import numpy as np

   class YukawaNucleus(Nucleus):

       def __init__(self, Z, g, m):
           self.g = g
           self.m = m

           # note that the following line is required
           # as this is a subclass of Nucleus
           # it passes the nuclear charge Z which is required for asymptotics
           super().__init__(Z)

       def potential(self, r, model):
           if model.lower() in ["yukawa", "y"]:
               return -self.g**2 * np.exp(-self.m * r) / r
           else:
               # this case can be omitted
               # it enables correct error handling and calling the default potentials
               return super().potential(r, model)
