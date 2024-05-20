Examples
========

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