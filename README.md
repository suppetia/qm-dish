# QM-DISH
A Lightweight **DI**rac **S**olver for **H**ydrogen-like Systems

## What is *dish*?
*dish* is a Python package which allows the simple calculation of wavefunctions and energy levels for Hydrogen-like systems.
The calculation is done by default in a relativistic context by solving the Dirac equation for a single electron in a spherical symmetric potential.  
A subpackage also allows to retrieve the non-relativistic wavefunctions and energy-levels by solving the Schrödinger equation.

## Setup

### Building the package
On Unix-based systems:  
1. Make sure to have the following tools installed:
   1. A working [Python](https://www.python.org/) (version < 3.12) solution and the python virtual environment package [venv](https://docs.python.org/3/library/venv.html) (on Unix systems this needs to be installed separately).
   2. The following Python packages [pip](https://pypi.org/project/pip/), [build](https://pypi.org/project/build/)
   3. *(optionally)* For increased performance it is highly recommended to compile an included fortran script. For that the following is required
      1. The Fortran compiler [gfortran](https://gcc.gnu.org/fortran/)
      2. The Python Package [numpy](https://numpy.org/)
   > On Ubuntu the installation can be done via:
   > ```bash
   > sudo apt install python3 python3-pip python3-venv python3-numpy gfortran
   > python3 -m pip install build
   > ```
2. Clone or download this repository.
3. In a shell in the repositories directory run `make`. The Python package will be built in the *dist*-directory.
   > The build script will detect your installation of your python installation using `which python3` and therefore also works in virtual python environments when they are activated in your current shell.
   > If you wish to use a specific python executable modify line 3 in the [Makefile](Makefile):
   > ```Makefile
   > # old version
   > #PY := $(shell which python3)
   > # modified version
   > PY := /path/to/your/executable
   > ```

If you encounter any problems using `make` you can build it by hand by running the following commands in the main directory:
> This assumes you have a working Python environment (with Python < 3.12 (!)) with numpy installed.
```bash
mkdir dist
cp -r src/ dist/
cp LICENSE MANIFEST.in pyproject.toml README.md dist/
cd dist/src/dish/util/numeric
python3 -m numpy.f2py -c adams_f.f90 -m adams_f -llapack
# if this fails to identify your fortran compiler use
#  --f90exec=/path/to/gfortran --f77exec=/path/to/gfortran
# as additional arguments

cd ../../../../..

python3 -m build dist --outdir dist
```


On a Windows system:
1. Make sure to have the following tools installed:
   1. A working [Python](https://www.python.org/) solution.
   2. The following Python packages [pip](https://pypi.org/project/pip/), [build](https://pypi.org/project/build/)
2. Clone or download this repository.
3. Change into the repositories directory and run 
   ```bash
   python -m build --outdir dist
   ``` 
   The Python package will be built in the *dist*-directory.


> The package is named `qm-dish` for installation to be distinguished from another package called dish which appears to be a shell implementation for Unix-systems.

### Installing the package
On **Unix** based systems after building the package run in the same directory
```bash
make install
```
> This will install the package using pip. 
> An installation using conda is currently not supported but building the package also provides the *sdist* from which a conda version can be build.

On a **Windows** system (*or if you encounter any problems*) run from the same directory
```commandline
python -m pip install qm-dish --find-links dist/
```
The freshly built package and the required dependencies will be installed.


## Quickstart

> **Note**: All calculations internally are performed dimensionless using [Hartree atomic units](https://en.wikipedia.org/wiki/Hartree_atomic_units).
> All classes and functions therefore expect values in atomic units.   
> To convert between units of the SI system and atomic units *dish* provides the method `convert_units`:
> ```python
> convert_units(old_unit, new_unit, value=1., old_unit_exp=1, new_unit_exp=1)
> ```
> which returns the converted value of *value* (given in *old_unit*) in *new_unit*.  
> (Simplified this returns `value * new_unit / old_unit`).  
> ! You need to assure that *old_unit* and *new_unit* have the same dimension as there are no checks performed!  
> *old_unit* and *new_unit* can be either numerical values or strings for the most common units (e.g. "E_h", "eV", "J").


To calculate electronic states of a Hydrogen-like system first the properties of the nucleus need to be specified.  
*dish* has build-in support for either a point-like nucleus (a pure Coulomb potential), a homogeneously charged ball-like nucleus or a nucleus which charge is described by a Fermi distribution:
$`\rho(r) = \rho_0/(1+\exp((r-c)/a))`$  
Therefore, to following properties need to be specified:
1. The nuclear charge *Ze* passed to the parameter `Z`.
2. The mass of the nucleus `M`. To perform calculations with a fixed core this can be set to *numpy.inf* .
3. The radius of the charge distribution `R0`. 
   1. For a point-like model this is irrelevant and can be set to 0.
   2. For a ball-like model the radius *r* of the sphere is $`r = R0`$ 
   3. For a Fermi-charge-distribution $`c = R0`$
4. (Optional and only necessary for a Fermi model) The _diffuseness_ parameter `a`. 
   > *a* defaults to $2.3 \text{fm} /a_0 / (4\cdot\ln(3))$ as described in *Parpia and Mohanty, Phys.Rev.A, 46 (1992), Number 7*

An example for Ca19+ looks like:

```python
from dish import Nucleus, convert_units

nuc = Nucleus(Z=20,
              R0=convert_units("m", "a_0", 3.4776e-15),
              M=convert_units("u", "m_e", 40.078)
              )
```

Then the electronic state which should be calculated should be specified.
This can be either done be passing the quantum numbers a tuple `(n, l, j)` (e.g. *(2, 0, 1/2)*) 
or by passing the atomic term symbol as a string `"<n><l-alias><j>"` (e.g. *"2s1/2"*).  
For the latter one there is another possible notation: `"2s1/2 == 2[0]+` where the name for the angular-momentum number can be substituted by the value in brackets
and as only single electron systems are calculated *j* will always be $l\pm1/2$ which can be passed by a *+* or a *-*.
```python
state = "3p+"
```

After specifying this properties one can run the solving algorithm. In a minimal setup the function call looks like:
```python
from dish import solve
result = solve(nucleus=nuc, state=state)
```
By default, for the nuclear potential the Fermi model is used.  
To change this pass the alias of the potential model a string to the *potential_model* parameter:
```python
result = solve(nucleus=nuc,
               state=state,
               potential_model="Fermi"  # other options are "ball"=="uniform" or "pointlike"=="coulomb"
               )
```

After the solving process the wavefunction can be obtained from the *SolvingResult* object which is stored in *result* via
```python
Psi = result.wave_function
```
The wavefunction is stored in a *RadialDiracWaveFunction* object which stores the array of points where the wavefunction is evaluated in the field *r*, the large component in the field *f* and the small component in the field *g*:
```python
Psi.r  # grid points -> stored in a numpy array
Psi.f  # small component values at the grid points -> stored in a numpy array
Psi.g  # large component values at the grid points -> stored in a numpy array
```

The energy of the *state* is stored in the field *energy*:
```python
result.energy
```

### Configuring the solving parameters
The solving algorithm uses a finite-differences approach to solve the coupled system of equations for the large and small component.
The energy is searched for which both components can be found continuous. 
To numerically solve the Dirac equation an [Adams-Moulton method](https://en.wikipedia.org/wiki/Linear_multistep_method#Adams%E2%80%93Moulton_methods) is used.

The parameters of the solving algorithm are chosen based on the state by default and can be adjusted by the user: 
#### The Grid
The radial grid on which the wave function is evaluated on is an exponential grid of the form
$`r(n) = r_0 \cdot (\exp(h\cdot n)-1)`$.  
This information is stored in a *DistanceGrid* object which can be constructed from the parameters *r0*, *h* and *N* which is the maximum number of *n* or $r_{max}$ which defines the region where the maximum *r* value is evaluated.
It is recommended to regulate *N* using the other 3 parameters and not passing it explicitly.  
It can be passed to the solve function via the parameter *r_grid*:
```python
from dish import DistanceGrid
result = solve(nucleus=nuc,
               state=state,
               r_grid=DistanceGrid(h=0.005, r0=1e-6, r_max=2)
               # another possibility is to pass this using a dictionary which will be parsed internally:
               # r_grid={"h": 0.005, "r0"=1e-6, "r_max"=2}
               # passing N instead of r_max is also possible but more inconvenient
               # r_grid=DistanceGrid(h=0.005, r0=1e-6, N=20000)
               )
```
> **Note**: The more *N* points the grid contains and therefore the smaller *r0* and *h* are chosen the more computation intensive is the solving process.
> For *N* <= $10^5$ the Python version provides enough speed that the solving finishes in under a second of computation time on decent hardware but for larger *N* it is highly recommended to use the Fortran version which speeds up the main part of the solving process by one to two orders of magnitude.

#### The Adams-Moulton method
Adams-Moulton (AM) methods are a linear multistep method to solve systems of ordinary differential equations numerically on a finite grid.
A multistep method of order *k* uses the information if the previous *k* points to determine the next point (for a detailed discussion see Chapter (???) of the underlying thesis).
Therefore, the AM method requires *k* initial values. In the algorithm the AM method is used from the most inner points in the outward direction and inwards from the most outer points.  
The initial values are calculated from asymptotic considerations. The order of the method to retrieve the initial values for the inward integration can be specified by the parameter *order_indir*.
The order of the AM itself can be set via *order_AM*.  
```python
result = solve(nucleus=nuc,
               state=state,
               order_AM=9,    # default value that has proven to be reliable
               order_indir=7  # default value
               )
```
Usually there is no need to modify this values but for some wavefunctions that oscillate quickly near the origin it might help to increase the order of the AM method.
> **Note**: The higher *order_AM* the more compute intensive is the solving process.

