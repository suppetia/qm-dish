# DISH
A Lightweight **DI**rac **S**olver for **H**ydrogen-like Systems

## Setup

On Unix-based systems:  
1. Make sure to have the following tools installed:
   1. A working [Python](https://www.python.org/) solution.
   2. The following Python packages [pip](https://pypi.org/project/pip/), [build](https://pypi.org/project/build/)
   3. *(optionally)* For increased performance it is highly recommended to compile an included fortran script. For that the following is required
      1. The Fortran compiler [gfortran](https://gcc.gnu.org/fortran/)
      2. The Python Package [numpy](https://numpy.org/)
   > On Ubuntu the installation can be done via:
   > ```bash
   > sudo apt install python3 python3-pip python3-numpy gfortran
   > python3 -m pip install build
   > ```
2. Clone or download this repository.
3. In the repositories directory run `make`. The Python package will be built in the *dist*-directory.

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
