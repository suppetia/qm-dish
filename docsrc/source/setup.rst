Setup
=====

dish requires CPython 3.7 or above but below 3.12 (since gympy2 is not working with Python3.12).
At the moment it is required to build the package and install it afterwards from the local sources.


Building the package
--------------------
On Unix-based systems
~~~~~~~~~~~~~~~~~~~~~
#. Make sure to have the following tools installed:

   * A working `Python`_ (version < 3.12) solution and the python virtual environment package `venv`_ (on Unix systems this needs to be installed separately).

   * The following Python packages `pip`_, `build`_

   * *(optionally)* For increased performance it is highly recommended to compile an included fortran script. For that the following is required

      #. The Fortran compiler `gfortran`_

      #. The Python Package `numpy`_

.. note::
    On Ubuntu the installation can be done via:

    .. code-block:: bash

        sudo apt install python3 python3-pip python3-venv python3-numpy gfortran
        python3 -m pip install build

2. Clone or download this repository.
3. In a shell in the repositories directory run ``make``.
   The Python package will be built in the *dist*-directory.

.. note::
    The build script will detect your installation of your python installation using ``which python3`` and therefore also works in virtual python environments when they are activated in your current shell.
    If you wish to use a specific python executable modify line 3 in the :doc:`Makefile <../../Makefile>`:

    .. code-block:: bash

        # old version
        #PY := $(shell which python3)
        # modified version
        PY := /path/to/your/executable


If you encounter any problems using `make` you can build it by hand by running the following commands in the main directory:

.. code-block:: bash

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


.. note:: This assumes you have a working Python environment (with Python < 3.12 (!)) with numpy installed.



On a Windows system
~~~~~~~~~~~~~~~~~~~

1. Make sure to have the following tools installed:

   1. A working `Python`_ solution.
   2. The following Python packages `pip`_, `build`_, `meson`_
2. Clone or download this repository.
3. Change into the repositories directory and run

   .. code-block:: bash

       python -m build --outdir dist

   The Python package will be built in the *dist*-directory.


.. note:: The package is named `qm-dish` for installation to be distinguished from another package called dish which appears to be a shell implementation for Unix-systems.


.. _Python: https://www.python.org/
.. _venv: https://docs.python.org/3/library/venv.html
.. _pip: https://pypi.org/project/pip/
.. _build: https://pypi.org/project/build/
.. _gfortran: https://gcc.gnu.org/fortran/
.. _numpy: https://numpy.org/
.. _meson: https://pypi.org/project/meson/



Installing the package
----------------------

On **Unix** based systems after building the package run in the same directory

.. code-block:: bash

    make install

.. note::
    This will install the package using pip.
    An automatic installation using conda is currently not supported but building the package also provides the *sdist* from which a conda version can be build.

On a **Windows** system (*or if you encounter any problems*) run from the same directory

.. code-block:: bash

    python -m pip install qm-dish --find-links dist/

The freshly built package and the required dependencies will be installed.

