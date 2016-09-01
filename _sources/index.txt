pyexodus
========

Pure Python (still requires h5py and numpy) package to write exodus files. It
emulates the Python API of the official Python exodus wrapper but is much much
easier to get going. Additionally it manages to avoid the installation of two
very large C/C++ libraries (netCDF and exodus) that are potentially tricky to
install, and is also easy to install on Windows.

Don't expect this to be complete anytime soon - we will add stuff as soon as we
need it. Currently it does exactly what we need it to do and nothing else. As
of now it can write some kinds of exodus files that can then be opened by
Paraview.

Installation
------------

.. code-block:: bash

    $ conda install -c conda-forge numpy h5netcdf pip pytest
    $ git clone https://github.com/SalvusHub/pyexodus.git
    $ cd pyexodus
    $ pip install -v -e .
    # Run the tests to make sure everything works.
    $ py.test


API
---

.. autoclass:: pyexodus.exodus
    :members:


.. note:: Acknowledgements

    * The offical Python API can be here: https://github.com/gsjaardema/seacas
    * Logo made by Roundicons from http://www.flaticon.com (Creative Commons BY 3.0 License)

