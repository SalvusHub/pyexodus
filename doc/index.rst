pyexodus
========

Pure Python (still requires h5py and numpy) package to write exodus files. It
emulates the Python API of the official Python exodus wrapper but is much much
easier to get going. Additionally it manages to avoid the installation of two
very large C/C++ libraries (netCDF and exodus) that are potentially tricky to
install, and is also easy to install on Windows.

Additionally it is quite a bit faster (I guess due to more direct memory to
file paths), uses less memory, and supports Python 2.7, 3.4, 3.5, and 3.6.

Don't expect this to be complete anytime soon - we will add stuff as soon as we
need it.

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

Deviations from the official API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By and large ``pyexodus`` aims to be fully compatible with the official Python exodus API.
Deviations are listed here.

* It supports optional compression. This is quite a bit slower to write but can have a very
  big impact on file size. See the ``compression`` argument of :class:`pyexodus.exodus`.
* :meth:`pyexodus.exodus.put_elem_connectivity` has two additional optional
  arguments: ``shift_indices`` and ``chunk_size_in_mb``.


.. autoclass:: pyexodus.exodus
    :members:


.. note:: Acknowledgements

    * The offical Python API can be here: https://github.com/gsjaardema/seacas
    * Logo made by Roundicons from http://www.flaticon.com (Creative Commons BY 3.0 License)

