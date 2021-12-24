
Getting Started
***************

.. contents::
   :local:
   :depth: 3

----
           
Requirements
============

* `python >= 3.x <https://www.python.org/>`_

  * `numpy <http://www.numpy.org/>`_
  * `scipy <https://www.scipy.org/>`_
  * `cython <http://cython.org/>`_
  * `h5py <http://www.h5py.org/>`_
  * `matplotlib <http://www.matplotlib.org/>`_
  * `psutil <https://pypi.org/project/psutil/>`_
  * `joblib <https://joblib.readthedocs.io/>`_
  * `six <https://six.readthedocs.io/>`_
  * `astropy <https://www.astropy.org/>`_
  * `yt >= 4.0 <https://bitbucket.org/yt_analysis/yt>`_

----
    
Installation
============

Python and friends
------------------

Since this is a python package, you must have python installed!
``CAESAR`` formally requires python-3.  Some basic functionality is
still compatible with python-2, but we have discontinued further
support for this in ``CAESAR``.

We strongly encourage using a pre-packaged python distribution,
such as `Anaconda <https://www.anaconda.com/products/individual>`_.
This will install an isolated python environment in your home
directory giving you full access to install and change packages
without fear of screwing up your system's default python install.
Another advantage is that it comes with nearly everything you need
to get started working with python (numpy/scipy/matplotlib/etc).

----


Dependencies
------------

Installing the main dependencies is very easy under Anaconda, or using the
python package manager `pip <https://pypi.python.org/pypi/pip>`_.

.. code-block:: bash

   $> conda install numpy scipy cython h5py matplotlib psutil joblib six astropy

Alternatively, if you do not wish to use Anaconda, these can all be installed
under ``pip`` by replacing ``conda`` with ``pip`` in the line above.  Some
of these automatically come with Anaconda, but the above command
will update these to the latest version if needed.

Be aware that in order for ``h5py`` to properly compile you must first have
`HDF5 <https://www.hdfgroup.org/HDF5/>`_ correctly installed (via
e.g. `apt-get`, `brew`, or manual compilation) and in your respective
environment paths.

The optional galaxy/halo photometry computation in ``CAESAR``
requires `python-fsps <http://dfm.io/python-fsps/current/>`_, which
is a python wrapper for the ``FSPS`` fortran package.  Please follow
their `installation instructions
<http://dfm.io/python-fsps/current/installation/>`_ to install this.
Furthermore, you will also need two other packages that are only
available via ``pip``:

.. code-block:: bash

   $> pip install synphot extinction

If you wish to use the MPI driver to run single instances of Caesar over
many cores via MPI, it is also necessary to install ``mpi4py``:

.. code-block:: bash

   $> conda install mpi4py

Note that ``CAESAR`` is natively OpenMP-parallel, and the MPI implementation
may be system-specific. 

If you wish to work with galaxy and halo particle lists (for instance to compute
your own quantities) it is highly recommended that you install ``pygadgetreader``:

.. code-block:: bash

   $> git clone https://github.com/dnarayanan/pygadgetreader.git
   $> cd pygadgetreader
   $> python setup.py install


----

yt
--

``CAESAR`` builds on the `yt <https://yt-project.org/>`_ simulation analysis toolkit.
``CAESAR`` requires yt version >=4.0, though a lot of functionality will still work with yt-3.6+.

We recommend installing ``yt`` via Anaconda:

.. code-block:: bash

   $> conda install -c conda-forge yt

but other installation options are `described here <https://yt-project.org/#getyt>`_.

If you already have ``yt``, you can check your version using ``yt version``, and
`update <http://yt-project.org/doc/installing.html#updating-yt-and-its-dependencies>`_
if necessary.

----

CAESAR
------

Now that we have all of the prerequisites out of the way we can clone
and install ``CAESAR``:

.. code-block:: bash

   $> git clone https://github.com/dnarayanan/caesar.git
   $> cd caesar
   $> python setup.py install

Once it finishes you should be ready to finally get some work done!

----

Updating
========

To update ``CAESAR`` simply pull the changes and reinstall:

.. code-block:: bash

   $> cd caesar
   $> git pull
   $> python setup.py install


