
Getting Started
***************

.. contents::
   :local:
   :depth: 3

----
           
Requirements
============

* `python >= 2.7.x <https://www.python.org/>`_

  * `mercurial <https://www.mercurial-scm.org/>`_
  * `numpy <http://www.numpy.org/>`_
  * `cython <http://cython.org/>`_
  * `h5py <http://www.h5py.org/>`_
  * `yt dev <https://bitbucket.org/yt_analysis/yt>`_

    * `sympy <http://www.sympy.org/en/index.html>`_
      
----
    
Installation
============

Python and friends
------------------

Since this is a python package, you must have python installed!
``CAESAR`` should be py2 and py3 compatible so feel free to pick
whichever you are most comforatable with.  If you are new to python
you might as well pick py3, as support for py2 will begin to diminish
over the next few years.

If you do not have a pre-existing python installation (other than your
system's native python, of which I do **not** recommend using) I
*strongly* encourage you to consider a pre-packaged distribution such
as `Anaconda <https://www.continuum.io/downloads>`_.  Anaconda for
instance, will install an isolated python environment in your home
directory giving you full access to install and change packages
without fear of screwing up your system's default python install.
Another advantage is that it comes with nearly everything you need to
get started working with python (numpy/scipy/matplotlib/etc).

If you are like me and want full control over what gets installed, a
minimalist option is `Miniconda
<http://conda.pydata.org/miniconda.html>`_.  This package gives you
the functionality of Anaconda without all of the default packages.


Anaconda / Miniconda
^^^^^^^^^^^^^^^^^^^^

Regardless of if you use Anaconda or Miniconda, all dependencies can
easily be installed via the ``conda`` command:

.. code-block:: bash

   $> conda install numpy cython h5py mercurial sympy

pip
^^^

If you do not elect to use the ``conda`` command, dependencies can
also be installed via `pip <https://pypi.python.org/pypi/pip>`_.  Be
aware that in order for `h5py` to properly compile you must first have
`HDF5 <https://www.hdfgroup.org/HDF5/>`_ correctly installed (via
`apt-get`, `brew`, or manual compilation) and in your respective
environment paths.

.. code-block:: bash

   $> pip install numpy cython h5py mercurial sympy

----
   
yt dev
------

The current release version of ``yt`` is 3.2.x; ``CAESAR`` relies on
methods currently residing in the development branch of ``yt`` not
slated to hit the stable branch until 3.3.  This means that ``yt``
must be installed from the development branch.  Luckily mercurial
makes this pretty easy!

First we need to clone the ``yt`` project:

.. code-block:: bash

   $> hg clone https://bitbucket.org/yt_analysis/yt

Now we change into the ``yt`` directory, and update to the ``yt`` branch:

.. code-block:: bash

   $> cd yt
   $> hg update yt

Finally, we build ``yt`` which may take a little while for everything to
compile:

.. code-block:: bash

   $> python setup.py install

----
   
CAESAR
------

Now that we have all of the prerequisites out of the way we can clone
and install ``CAESAR``:

.. code-block:: bash

   $> hg clone https://bitbucket.org/rthompson/caesar
   $> cd caesar
   $> python setup.py install

Once it finishes you should be ready to finally get some work done!
