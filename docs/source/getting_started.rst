
Getting Started
***************

.. contents::
   :local:
   :depth: 3

----
           
Requirements
============

* `python >= 3.x <https://www.python.org/>`_

  * `numpy >= 1.13.x <http://www.numpy.org/>`_
  * `scipy <https://www.scipy.org/>`_
  * `cython <http://cython.org/>`_
  * `h5py <http://www.h5py.org/>`_
  * `yt >= 3.3 <https://bitbucket.org/yt_analysis/yt>`_

    * `sympy <http://www.sympy.org/en/index.html>`_
      
----
    
Installation
============

Python and friends
------------------

Since this is a python package, you must have python installed!
``CAESAR`` should be py2 and py3 compatible so feel free to pick
whichever you are most comforatable with.  This said, the developers
do not support issues related to py2 anymore.

If you do not have a pre-existing python installation (other than your
system's native python, of which we do **not** recommend using) we
*strongly* encourage you to consider a pre-packaged distribution such
as `Anaconda <https://www.continuum.io/downloads>`_.  Anaconda for
instance, will install an isolated python environment in your home
directory giving you full access to install and change packages
without fear of screwing up your system's default python install.
Another advantage is that it comes with nearly everything you need to
get started working with python (numpy/scipy/matplotlib/etc).

A more
minimalist option is `Miniconda
<http://conda.pydata.org/miniconda.html>`_.  This package gives you
the functionality of Anaconda without all of the default packages.


Anaconda / Miniconda
^^^^^^^^^^^^^^^^^^^^

Regardless of if you use Anaconda or Miniconda, all dependencies can
easily be installed via the ``conda`` command:

.. code-block:: bash

   $> conda install numpy scipy cython h5py mercurial sympy

pip
^^^

If you do not elect to use the ``conda`` command, dependencies can
also be installed via `pip <https://pypi.python.org/pypi/pip>`_.  Be
aware that in order for `h5py` to properly compile you must first have
`HDF5 <https://www.hdfgroup.org/HDF5/>`_ correctly installed (via
`apt-get`, `brew`, or manual compilation) and in your respective
environment paths.

.. code-block:: bash

   $> pip install numpy scipy cython h5py mercurial sympy

----
   
yt
--

``CAESAR`` relies on yt version 3.3 or above.  If you currently 
have ``yt`` installed and its version number is less than 3.3 you
must 
`update <http://yt-project.org/doc/installing.html#updating-yt-and-its-dependencies>`_
before ``CAESAR`` will function.  You can check your current ``yt``
version via: 

.. code-block:: bash

    $> yt version


To install a fresh copy of ``yt`` we must first clone it:

.. code-block:: bash

   $> hg clone https://bitbucket.org/yt_analysis/yt

Finally, we build ``yt`` which may take a little while for everything to
compile:

.. code-block:: bash

   $> cd yt
   $> python setup.py install

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

To update the code you simply need to pull down changes and reinstall:

.. code-block:: bash

   $> cd caesar
   $> git pull
   $> python setup.py install


