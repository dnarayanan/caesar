
VTK Visualization
=================

There are some built in methods to visualize particle clouds via VTK.
These require the `python-vtk <http://www.vtk.org/>`_ wrapper to be
installed.  Unfortunately, compiling this wrapper manually is quite
painful - I highly suggest you utilize the ``conda`` package manager
to take care of this one for you via:

.. code-block:: bash

   $> conda install vtk

Afterwards the VTK methods described below should work.

.. toctree::
   :maxdepth: 2

   vtk_funcs              
   vtk_vis
