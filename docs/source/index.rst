.. CAESAR documentation master file, created by
   sphinx-quickstart on Mon May 16 19:26:15 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CAESAR's documentation!
==================================

``CAESAR`` is a python framework for analyzing the outputs from
cosmological simulations.  It aims to provide a simple yet intuitive
interface for exploring data within your outputs.  It starts by
identifying objects (halos and galaxies), calculates a number of
properties for each individual object, and finally links objects to
each other via simple relationships.  The resulting output is a
portable ``HDF5`` file that can be read in and explored without the
original simulation binary.

``CAESAR`` builds upon the `yt <http://yt-project.org>`_ project,
which provides support for a number of `simulation codes
<http://yt-project.org/doc/reference/code_support.html>`_, and
`symbolic units
<http://yt-project.org/doc/analyzing/units/index.html>`_.  All
meaningful quantities stored within a ``CAESAR`` output have these
units attached removing a lot of ambiguity when working with your
data.  No more will you forget if you properly accounted for little
*h* or if you are working in comoving/physical coordinates.  Currently
``CAESAR`` supports the following codes/formats, with more coming
online in the near future:

1. `GADGET <http://wwwmpa.mpa-garching.mpg.de/~volker/gadget/>`_
2. `GIZMO <http://www.tapir.caltech.edu/~phopkins/Site/GIZMO.html>`_
3. `TIPSY <http://www-hpcc.astro.washington.edu/tools/tipsy/tipsy.html>`_
4. `ENZO <http://enzo-project.org/>`_

To get started, follow the Getting Started link below!

----

Contents
========

.. toctree::
   :maxdepth: 2

   getting_started
   running
   loading


----

.. toctree::
   :maxdepth: 2

   code_reference/reference
   
----

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

