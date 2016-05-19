
Loading CAESAR files
********************

.. contents::
   :local:
   :depth: 3

----

Command Line
============

Using the CLI we can load our CAESAR file from the previous example
automatically and have it drop us at an ``ipython`` prompt via:

.. code-block:: bash

   $> caesar caesar_snapshot.hdf5

This will open up the ``caesar_snapshot.hdf5`` file and check for the
``CAESAR=True`` attribute in the HDF5 header.  If found it will
proceed to deserialize the ``CAESAR`` file and drop you to an
interactive python prompt with access to the :class:`main.CAESAR`
object via the ``obj`` variable.  At this point you are free to
explore the data structure and manipulate at will.

----

Scripted
========

In order to do more in depth analysis, you will likely want to built
your own analysis scripts.  Before getting into the nuts and bolts of
your analysis you will need to load in your ``CAESAR`` file to gain
access to all objects and their respective attributes.  This can be
accomplished with the following code:

.. code-block:: python

   import caesar

   # define input file
   infile = 'caesar_snapshot.hdf5'

   # load in input file
   obj = caesar.load(infile)


