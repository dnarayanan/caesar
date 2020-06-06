
Units
*****

.. contents::
   :local:
   :depth: 3

----

``CAESAR`` leverages yt's `symbolic units
<http://yt-project.org/doc/analyzing/units/index.html>`_.  Every
meaningful quantity *should* have a unit attached to it.  One of this
advantages this provides is that you no longer have to keep track of
little *h* or remembering if you are dealing with comoving or physical
coordinates.

Let's take a look at some quick examples of what the units look like,
and how we might take advantage of the easy conversion methods.
Say we have a ``CAESAR`` object with ``obj.simulation.redshift=2`` .
``Caesar`` generally
defaults its length units to be comoving kpc (``kpccm``):

.. code-block:: python

   In [1]: obj.galaxies[0].radii['total']
   Out[1]: 22.2676089684 kpccm

Note the ``cm`` tacked on, which stands for *comoving*.

Because this particular galaxy is at *z=2* we may want to convert that
radius to *physical* kiloparsecs:

.. code-block:: python
                
   In [2]: obj.galaxies[0].radii['total'].to('kpc')
   Out[2]: 7.42253557308 kpc
   
or to physical *kpc/h* (using ``obj.simulation.hubble_constant=0.7``):

.. code-block:: python

   In [3]: obj.galaxies[0].radii['total'].to('kpc/h')
   Out[3]: 5.19577490116 kpc/h

When adding and subtracting quantities, they will be all be converted 
to the units of the first quantity.  You don't have to worry about
homegenizing the units yourself!

Working with units
******************

Quantities that are added or subtracted must have convertible units.
This means you cannot add a simple number to a quantity with
symbolic units; you must first assign a unit to that number (or
array).

To assign a unit, you can use the yt functions ``YTQuantity`` and ``YTArray``:

.. code-block:: python

   In [4]: from yt import YTQuantity
   In [5]: x = YTQuantity(10, 'Mpc')
   In [6]: print(x.to('kpc'))
   Out[6]: 10000.0 kpc

Similarly, use ``YTArray`` for arrays.  

If you need to get rid of the units and return a value for any reason,
simply append ``.d`` to the quantity:

.. code-block:: python

   In [7]: print(x.d)
   Out[7]: 10
   In [8]: print(x.to('kpc').d)
   Out[8]: 10000.0

For further information and tutorials regarding yt's units
please visit the `symbolic unit
<http://yt-project.org/doc/analyzing/units/index.html>`_ page.

The various units and unit systems that are available in yt are
described `here
<https://yt-project.org/doc/analyzing/units/unit_systems.html>`_ .
