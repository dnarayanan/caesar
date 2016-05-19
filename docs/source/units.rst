
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

.. code-block:: python

   In [1]: obj.galaxies[0].radii['total']
   Out[1]: 22.2676089684 kpccm

The total radius for this particular galaxy is ``~22 kpccm``.  That
``cm`` tacked onto the end of the units stands for *comoving*.
Because this particular galaxy is at *z=2* we may want to convert that
radius to physical units:

.. code-block:: python
                
   In [2]: obj.galaxies[0].radii['total'].to('kpc')
   Out[2]: 7.42253557308 kpc
   
That was easy right?  What if we want to convert to physical *kpc/h*?

.. code-block:: python

   In [3]: obj.galaxies[0].radii['total'].to('kpc/h')
   Out[3]: 5.19577490116 kpc/h

Keeping track of units suddenly becomes much more intuitive and
easier.  For further information and tutorials regarding yt's units
please visit the `symbolic unit
<http://yt-project.org/doc/analyzing/units/index.html>`_ page.
