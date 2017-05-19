
Using CAESAR
************

.. contents::
   :local:
   :depth: 3

----

Data Structure
==============

Within the :class:`main.CAESAR` object you will find a number of
*lists* containing any number of :class:`group.Group` objects.  The
primary lists are the ``halos`` and ``galaxies`` list; within each of
these you will find every object identified by ``CAESAR``.  Below is a
quick relationship diagram describing how objects are linked together:

.. image:: https://www.dropbox.com/s/df50wkx3gzrtbfa/caesar_classes.png?raw=1
   :target: https://www.dropbox.com/s/df50wkx3gzrtbfa/caesar_classes.png?raw=1
   :alt: CAESAR Data Structure

From this you can see that each ``Halo`` object has a list of
galaxies, a link to its central galaxy, and a sub-list of satellite
galaxies (those who are not a central).  Each ``Galaxy`` object has a
link to its parent ``Halo``, a boolean describing if it is a central,
and a sub-list linking to all of the satellite galaxies for its parent
halo.
      
----

Usage
=====

Usage of ``CAESAR`` all comes down to *what* you want to do.  The real
power of the code comes with the simple relationships that link each
object.  As we saw in the previous section, each :class:`group.Group`
has a set of relationships.  We can exploit these to intuitively query
our data.  For example, say you wanted to get an array of all galaxy
masses, how would you most efficiently do that?  The easiest way (in
my opinion) would be to use python's `list comprehension
<https://docs.python.org/2/tutorial/datastructures.html#list-comprehensions>`_.
Here is a quick example (assuming you have the :class:`main.CAESAR`
object loaded into the ``obj`` variable):

.. code-block:: python
                
   galaxy_masses = [i.masses['total'] for i in obj.galaxies]

This is basically a compact way of writing:

.. code-block:: python

   galaxy_masses = []
   for gal in obj.galaxies:
       galaxy_masses.append(gal.masses['total'])

Now that in itself is not all that impressive.  Things get a bit more
interesting when we start exploiting the object relationships.  As
another example, say we wanted the *parent halo mass* of each galaxy?
Lets see how that is done:

.. code-block:: python

   parent_halo_masses = [i.halo.masses['total'] for i in obj.galaxies]

Since each :class:`group.Galaxy` has a link to its parent
:class:`group.Halo`, we have access to all of the parent halo's
attributes.  We can also begin to add conditional statements to our
list comprehension statements to further refine our results; let's
only look at the halo masses of massive galaxies:

.. code-block:: python

   central_galaxy_halo_masses = [i.halo.masses['total'] for i in obj.galaxies if i.masses['total'] > 1.0e12]

Obviously we can make these list comprehensions infinitely
complicated, but I think you get the gist.  The bottom line is:
**CAESAR provides a convenient and intuitive way to relate objects to
one another**.
