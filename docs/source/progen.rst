
Progen
******

.. contents::
   :local:
   :depth: 3

----

The ``progen`` module in ``CAESAR`` can compute rudimentary merger
tree info for each group (galaxy/halo/cloud), by computing the most
massive progenitor(s) or descendant(s) for each group in a different
snapshot.  Groups are linked by finding the most particles in common
of a specified particle type (e.g. ``star``).  If snapshot numbers
are specified in rising order, then descendants are computed; if
in falling order, then progenitors are computed.  The information
is appended into the ``CAESAR`` file within the hdf5 dataset
``tree_data``, and are stored separately for progenitors
and descendant, as well as separately for each group type ands
particle type.

Progen over many snapshots
==========================

``run_progen()`` offers the simplest way to run ``progen`` over a list of snapshot numbers, e.g.:

.. code-block:: python

   In [1]: caesar.progen.run_progen('/path/to/snapshots/for/m25n256', 'snap_m25n256_', list(range(151,0,-1), prefix='caesar_')

This will find progenitors (since the snapshots are specified in falling order) in snaphots 0-151 for the snapshots in the directory provided as the first argument, with the snapshot basename provided as the second argument.  
Snapshot numbers for which a snapshot and/or a Caesar file are not found, or for which there is no ``halo_data``, are ignored (with a warning).

The ``prefix`` option specifies the name prefix for the corresponding ``CAESAR`` file in the ``Groups`` subdirectory; in this case, ``snap_m25n256_151.hdf5`` should have its ``CAESAR`` file in ``Groups/caesar_m25n256_151.hdf5``, etc.
The example above uses default options for linking progenitors/descendants; other choices can be specified as noted in "Progen options" below.  ``run_progen()`` does not return anything.

Progen on a single snapshot
===========================

``progen_finder()`` links the groups in two specified ``CAESAR`` objects, and then writes it to the specified ``CAESAR`` file.  While normally called from ``run_progen()``, it can be run stand-alone as well.  This is useful if e.g. your locations for snapshots and Caesar files are not working as assumed in ``run_progen()``.  For example:

.. code-block:: python

   In [1]: import caesar
   In [2]: obj1 = caesar.load(caesarfile1)
   In [3]: obj2 = caesar.load(caesarfile2)
   In [4]: my_progens = caesar.progen.progen_finder(obj1, obj2, caesarfile1)

plus any options you desire as listed in "Progen options".  

``progen_finder()`` returns the progenitor or descendant list, in addition to (optionally) writing to the ``CAESAR`` file.  With ``overwrite=False``, this allows you to simply return the list without actually writing anything to the Caesar file. This is useful if you don't have write permission, or if you need to link two specific snapshots that are not in the default (usually sequential) list stored in the ``CAESAR`` file.

Progen options
==============

The following options can be passed to ``run_progen()`` or ``progen_finder()``:

* ``data_type``: Group type to find progen/descend info for; can be ``galaxy``, ``halo``, or ``cloud``.  *Default:* ``galaxy``
* ``part_type``: Particle type to find progen/descend info for.  *Default:* ``star``
* ``n_most``: Finds the ``n_most`` most massive progenitors/descendants.  If ``n_most``>1, 
  the info is then stored in a array of size ``(ngroups,n_most)``.  *Default:* 1
* ``min_in_common``: Requires that the current group and the prog/desc group have at least this fraction of particles in common to be considered valid.  *Default:* 0.1
* ``overwrite``: If ``True``, (over)writes info into ``CAESAR`` file.  If ``False``, then if it already exists read it in and return it; but if it doesn't already exist, compute and return it but don't touch the ``CAESAR`` file. *Default:* ``True``
* ``nproc``: Number of OpenMP cores (using `joblib <https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html>`_, passed as ``n_jobs``).  ``progen`` is already very fast, so this isn't terribly useful, except maybe for DM halos where there are lots of groups and particles.  *Default:* 1


Where is the info stored?
=========================

By default, the progenitor/descendant info is stored in the ``tree_data`` dataset within the ``CAESAR`` file.  This is a separate dataset from ``galaxy_data``, ``halo_data``, etc.  Within this, the information is stored as numpy arrays of integers, where each integer corresponds to the index of the group in the other snapshot that is its progenitor/descendant info.

The identifier for each array is created by concatenating three pieces of information: Whether it is a progenitor or descendant; the group type; and the particle time.  An example might be ``progen_galaxy_star``, meaning that the indexes in that array are progenitors of galaxies linked via most numbers of stars in common.  This array will have exactly as many entries as there are galaxies in ``galaxy_data``.  

Each of 3 group types can be linked in two ways (``progen``/``descend``) via each of (in principle) 6 particle types, making for 36 potential arrays being stored in ``tree_data``. Of course, galaxies and clouds do not include dark matter particles, so e.g. ``descend_galaxy_dm`` or ``progen_cloud_dm2`` will never exist; thus there are actually 28 potential arrays.

Additionally, ``tree_data`` hold the redshift for which the progenitors and/or descendants have been identified.  You can retrieve info using the ``get_progen_redshift()`` command:

.. code-block:: python

   In [1]: z_descend = caesar.progen.get_progen_redshift(my_caesar_file,'descend_galaxy_star')

or similarly for any other choice of ``progen_XXX_YYY`` or ``descend_XXX_YYY`` info.  


