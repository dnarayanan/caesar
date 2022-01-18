
Running CAESAR
**************

.. contents::
   :local:
   :depth: 3

``CAESAR`` offers three basic functions:

[1] Identify galaxies and halos, compute a wide range of properties for each object, and cross-match them.

[2] Compute photometry accounting for the line-of-sight dust extinction to each star in the object.

[3] Compute the N most massive progenitors/descendants for any galaxy or halo in another snapshot
(see Progenitors docs page for usage).

----

Scripted
========

It is generally recommended to run ``CAESAR`` within a script for computing galaxy
and halo properties.  This allows for more
precise control over the various options, looping over many files, parallelizing, etc. 
Here is a basic script for running ``member_search``:

.. code-block:: python

   import yt
   import caesar

   # first we load your snapshot into yt
   ds = yt.load('my_snapshot')

   # now we create a CAESAR object, and pass along the yt dataset
   obj = caesar.CAESAR(ds)

   # now we execute member_search(), which identifies halos, galaxies, and computes
   # properties (including photometry; this requires installing FSPS) on 16 OpenMP cores
   obj.member_search(haloid='fof',fof6d_file='my_fof6dfile',fsps_bands='uvoir',ssp_model='FSPS',ssp_table_file='FSPS_Chab_EL.hdf5',ext_law='composite',nproc=16)

   # finally we save the CAESAR galaxy/halo catalog to your desired filename
   obj.save('my_caesar_file.hdf5')

member_search() options
----------------------- 

Here is a more detailed description of the options shown above:

* **nproc**:  Number of cores for OpenMP parallelization.  This follows the ``joblib`` convention that negative numbers correspond to using all except ``nproc+1`` cores, e.g. ``nproc=-2`` uses all but 1 cores. *Default:* 1

* **haloid**:  Source for particle halo ID's.  
  ``haloid='fof'`` uses a 3D Friends-of-Friends (3DFOF) with b=0.2 to identify halos.  
  ``haloid='snap'`` reads halo membership info for each particle from the snapshot variable ``HaloID``, if present.  
  *Default:* 'fof'

* **fof6d_file**:  Stores results of 6DFOF galaxy finder in a file for future retrieval.  If file does not exist, it is created; if it exists, the galaxy membership information is read from this file instead of running the 6DFOF.  *Default:* *None*

* **fsps_bands**:  Triggers optional photometry computation, in specified list of bands. The ``fsps.list_filters()`` command under ``python-fsps`` lists the available bands.  One can also specify a string (minimum 4 characters) that will be matched to all available bands, e.g. ``fsps_bands=['sdss','jwst']`` will compute all bands that include the phrase ``sdss`` or ``jwst``. *Default:* *None*

* **ssp_model**:  Choice of ``FSPS``, ``BPASS``, or ``BC03`` (Bruzual-Charlot 2003).  *Default:* *None*

* **ssp_table_file**: Path to lookup table for FSPS photometry.  If this file does not exist or this keyword is unspecified, it will be generated; this takes some time.  If it exists, ``CAESAR`` will read it in. *Default:* *None*


----

Command Line
============

NOTE: CURRENTLY, RUNNING FROM THE COMMAND-LINE IS NOT OPERATIONAL.  
Please use the Scripted method described above.

..
   Running ``CAESAR``'s primary functionality is very simple.  The command line interface (CLI) allows you to quickly execute ``CAESAR`` on a single snapshot:

   .. code-block:: bash

      $> caesar snapshot

   This will run the code, an output a catalog file named ``caesar_snapshot.hdf5``.



