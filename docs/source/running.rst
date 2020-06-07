
Running CAESAR
**************

.. contents::
   :local:
   :depth: 3

``CAESAR`` offers three basic functions:

[1] Identify galaxies and halos, compute a wide range of properties for each object, and cross-match them;

[2] Compute photometry accounting for the line-of-sight dust extinction to each star in the object; and

[3] Compute the most massive progenitors/descendant for any galaxy or halo in another snapshot [UNDER CONSTRUCTION].

----

Command Line
============

Running ``CAESAR``'s first (and primary) functionality is very simple.  
The command line interface (CLI) allows you to quickly execute
``CAESAR`` on a single snapshot:

.. code-block:: bash

   $> caesar snapshot

This will run the code, an output a catalog file named ``caesar_snapshot.hdf5``.

CLI Options
-----------

* LIST OPTIONS HERE [TODO]

----

Scripted
========

It is also possible to run ``CAESAR`` within a script.  This is 
useful e.g. for running over multiple snapshots.

Here is a simple example of how one might write a script to perform the same action as
the CLI command above; it also details what the CLI command does:

.. code-block:: python

   import yt
   import caesar

   # here we define the name of our simulation snapshot
   snap = 'snapshot'

   # now we load that snapshot into yt
   ds = yt.load(snap)

   # create a new CAESAR object, and pass along the yt dataset
   obj = caesar.CAESAR(ds)

   # now we execute the member_search() method, which handles the bulk
   # of the work
   obj.member_search()

   # now we save the output
   obj.save('caesar_snapshot.hdf5')
  
member_search() options
----------------------- 

* **nproc**:  Number of cores for OpenMP parallelization.  This follows the ``joblib`` convention that negative numbers correspond to using all except ``nproc+1`` cores, e.g. ``nproc=-2`` uses all but 1 cores. *Default:* 1

* **haloid**:  Source for halo ID's.  *Default:* 'fof'

  * ``haloid='fof'`` uses a 3D Friends-of-Friends (3DFOF) with b=0.2 to identify halos.  

  * ``haloid='snap'`` reads halo membership info for each particle from the snapshot variable ``HaloID``.  

* **fof6d_file**:  Stores results of 6DFOF galaxy finder in a file for future retrieval.  If file does not exist, it is created; if it exists, the galaxy membership information is read from this file instead of running the 6DFOF.  *Default:* *None*

* **fsps_bands**:  Triggers optional photometry computation, in specified list of bands. The ``fsps.list_filters()`` command under ``python-fsps`` lists the available bands.  One can also specify a string (minimum 4 characters) that will be matched to all available bands, e.g. ``fsps_bands=['sdss','jwst']`` will compute all bands that include the phrase ``sdss`` or ``jwst``. *Default:* *None*

* **ssp_table_file**: Path to lookup table for FSPS photometry.  If this file does not exist or this keyword is unspecified, it will be generated; this takes some time.  If it exists, ``CAESAR`` will read it in. *Default:* *None*




