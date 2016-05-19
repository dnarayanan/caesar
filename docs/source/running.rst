
Running CAESAR
**************

.. contents::
   :local:
   :depth: 3

----

Command Line
============

Running ``CAESAR`` has been made extremely simple!  There is a nifty
command line interface (CLI) which allows you to quickly execute
``CAESAR`` on a single snapshot.  This is done by:

.. code-block:: bash

   $> caesar snapshot

This will run the code, an output a file named
``caesar_snapshot.hdf5``.

----

Scripted
========

Scripting ``CAESAR`` is also quite simple.  The following is a simple
example of how one might write a script to perform the same action as
was done in the above section (Simple Usage):

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
   
