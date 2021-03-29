
Aperture Quantities
*******************

.. contents::
   :local:
   :depth: 3

----

``CAESAR`` comes with a stand-alone function to sum quantities within a user-specified aperture around galaxies, called ``get_aperture_masses()``.  This operates directly on galaxies and halos from a ``CAESAR`` catalog, so must be used after the catalog has been generated.

``get_aperture_masses()`` can compute masses for any particle type, or HI/H2/SFR.  The aperture size can be specified as a constant for all galaxies, or an array of length the number of galaxies.  This can be done in 3-D, or in 2-D projected along any principal axis.  It is fully cython parallel.

Note that ``get_aperture_masses()`` only sums over particles within a galaxy's halo.  For a large aperture, or for satellites near the edge of the halo, this may not give fully accurate answers.  Also, this means 2-D projections are done through the entire halo.

Usage
=====

This example computes the quantities listed in ``myquants`` in a 2-D aperture projected in ``z``, within an aperture given by twice the stellar half mass radius for each galaxy, on 8 cores:

.. code-block:: python

   In [1]: import caesar
   In [2]: from caesar.hydrogen_mass_calc import get_aperture_masses
   In [3]: sim = caesar.load(CAESARFILE)
   In [4]: myquants = ['gas','star','dm','sfr','HI','H2']
   In [5]: rhalf = np.array([i.radii['stellar_half_mass'] for i in sim.galaxies])
   In [6]: m_apert = get_aperture_masses(SNAPFILE,sim.galaxies,sim.halos,quantities=myquants,aperture=2*rhalf,projection='z',nproc=8)

``get_aperture_masses()`` returns a 2-D array of size ``(Nquants,Ngal)``, with the aperture-summed quantities for each galaxy, in the order specified in the ``quantities`` option.  ``CAESARFILE`` and ``SNAPFILE`` are the filenames of the ``CAESAR`` catalog and particle snapshot, respectively.

Options
=======

``get_aperture_masses()`` requires the original simulation snapshot, as well as the ``galaxies`` and ``halos`` objects from the corresponding ``CAESAR`` catalog.  It operates only on galaxies, and cannot operate on a subset of objects.

The following options can be passed to ``get_aperture_masses()``:

* ``quantities``: Can be any particle type (e.g. ``'gas'``), or else ``'HI'``, ``'H2'`` or ``'sfr'``.  *Default:* ``['gas','star','dm']``
* ``aperture``:  The aperture size.  Can be a constant, which is assumed to be in comoving kpc, or else an array of length ``Ngalaxies``.  *Default:* ``30``
* ``projection``: ``None`` gives the 3-D aperture values.  Specifying ``'x'``, ``'y'``, ``'z'`` gives the 2-D aperture projected along that axis. *Default:* ``None``
* ``exclude``: ``None`` includes all particles in halo.  ``satellites`` or ``central`` excludes particles in satellite or central galaxies, respectively.  ``galaxies``/``both`` excludes all particles in galaxies. *Default:* ``None``
* ``nproc``: Number of OpenMP cores (using `joblib <https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html>`_, passed as ``n_jobs``). *Default:* 1

The routine returns a single 2-D array of length ``(Nquants,Ngal)``, where ``Nquants=len(quantities)`` and ``Ngal=len(sim.galaxies)``.

