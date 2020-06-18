
Photometry
**********

.. contents::
   :local:
   :depth: 3

----

``CAESAR`` can optionally compute photometry for any object(s) in any
available `FSPS band <http://dfm.io/python-fsps/current/>`_.  This is done
by computing the line-of-sight dust extinction to each star, attenuating
its spectrum with a user-selectable attenuation law, summing the spectra
of all stars in the object, and applying the desired bandpasses.

NOTE: ``CAESAR`` does *not* do true dust radiative transfer!
To e.g. model the far-IR spectrum or predict extinction laws, you
should use `Powderday <https://powderday.readthedocs.io/en/latest/>`_.
``CAESAR``'s main advantage is that it is much faster (thanks to being
cython parallelized), so adds only modest compute time over the main
``member_search()`` routine.  It also gives the user more direct control
over the attenuation law used, which may be desirable in some instances.

Installation
============

Computing photometry requires two additional packages to be installed:

1. `python-fsps <http://dfm.io/python-fsps/current/installation/>`_: Follow
   the instructions, which requires installing and compiling ``FSPS``.
2. `synphot <https://synphot.readthedocs.io/en/latest/>`_: Available via
   ``pip`` or in ``conda-forge``.


Running in member_search
========================

The photometry computation for galaxies can be activated in
``member_search()``.  This is done by specifying the ``band_names``
option, which both invokes photometry and tells ``CAESAR`` which bands
to compute.

``CAESAR`` will compute 4 magnitudes for each galaxy, corresponding
to apparent and absolute magnitudes, with and without dust.  These are
stored in dictionaries ``absmag``, ``absmag_nodust``, ``appmag``, and
``appmag_nodust``, with keywords corresponding to each requested band
(e.g. ``absmag['sdss_r']`` returns the absolute magnitude in the SDSS
*r* band).  When invoked within ``member_search()``, ``CAESAR`` only
computes photometry for *galaxies*.  For doing halos/clouds/subset of
galaxies, see *Running stand-alone* below.

Options:

1. ``band_names``: (REQUIRED): The list of band(s) to compute, selected
   from `python-fsps <http://dfm.io/python-fsps/current/installation/>`_
   (use ``fsps.list_filters()`` to see options).  You can also specify a 
   substring (min. 4 characters) to get the set of bands that contain 
   that substring, so e.g. ``'sdss'`` will compute all available SDSS bands.  
   The ``v`` band is always computed; the difference 
   between the ``absmag`` and ``absmag_nodust`` is A_V.
   There are two special options: ``'all'`` computes all FSPS bands, 
   while ``'uvoir'`` computes all bands bluewards of 5 microns.

2.  ``ssp_table_file``: An FSPS lookup table is generated
    and stored in the specified ``hdf5`` file.
    If the file already exists, it is read in, which can save a lot of time.
    *Default*: ``SSP_Chab_EL.hdf5``

3. ``ext_law``: Specifies the extinction law to use.  Current options
   are ``calzetti``, ``chevallard``, ``conroy``, ``cardelli`` (equivalently ``mw``),
   ``smc``, and ``lmc``.  There are two composite extinction laws available:
   ``mix_calz_mw`` uses ``mw`` for galaxies with specific star formation 
   rate sSFR<0.1 Gyr^-1, ``calzetti`` for sSFR>1, and a linear combination
   in between.  ``composite`` additionally adds a metallicity dependence,
   using ``mix_calz_mw`` for Z>Solar, ``smc`` for Z<0.1*Solar, and a linear
   combination in between.  *Default*: ``composite``

4. ``view_dir``: Sets viewing direction for computing LOS extinction. Choices 
   are ``x``, ``y``, ``z``.  *Default*: ``x``

5. ``use_dust``: If present, uses the particles' dust masses to compute the 
   LOS extinction.  Otherwise uses the metals, with an assumed dust-to-metals
   ratio of 0.4, reduced for sub-solar metallicities. *Default*: ``True``

6. ``use_cosmic_ext``: Applies redshift-dependent Madau(1995) IGM attenuation 
   to spectra.  This is computed using `synphot <https://synphot.readthedocs.io/en/latest/>`_.
   *Default*: ``True``

7. ``nproc``: Number of cores to use.  If -1, it tries to use the ``CAESAR`` object's
   value, or else defaults to 1.  *Default*: -1

For example, this will invoke ``member_search`` for a ``CAESAR``
object ``obj``, which will do everything as before, but at the end
will additionally compute galaxy photometry for all SDSS and
Hubble/WFC3 filters using an LMC extinction law in the ``z``
direction.

.. code-block:: python

   In [1]: obj.member_search(band_names='[sdss,wfc3]',ssp_table_file='SSP_Chab_EL.hdf5',ext_law='lmc',view_dir='z')



Running stand-alone
===================

The photometry module can also be run stand-alone for specified objects, or
to get individual magnitudes for stars.

Sets of objects
---------------

Any object with stars and gas (stored in ``slist`` and ``glist``) can
have its photometry computed.  The steps are to first create a photometry object,
and then invoke the photometry computation on it.

For example, to run photometry for all halos in a pre-existing ``CAESAR`` catalog:

.. code-block:: python

   In [1]: from caesar.pyloser.pyloser import photometry
   In [2]: ds  = yt.load(SNAP)
   In [3]: sim = caesar.load('my_caesar_file.hdf5')
   In [4]: galphot = photometry(sim,sim.halos,ds=ds,band_names='sdss',nproc=16)
   In [5]: galphot.run_pyloser()

All options as listed above are passable to ``photometry()``.  The
computed SDSS photometry will be available in the usual dictionaries
``absmag``, ``absmag_nodust``, ``appmag``, and ``appmag_nodust``,
for each halo.

Individual star magnitudes
--------------------------

Not available yet.

Generating an SSP lookup table
------------------------------

To create a new table, run ``generate_ssp_table`` with the
desired ``FSPS`` options:

.. code-block:: python

   In [1]: from caesar.pyloser.pyloser import generate_ssp_table
   In [2]: generate_ssp_table('my_new_SSP_table.hdf5',Zsol=Solar['total'],fsps_imf_type=1,
           fsps_nebular=True,fsps_sfh=0,fsps_zcontinuous=1,oversample=[2,2])

The ``oversample`` option oversamples in [age,metallicity] by the specified factors
from the native ``FSPS`` ranges, in order to get more accurate interpolation.  Note
that this creates a larger output file, by the product of those values.


