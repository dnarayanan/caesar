
Photometry
**********

.. contents::
   :local:
   :depth: 3

----

``CAESAR`` can optionally compute photometry for any object(s) in
any available `FSPS band <http://dfm.io/python-fsps/current/>`_.
This is done as in `Pyloser <https://pyloser.readthedocs.io/en/latest/>`_:
Compute the dust extinction to each star based on the line-of-sight
dust column, attenuate its spectrum with a user-selectable attenuation
law, sum the spectra of all stars in the object, and apply the
desired bandpasses.

NOTE: ``CAESAR`` accounts for dust but does *not* do proper dust
radiative transfer!  To e.g. model the far-IR spectrum or predict
extinction laws, you can use `Powderday
<https://powderday.readthedocs.io/en/latest/>`_.  The main advantage
of ``CAESAR`` is speed.  Also, it gives the user more direct control over
the attenuation law used, which may be desirable in some instances.
Results are similar to ``Powderday`` for most galaxies, but differences
at the level of ~0.1 magnitudes are not uncommon.

Installation
============

To compute photometry, two additional packages must be installed:

* `python-fsps <http://dfm.io/python-fsps/current/installation/>`_: Follow
  the instructions, which requires installing and compiling ``FSPS``.
* `synphot <https://synphot.readthedocs.io/en/latest/>`_: Available via
  ``pip`` or in ``conda-forge``.


Running in member_search
========================

The photometry computation for galaxies can be conveniently done as part
of ``member_search()``.  This is invoked by specifying the ``band_names``
option to ``member_search()``.

``CAESAR`` will compute 4 magnitudes for each galaxy, corresponding
to apparent and absolute magnitudes, with and without dust.  These
are stored in dictionaries ``absmag``, ``absmag_nodust``, ``appmag``,
and ``appmag_nodust``, with keywords corresponding to each requested
band (e.g. ``absmag['sdss_r']``) When invoked within ``member_search()``,
``CAESAR`` computes photometry for *all galaxies*.  For doing
halos/clouds/subset of galaxies, see *Running stand-alone* below.

For example, the following command will invoke ``member_search``
for a ``CAESAR`` object ``obj``, which will do everything as before,
then will additionally compute galaxy photometry for all SDSS and
Hubble/WFC3 filters using an LMC extinction law viewed along the ``z``
axis:

.. code-block:: python

   In [1]: obj.member_search(band_names='[sdss,wfc3]',ssp_table_file='SSP_Chab_EL.hdf5',ext_law='lmc',view_dir='z')

Running stand-alone
===================

The photometry module can also be run stand-alone for specified objects.
Any object with stars and gas (stored in ``slist`` and ``glist``) can
have its photometry computed.  To do so, first create a photometry object,
and then apply ``run_pyloser()`` to it.

For example, to run photometry for all halos in a pre-existing ``CAESAR`` catalog:

.. code-block:: python

   In [1]: from caesar.pyloser.pyloser import photometry
   In [2]: ds  = yt.load(SNAP)
   In [3]: sim = caesar.load('my_caesar_file.hdf5')
   In [4]: galphot = photometry(sim,sim.halos,ds=ds,band_names='sdss',nproc=16)
   In [5]: galphot.run_pyloser()

All options as listed under "Photometry Options" are passable to
``photometry``.  The computed SDSS photometry will be available in
the usual dictionaries ``absmag``, ``absmag_nodust``, ``appmag``,
and ``appmag_nodust``, for each halo.


Photometry Options
==================

The following options can be passed to ``member_search()`` or when 
instantiating the ``photometry`` class:

*  ``band_names``: (REQUIRED): The list of band(s) to compute, selected
   from `python-fsps <http://dfm.io/python-fsps/current/installation/>`_
   (use ``fsps.list_filters()`` to see options).  You can also specify a 
   substring (min. 4 characters) to do all bands that contain 
   that substring, e.g. ``'sdss'`` will compute all available SDSS bands.  
   The ``v`` band is always computed; the difference 
   between the ``absmag`` and ``absmag_nodust`` gives A_V.
   There are two special options: ``'all'`` computes all FSPS bands, 
   while ``'uvoir'`` computes all bands bluewards of 5 microns. *Default*: ``['v']``

*  ``ssp_table_file``: Filename containing FSPS spectra lookup table.  If it 
   doesn't exist, it is generated assuming a Chabrier IMF with nebular emission
   and saved to this filename for future use.  If you prefer different FSPS
   options, first generate it using ``generate_SSP_table``, and read it in here.  
   *Default*: ``'SSP_Chab_EL.hdf5'``

*  ``ext_law``: Specifies the extinction law to use.  Current options
   are ``calzetti``, ``chevallard``, ``conroy``, ``cardelli`` (equivalently ``mw``),
   ``smc``, and ``lmc``.  There are two composite extinction laws available:
   ``mix_calz_mw`` uses ``mw`` for galaxies with specific star formation 
   rate sSFR<0.1 Gyr^-1, ``calzetti`` for sSFR>1, and a linear combination
   in between.  ``composite`` additionally adds a metallicity dependence,
   using ``mix_calz_mw`` for Z>Solar, ``smc`` for Z<0.1*Solar, and a linear
   combination in between.  *Default*: ``'composite'``

*  ``view_dir``: Sets viewing direction for computing LOS extinction. Choices 
   are ``x``, ``y``, ``z``.  *Default*: ``'x'``

*  ``use_dust``: If present, uses the particles' dust masses to compute the 
   LOS extinction.  Otherwise uses the metals, with an assumed dust-to-metals
   ratio of 0.4, reduced for sub-solar metallicities. *Default*: ``True``

*  ``use_cosmic_ext``: Applies redshift-dependent Madau(1995) IGM attenuation 
   to spectra.  This is computed using 
   `synphot.etau_madau() <https://synphot.readthedocs.io/en/latest/synphot/tutorials.html#lyman-alpha-extinction>`_.
   *Default*: ``True``

*  ``nproc``: Number of cores to use.  If -1, it tries to use the ``CAESAR`` object's
   value, or else defaults to 1.  *Default*: -1


Generating a lookup table
=========================

If you don't want Caesar's default choices of Chabrier IMF and nebular emission with
all other options set to the python-FSPS default, you will need to create a new table
and specify it with ``ssp_table_file`` when instantiating ``photometry``.

To create a new SSP lookup table, run ``generate_ssp_table`` with the
desired ``FSPS`` options.  For example:

.. code-block:: python

   In [1]: from caesar.pyloser.pyloser import generate_ssp_table
   In [2]: generate_ssp_table('my_new_SSP_table.hdf5',Zsol=0.0134,oversample=[2,2],imf_type=1,add_neb_emission=True,sfh=0,zcontinuous=1)

Options:

* ``oversample`` oversamples in [age,metallicity] by the specified factors
  from the native ``FSPS`` ranges, in order to get more accurate interpolation.  Note
  that setting these >1 creates a larger output file, by the product of those values.
  *Default*: ``[2,2]``

* ``Zsol`` sets the metallicity in solar units in order to convert the FSPS
  metallicity values into a solar abundance scale. *Default*: ``Solar['total']`` (see pyloser.py)

* The remaining ``**kwargs`` options are passed directly to `fsps.StellarPopulations
  <http://dfm.io/python-fsps/current/stellarpop_api/#example>`_,
  so any stellar population available in ``python-FSPS`` can be generated.
  NOTE: ``sfh=0`` and ``zcontinuous=1`` should always be used.

If you have a lookup table and don't know the options used to generate it,
you can list the ``fsps_options`` data block using the 
`h5dump <https://support.hdfgroup.org/HDF5/doc/RM/Tools/h5dump.htm>`_
command at the system prompt:

.. code-block:: sh

   % h5dump -d fsps_options my_new_SSP_table.hdf5

This will give you a bunch of hdf5 header info but at the end will be the ``DATA`` block
which lists the FSPS options used.

Performance tips
================

* The code is cython parallelized over objects, so for efficiency it is
  best to run many objects within a single ``photometry`` instance.
  Try not to do a single galaxy at a time!  
* Generally, computing the extinction
  and spectra takes most of the time; once the spectra are computed,
  applying bandpasses is fast.  So it is also better to generate as
  many bands as possible in one call.

