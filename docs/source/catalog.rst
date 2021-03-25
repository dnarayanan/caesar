
Catalog Quantities
******************

.. contents::
   :local:
   :depth: 3

----

``CAESAR`` computes many quantities for galaxies (from 6DFoF) and halos identified within a given simulation snapshot.  Below we describe the structure of the catalog file, and the quantities that are computed.

Structure
=========

The top level of the catalog ``hdf5`` file contains:

.. code-block:: python

   $> h5ls CAESARFILE
   galaxy_data              Group
   global_lists             Group
   halo_data                Group
   simulation_attributes    Group
   tree_data                Group

``galaxy_data`` and ``halo_data`` contain the galaxy and halo catalogs, respectively.  ``simulation_attributes`` contains various simulation parameters.  ``global_lists`` contains some auxiliary lists; there is no good reason to directly access this.  Finally, ``tree_data`` contains the output of running ``progen``, which is not initially created by ``CAESAR`` but can be added later (click on *Progenitors* tab for more info).

galaxy_data
===========

``galaxy_data`` contains some computed quantities for each galaxy, but many of the quantities are in *dictionaries* which are listed in ``dicts`` (described below).  There is also ``lists``, which stores particle lists, but it shouldn't be necessary to directly access this.

The list of quantities can be seen using ``h5ls``:

.. code-block:: python

   $> h5ls CAESARFILE/galaxy_data

Quantities stored at the top level in ``galaxy_data`` are:

* ``GroupID`` -- A sequential ID number for each galaxy

* ``parent_halo_index`` -- Index number in ``halo_data`` for the galaxy's parent halo

* ``central`` -- Flag to indicate whether galaxy is central (i.e. most massive in stars) within its halo (1), or a satellite (0)

* ``pos,vel`` -- Center-of-mass (CoM) position and velocity, including units (typically kpccm)

* ``ngas,nstar,ndm,nbh`` -- Number of particles of each particle type

* ``sfr`` -- Instantaneous star formation rate in Msun/yr, from summing SFR in gas particles

* ``sfr_100`` -- SFR averaged over last 100 Myr, from star particles formed in that time.

* ``bhmdot``, ``bh_fedd`` -- Central black hole accretion rate in Msun/yr, and central BH eddington ratio.  The central black hole is taken to be the most massive one, if there are multiple BH in the galaxy.

* ``L_FIR`` -- If photometry was done, this will contain the bolometric far-IR luminosity in erg/s (i.e. the total energy absorbed by dust extinction)

* Various list start/end values -- These are indexes for the particle lists; these should not be accessed directly, but rather through ``glist``, ``slist``, etc. (see below)

Dictionary quantities
=====================

``dicts`` contains the majority of the computed quantities.  These are accessed via a dictionary key, e.g. ``obj.galaxies[0].masses['stellar']`` gives the stellar mass of the first galaxy in the catalog.  The full list of quantities in any given file can be seen using:

.. code-block:: python

   $> h5ls CAESARFILE/galaxy_data/dicts

``dicts`` contains the following quantities:

* ``masses``:  ``['gas','stellar','dm','dust','bh','HI','H2']`` as well as many corresponding quantities within 30 kpccm spherical apertures denoted by ``_30kpc`` attached to each name.  The first 5 of these come directly by summing particle masses.  The ``HI`` and ``H2`` masses come from assigning all the gas in the halo to its most bound galaxy within the halo (see Dave et al. 2020).  Note that the ``dm`` mass from 6DFOF is 0 by definition, since the 6DFOF does not consider DM particles; the ``dm_30kpc`` however will be nonzero.

* ``radii``:  ``['gas_XX','stellar_XX','dm_XX','bh_XX','baryon_XX', 'total_XX']``, where ``XX`` is ``half_mass``, ``r20``, or ``r80``, which are the radii enclosing 50, 20, and 80 percent of the mass of the given type.  The galaxy center of mass from which the radii are found is recomputed for each type.  ``baryon`` includes ``gas``, ``stellar``, and ``bh``, while ``total`` includes ``dm`` as well.

* ``metallicities``: ``['mass_weighted','sfr_weighted','stellar','mass_weighted_cgm','temp_weighted_cgm']``.  The first two are gas-phase, weighted as indicated.  The stellar metallicity is mass-weighted.  The CGM metallicities are for gas outside galaxies (*n_H<0.13* H atom/cm^3); this is only meaningful for halos.  These are in total metal mass fractions (not solar-scaled).  This uses Metallicity[0] from the snapshot, which is the total metallicity; ``CAESAR`` does not have any information regarding specific elements, this must be obtained from the snapshot directly if desired using e.g. ``pygadgetreader``.

* ``velocity_dispersions``: ``['gas','stellar','dm','bh','baryon', 'total']``. Mass-weighted velocity dispersions for each particle type, computed around the CoM velocity (recomputed for each type).  These are in km/s.

* ``rotation``: ``['gas_XX','stellar_XX','dm_XX','bh_XX','baryon_XX', 'total_XX']``, where ``XX`` here can be ``L``, ``ALPHA``, ``BETA``, ``BoverT``, and ``kappa_rot``.  ``L`` (3 components) is the angular momentum vector of the galaxy in Msun-kpccm-km/s.  ``ALPHA`` and ``BETA`` are rotation angles required to rotate the galaxy to align with the angular momentum.  ``BoverT`` is bulge-to-total mass ratio, where the bulge mass is defined kinematically as twice the counter-rotating mass.  ``kappa_rot`` is the fraction of kinetic energy in rotation, as defined in Sales et al. (2012).

* ``ages``: ``['mass_weighted','metal_weighted']`` Mean stellar ages, weighted by mass or (additionally) metallicity.

* ``temperatures``: ``['mass_weighted','mass_weighted_cgm','temp_weighted_cgm']`` These are the average temperatures of the gas within galaxies or in the CGM. Owing to the assumed equation of state in cosmological simulations, this is typically not very meaningful for galaxies.  However, it is useful for halos.

* ``local_mass_density`` and ``local_number_density``: ``[300,1000,3000]``. Environmental measures giving the mass and number density of ``CAESAR`` galaxies within spherical tophat apertures as indicated in kpccm.

* Photometry:  ``absmag`` and ``appmag``, along with corresponding ``_nodust`` values, for all the photometric bands computed (if photometry was run).  More information is available in the Photometry docs.

halo data
=========

``halo_data`` contains many of the same quantities as ``galaxy_data``.  However, there are some crucial differences.

At the top level, there are some new quantities:

* ``minpotpos``, ``minpotvel``:  Position and velocity of the particle with the lowest potential in the halo.  This is often a more useful that the CoM values within halos, since FoF halos can be quite irregular in shape.

* ``central_galaxy``: GroupID of central galaxy in the halo.

* ``galaxy_index_list_start/end``: This is the indexing for the list of galaxy GroupID's in the halo.  DO NOT USE THESE VALUES DIRECTLY TO LOOK IN ``galaxy_data``!  These are cross-indexed, so to get the galaxy indexes within a given halo use 

.. code-block:: python

   In[1]: halogals = np.asarray([i.galaxy_index_list for i in obj.halos])


Meanwhile, in ``halo_data/dicts``, beyond all the ``galaxy_data`` dictionaries (except photometry) there is a new dictionary called ``virial_quantities``:

* ``virial_quantities``: ``['circular_velocity','spin_param','temperature','mXXXc', 'rXXXc']``: Circular velocity=*sqrt(GM_tot/R_tot)* where *R_tot* is the equivalent radius that would enclose *M_tot* at an overdensity of 200 times the critical.  The ``XXX`` quantities for mass and radii are computed within ``200``, ``500``, or ``2500`` times the critical density, by expanding a sphere around ``minpotpos`` until the mean density within drops below that value.  Note that only halo particles are included, so owing to the irregular shapes of FoF halos, this can lead to ``200`` quantities sometimes missing significant mass; for ``500`` and ``2500`` the effects are quite small.  Overall, these values should be regarded as somewhat approximate to be used for rough analyses.


Particle lists
==============

Each halo and galaxy contains a list of particles indexed by particle type.  For gas, stars, DM, and BHs these are ``glist``, ``slist``, ``dmlist``, and ``bhlist``, respectively. These lists contain the indexes of particles of a given type within the original snapshot.  These lists allow the user to compute any desired quantity, by looking up the required quantities within the original snapshot.

To use these lists, one must read in the particles from the snapshot.  This can be done for instance using ``pygadgetreader``.  For instance, the ``CAESAR`` file does not contain metallicities of individual elements.  So one might desire, e.g. the SFR-weighted oxygen abundance.  

To do this, we first use ``pygadgetreader`` to read in the particle lists:

.. code-block:: python

   In[1]: import caesar
   In[2]: from readgadget import readsnap  # pygadgetreader
   In[3]: obj = caesar.load(CAESARFILE)
   In;4]: h = obj.simulation.hubble_constant  # H0/100
   In[5]: gsfr = readsnap(SNAPFILE,'sfr','gas',units=1) # particle SFRs in Mo/yr
   In[6]: gmetarray = readsnap(SNAPFILE,'Metallicity','gas') # For Simba, this is an 11-element array per particle
   In[7]: pOgas = np.asarray([i[4] for i in gmetarray])  # For Simba, oxygen is 5th element in the Metallicity array

Next, we use ``glist`` to compile the particles in each galaxy, and use them to compute the SFR-weighted oxygen abundance:

.. code-block:: python

   In[8]: Zoxy = []
   In[9]: for g in sim.galaxies:
   In[10]:     psfr = np.array([gsfr[k] for k in g.glist]) # particle sfr's
   In[11]:     ZO = np.array([pOgas[k] for k in g.glist]) # oxygen mass fraction
   In[12]:     Zoxy.append(np.sum(ZO*psfr)/np.sum(psfr))

This fills an array ``Zoxy`` with the SFR-weighted metallicity.

In this way, ``CAESAR`` (plus a particle reader of your choice) enables the computation of any quantity associated with a given galaxy or halo object.

