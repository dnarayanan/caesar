import numpy as np
from yt.funcs import mylog
from caesar.utils import memlog
from yt.units.yt_array import YTQuantity
cimport numpy as np
cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cython.parallel import prange
from caesar.property_manager import MY_DTYPE

from libc.stdio cimport printf, fflush, stderr, stdout
from libc.math cimport sqrt as c_sqrt, fabs as c_fabs, log10 as c_log10

cdef extern from "math.h":
    double sqrt(double x)

def check_values(obj):
    """Check to make sure that we have the required fields available to
    perform the hydrogen mass frac calculation.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Main CAESAR object.
    
    Returns
    -------
    bool
        Returns True if all fields are present, False otherwise.

    """
    from caesar.property_manager import has_property
    
    ## check if values needed are present
    required_data = ['temp','sfr','nh','mass','rho','pos']
    missing = []
    for rd in required_data:
        prop = has_property(obj, 'gas', rd)
        if not prop:
            missing.append(rd)
    if len(missing) > 0:
        mylog.warning('Could not find the following fields: %s; skipping HI/H2 calc' % missing)
        return False
    return True


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def hydrogen_mass_calc(obj,**kwargs):
    """Calculate the neutral and molecular mass contents of SPH particles.
    
    For non star forming gas particles assigned to halos we calculate
    the neutral fraction based on equations from Popping+09 and
    Rahmati+13.  If H2 block is not present in the simulation file we
    estimate the neutral and molecular fraciton via Leroy+08.  Once
    these fractions are calculated we assign HI/H2 masses to galaxies
    & halos based on their mass-weighted distances.

    This is not necessary for simulations that self-consistently determine
    HI and H2 fractions.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Main CAESAR object.

    Returns
    -------
    HImass, H2mass : np.ndarray, np.ndarray
        Contains the HImass and H2 mass of each individual particle.

    """
    if not obj.simulation.baryons_present or obj.ngalaxies == 0:
        return 0,0
    if not check_values(obj):
        return 0,0

    #if not 'compute_selfshielding' in kwargs or not kwargs['compute_selfshielding']:
    #    get_HI_masses(obj)
#    return 0.0

    tmp_str = 'Calculating HI/H2 masses for'
    cdef bint all_gas = 0
    if 'calculate_H_for_all_gas' in obj._kwargs and obj._kwargs['calculate_H_for_all_gas']:
        all_gas = 1
        tmp_str = '%s all gas' % tmp_str
    else:
        tmp_str = '%s dense halo gas' % tmp_str

    mylog.info(tmp_str)

    cdef double gamma_HI
    from .treecool_data import UVB
    uvb = UVB['FG11']

    sim = obj.simulation
    
    if np.log10(obj.simulation.redshift + 1.0) > uvb['logz'][len(uvb['logz'])-1]:
        gamma_HI = 0.0
    else:
        gamma_HI = np.interp(np.log10(obj.simulation.redshift + 1.0),
                             uvb['logz'],uvb['gH0'])

    ## density thresholds in atoms/cm^3
    cdef double    low_rho_thresh = 0.001       # atoms/cm^3
    cdef double    rho_thresh     = 0.13        # atoms/cm^3

    cdef double    XH          = obj.simulation.XH
    cdef double    proton_mass = 1.67262178e-24 # g
    cdef double    FSHIELD     = 0.99
    cdef double    redshift    = obj.simulation.redshift
    
    ## Leroy et al 2008, Fig17 (THINGS) Table 6 ##
    cdef double    P0BLITZ     = 1.7e4
    cdef double    ALPHA0BLITZ = 0.8

    ## Poppin+09 constants (equation 7)
    cdef double beta, xi, nHss, fgamma_HI, C
    cdef double a  = 7.982e-11                  # cm^3/s
    cdef double b  = 0.7480
    cdef double T0 = 3.148                      # K
    cdef double T1 = 7.036e5                    # K

    cdef double sigHI     = 3.27e-18 * (1.0+redshift)**(-0.2)
    cdef double fbaryon   = obj.simulation.omega_baryon / obj.simulation.omega_matter
    cdef double nHss_part = 6.73e-3 * (sigHI/2.49e-18)**(-2./3.) * (fbaryon / 0.17)**(-1./3.)
    
    ## global lists
    cdef np.int64_t[:] halo_glist   = np.array(obj.global_particle_lists.halo_glist,dtype=np.int64)
    cdef np.int64_t[:] galaxy_glist = np.array(obj.global_particle_lists.galaxy_glist,dtype=np.int64)

    cdef int ngas = len(obj.global_particle_lists.halo_glist)
    cdef int nhalogas = len(obj.global_particle_lists.halo_glist[obj.global_particle_lists.halo_glist>=0])
    cdef int ngalgas = len(obj.global_particle_lists.galaxy_glist[obj.global_particle_lists.galaxy_glist>=0])

    ## gas properties
    from caesar.property_manager import get_property, has_property

    #cdef np.float64_t[:,:] gpos  = obj.data_manager.pos[obj.data_manager.glist]
    cdef np.int64_t[:]   idlist = obj.data_manager.glist[obj.global_particle_lists.halo_glist>=0]
    cdef np.float64_t[:,:]   gpos = obj.data_manager.pos[obj.data_manager.glist][obj.global_particle_lists.halo_glist>=0]
    cdef np.float64_t[:]   gmass = obj.data_manager.mass[obj.data_manager.glist][obj.global_particle_lists.halo_glist>=0]
    cdef np.float64_t[:]   grhoH = get_property(obj, 'rho', 'gas').in_cgs().d[obj.global_particle_lists.halo_glist>=0] * XH / proton_mass
    #cdef np.float64_t[:]    gsfr = obj.data_manager.gsfr.d
    cdef np.float64_t[:]   gtemp = obj.data_manager.gT.to('K').d[obj.global_particle_lists.halo_glist>=0]
    cdef np.float64_t[:]   gnh = get_property(obj, 'nh', 'gas').d[obj.global_particle_lists.halo_glist>=0]
    cdef np.float64_t[:]   gfH2
        
    cdef int i,j
    cdef int nhalos    = len(obj.halos)
    cdef int ngalaxies = len(obj.galaxies)
        
    cdef bint H2_data_present = 0
    if has_property(obj, 'gas', 'fH2'):
        gfH2  = get_property(obj, 'fH2', 'gas').d[obj.global_particle_lists.halo_glist>=0]
        H2_data_present = 1
    else:
        mylog.warning('Could not locate molecular fraction data. Computing self-shielding via Rahmati+13, H2 via Leroy+08')

    cdef bint selfshield_flag = 0
    if 'compute_selfshielding' in kwargs and kwargs['compute_selfshielding']:
        selfshield_flag = 1

    cdef np.ndarray[np.float64_t,ndim=1] HImass = np.zeros(nhalogas)
    cdef np.ndarray[np.float64_t,ndim=1] H2mass = np.zeros(nhalogas)

    cdef double fHI, fH2
    cdef double cold_phase_massfrac, Rmol

    ## calculate HI & H2 mass for each valid gas particle
    for i in range(0,nhalogas):
        ## skip if not assigned to a halo
        if not all_gas and halo_glist[idlist[i]] < 0:
                continue

        ## skip if density is too low
        if not all_gas and grhoH[i] < low_rho_thresh:
                continue


        fHI = gnh[i]
        fH2 = 0.0

        ## low density non-self shielded gas
        if grhoH[i] < rho_thresh and 'compute_selfshielding' in obj._kwargs and obj._kwargs['compute_selfshielding']:
            ### Popping+09 equations 3, 7, 4
            #xi       = fHI
            beta     = a / (sqrt(gtemp[i]/T0) *
                            (1.0 + sqrt(gtemp[i]/T0))**(1.0-b) *
                            (1.0 + sqrt(gtemp[i]/T1))**(1.0+b))   # cm^3/s
        #gamma_HI = (1.0-xi)*(1.0-xi) * grhoH[i] * beta / xi   # 1/s
            
            ## Rahmati+13 equations 2, 1
            nHss      = nHss_part * (gtemp[i] * 1.0e-4)**0.17 * (gamma_HI * 1.0e12)**(2./3.)
            fgamma_HI = 0.98 * (1.0 + (grhoH[i] / nHss)**(1.64))**(-2.28) + 0.02 * (1.0 + grhoH[i] / nHss)**(-0.84)
            
            ## Popping+09 equations 6, 5
            C = grhoH[i] * beta / (gamma_HI * fgamma_HI)
            fHI = (2.0 * C + 1.0 - sqrt((2.0*C+1.0)*(2.0*C+1.0) - 4.0 * C * C)) / (2.0*C)

        if grhoH[i] >= rho_thresh:  # dense gas
            if H2_data_present:  # use H2 from snapshot
                fH2 = gfH2[i]
                fHI = 1.0 - fH2            
                if fH2 > 1.0:   # shouldn't happen...?
                    fH2 = 1.0
                    fHI = 0.0
            else:   # estimate H2 via Leroy+08
                cold_phase_massfrac = (1.0e8 - gtemp[i])/1.0e8
                Rmol = (grhoH[i] * gtemp[i] / P0BLITZ)**ALPHA0BLITZ
                fHI  = FSHIELD * cold_phase_massfrac / (1.0 + Rmol)
                fH2  = FSHIELD - fHI

        HImass[i]    = fHI * XH * gmass[i]
        H2mass[i]    = fH2 * XH * gmass[i]

    ## tally halo HI & H2 masses
    cdef np.float64_t[:] halo_HImass = np.zeros(nhalos)
    cdef np.float64_t[:] halo_H2mass = np.zeros(nhalos)
    for i in range(0,nhalogas):
        if halo_glist[idlist[i]] < 0:
            continue
        halo_HImass[halo_glist[idlist[i]]] += HImass[i]
        halo_H2mass[halo_glist[idlist[i]]] += H2mass[i]

    ## assign halo HI & H2 masses to respective halos
    for i in range(0,nhalos):
        obj.halos[i].masses['HI'] = obj.yt_dataset.quan(halo_HImass[i], obj.units['mass'])
        obj.halos[i].masses['H2'] = obj.yt_dataset.quan(halo_H2mass[i], obj.units['mass'])

    #for i in range(0,3): print(np.log10(obj.halos[i].masses['HI']),np.log10(obj.halos[i].masses['H2']))

    if len(obj.galaxies) == 0:
        return HImass,H2mass
        
    ## galaxy HI+H2 calculation
    cdef np.ndarray[np.float64_t,ndim=2] galaxy_pos  = np.array([s.pos for s in obj.galaxies])
    cdef np.ndarray[np.float64_t,ndim=1] galaxy_mass = np.array([s.masses['total'] for s in obj.galaxies])
    cdef np.float64_t[:] galaxy_HImass = np.zeros(ngalaxies)
    cdef np.float64_t[:] galaxy_H2mass = np.zeros(ngalaxies)
    cdef list halos = obj.halos
    cdef object h

    ## set HI fractions to zero for gas below threshold
    for i in range(0,nhalogas):
        if grhoH[i] < low_rho_thresh: HImass[i] = 0.

    ## loop over halos, calculate galaxy HI+H2 for each galaxy in halo
    from yt.extern.tqdm import tqdm
    for h in tqdm(obj.halos,desc='Halo'):
        ## if not galaxies, skip
        if len(h.galaxy_index_list) == 0:
            continue

        internal_galaxy_pos  = galaxy_pos[h.galaxy_index_list]
        internal_galaxy_mass = galaxy_mass[h.galaxy_index_list]
        internal_galaxy_index_list = np.array(h.galaxy_index_list,dtype=np.int32)

        assign_halo_gas_to_galaxies(h.GroupID, 
                                    internal_galaxy_pos,
                                    internal_galaxy_mass,
                                    internal_galaxy_index_list,
                                    galaxy_glist, halo_glist, idlist, 
                                    gpos, HImass, H2mass,
                                    galaxy_HImass, galaxy_H2mass,
                                    obj.simulation.boxsize.d)

        #if h.GroupID < 3: 
        #    for i in range(0,len(internal_galaxy_index_list)):
        #        print(i,np.log10(galaxy_HImass[i]+1.),np.log10(galaxy_H2mass[i]+1.),np.log10(h.masses['total'].d))

    ## assign galaxy HI & H2 masses to respective galaxies
    for i in range(0,ngalaxies):
        obj.galaxies[i].masses['HI'] = obj.yt_dataset.quan(galaxy_HImass[i], obj.units['mass'])
        obj.galaxies[i].masses['H2'] = obj.yt_dataset.quan(galaxy_H2mass[i], obj.units['mass'])
        #if i<20: print(i,np.log10(galaxy_HImass[i]+1.),np.log10(galaxy_H2mass[i]+1.),np.log10(obj.galaxies[i].masses['stellar'].d),obj.galaxies[i].sfr)

    return HImass,H2mass

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def assign_halo_gas_to_galaxies(
        ## internal to the halo
        int haloID,
        np.ndarray[np.float64_t,ndim=2]   internal_galaxy_pos,
        np.ndarray[np.float64_t,ndim=1]   internal_galaxy_mass,
        np.ndarray[np.int32_t,ndim=1]     internal_galaxy_index_list,
        ## global values
        np.int64_t[:]     galaxy_glist,
        np.int64_t[:]     halo_glist,
        np.int64_t[:]     idlist,
        np.float64_t[:,:] gpos,
        np.float64_t[:]   HImass,
        np.float64_t[:]   H2mass,
        np.float64_t[:]   galaxy_HImass,
        np.float64_t[:]   galaxy_H2mass,
        double boxsize
):    
    """Function to assign halo gas to galaxies.

    When we assign galaxies in CAESAR, we only consider dense gas.
    But when considering HI gas however, it is often desirable to
    also consider low-density gas 'outside' of the galaxy.  This
    function calculates the mass weighted distance to each galaxy
    within a given halo and assigns low-density gas to the 'nearest'
    galaxy.

    Typically called from :func:`hydrogen_mass_calc.hydrogen_mass_calc`.

    """
    cdef int n_internal_galaxies = internal_galaxy_mass.shape[0]

    cdef int i,j,k,gas_galaxy_index,max_index
    cdef double d2,mwd,max_mwd,d2_at_max

    for i in range(0,len(HImass)):
        if HImass[i] == 0. and H2mass[i] == 0.: continue  # no mass to assign
        if halo_glist[idlist[i]] != haloID: continue  # not in this halo
        max_index = 0
        max_mwd   = 0.0
        d2_at_max   = 0.0
        for k in range(0,n_internal_galaxies):
            d2 = ( periodic(gpos[i,0] - internal_galaxy_pos[k,0], 0.5*boxsize, boxsize)**2 +
                   periodic(gpos[i,1] - internal_galaxy_pos[k,1], 0.5*boxsize, boxsize)**2 +
                   periodic(gpos[i,2] - internal_galaxy_pos[k,2], 0.5*boxsize, boxsize)**2 )
            mwd = internal_galaxy_mass[k] / d2
            if mwd > max_mwd:
                    max_mwd   = mwd
                    d2_at_max   = d2
                    max_index = k
        galaxy_HImass[internal_galaxy_index_list[max_index]] += HImass[i]
        galaxy_H2mass[internal_galaxy_index_list[max_index]] += H2mass[i]
        #/if haloID == 0: print(i,max_index,internal_galaxy_index_list[max_index],np.sqrt(d2_at_max),max_mwd,np.log10(HImass[i]),np.log10(galaxy_HImass[internal_galaxy_index_list[max_index]]+1.))
    

cdef double periodic(double x, double halfbox, double boxsize):
    if x < -halfbox:
        x += boxsize
    if x >  halfbox:
        x -= boxsize
    return x


''' Romeel's new functions to do the HI/H2 mass calculation, as well as aperture masses '''
''' Everything above this is defunct after the upgrade '''


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def get_HIH2_masses(galaxies,aperture=30,rho_thresh=0.13):

    # get HI and H2 particle masses
    from caesar.group import collate_group_ids
    from caesar.property_manager import has_property

    _, grpids, gid_bins = collate_group_ids(galaxies.obj.halo_list,'gas',galaxies.obj.simulation.ngas)
    if ('compute_selfshielding' in galaxies.obj._kwargs and galaxies.obj._kwargs['compute_selfshielding']) or not has_property(galaxies.obj, 'gas', 'fh2'):
        compute_selfshield(galaxies.obj,grpids,rho_thresh)

    # set up mass computation
    galpos = np.asarray([i.pos for i in galaxies.obj.galaxy_list], dtype=np.float64)
    galmass = np.asarray([i.masses['total'] for i in galaxies.obj.galaxy_list], dtype=np.float64)
    cdef:
        ## global quantities
        int         nhalo = len(galaxies.obj.halo_list)
        int         ngal = len(galaxies.obj.galaxy_list)
        int         npart = len(grpids)
        double      XH = galaxies.obj.simulation.XH
        long int[:] hid_bins = gid_bins   # starting indexes of particle IDs in each halo
        double[:,:] galaxy_pos = galpos
        double[:]   galaxy_mass = galmass
        float[:,:]  gas_pos = galaxies.obj.data_manager.pos[grpids]
        float[:]    gas_mass = galaxies.obj.data_manager.mass[grpids]
        float[:]    HImass = galaxies.obj.data_manager.gfHI[grpids]
        float[:]    H2mass = galaxies.obj.data_manager.gfH2[grpids]
        int[:]      galaxy_indexes = np.zeros(ngal,dtype=np.int32)
        int[:]      galind_bins = np.zeros(nhalo+1,dtype=np.int32)
        ## general variables
        double      rho_th = rho_thresh        # atoms/cm^3
        int         ih,ig,istart,iend,igstart,igend
        double      Lbox = galaxies.obj.simulation.boxsize.d
        double      apert2 = aperture*aperture
        double      myHI, myH2
        ## things to compute
        double[:]   galaxy_HImass = np.zeros(ngal)
        double[:]   apert_HImass = np.zeros(ngal)
        double[:]   apert_H2mass = np.zeros(ngal)
        double[:]   halo_HImass = np.zeros(nhalo)
        double[:]   halo_H2mass = np.zeros(nhalo)

    memlog('Doing HI/H2 calculation for %d galaxies in %d halos'%(ngal,nhalo))
    # set up list of galaxies to process, associated to halos
    ngal = 0
    for ih in range(nhalo):
        for ig in range(len(galaxies.obj.halo_list[ih].galaxy_index_list)):
            galaxy_indexes[ngal+ig] = galaxies.obj.halo_list[ih].galaxy_index_list[ig]
        ngal += len(galaxies.obj.halo_list[ih].galaxy_index_list)
        galind_bins[ih+1] = ngal
    assert ngal==len(galaxies.obj.galaxy_list),"Assertion failed in galaxy counts: %d != %d"%(ngal,len(galaxies.obj.galaxy_list))

    # compile HI and H2 masses for galaxies in halos
    for ih in prange(npart,nogil=True):
        HImass[ih] *= XH * gas_mass[ih]
        H2mass[ih] *= XH * gas_mass[ih]

    for ih in prange(nhalo,nogil=True):
    #for ih in range(nhalo):
        istart = hid_bins[ih]
        iend = hid_bins[ih+1]
        for ig in range(istart,iend):
            halo_HImass[ih] += HImass[ig]
        igstart = galind_bins[ih]
        igend = galind_bins[ih+1]
        if igstart < igend:
            _get_galaxy_hydrogen_masses(igstart, igend, istart, iend, galaxy_pos, galaxy_mass, gas_pos, HImass, H2mass, Lbox, galaxy_HImass, apert_HImass, apert_H2mass, apert2)

    # fill galaxy and halo lists
    for ih in range(nhalo):
        galaxies.obj.halo_list[ih].masses['HI'] = galaxies.obj.yt_dataset.quan(halo_HImass[ih], galaxies.obj.units['mass'])
    apert_str = '%dkpc'%aperture
    for ig in range(ngal):
        galaxies.obj.galaxy_list[ig].masses['HI'] = galaxies.obj.yt_dataset.quan(galaxy_HImass[ig], galaxies.obj.units['mass'])
        galaxies.obj.galaxy_list[ig].masses['HI_%s'%(apert_str)] = galaxies.obj.yt_dataset.quan(apert_HImass[ig], galaxies.obj.units['mass'])
        galaxies.obj.galaxy_list[ig].masses['H2_%s'%(apert_str)] = galaxies.obj.yt_dataset.quan(apert_H2mass[ig], galaxies.obj.units['mass'])
        #if ig < 10: print(ig,np.log10(galaxies.obj.galaxy_list[ig].masses['HI']),np.log10(galaxies.obj.galaxy_list[ig].masses['H2']), np.log10(galaxies.obj.galaxy_list[ig].masses['HI_30kpc']), np.log10(galaxies.obj.galaxy_list[ig].masses['H2_30kpc']))

    return 


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _get_galaxy_hydrogen_masses(int igstart, int igend, int istart, int iend, double[:,:] galaxy_pos, double[:] galaxy_mass, float[:,:] gpos, float[:] HImass, float[:] H2mass, float Lbox, double[:] galaxy_HImass, double[:] apert_HImass, double[:] apert_H2mass, double apert2) nogil:
    """Function to assign halo gas to galaxies.

    When we assign galaxies in CAESAR, we only consider dense gas.
    But when considering HI gas however, it is often desirable to
    also consider low-density gas 'outside' of the galaxy.  This
    function calculates the mass weighted distance to each galaxy
    within a given halo and assigns low-density gas to the 'nearest'
    galaxy.
    """

    cdef int i,j,k,max_index
    cdef double d2,mwd,max_mwd
    cdef double dx[3]
    cdef int ndim = 3

    for i in range(istart,iend):
        if HImass[i] == 0. and H2mass[i] == 0.: continue  # no mass to assign
        max_index = 0
        max_mwd   = 0.0
        for j in range(igstart,igend):
            d2 = 0.0
            for k in range(ndim):
                dx[k] = c_fabs(gpos[i,k] - galaxy_pos[j,k])
                if dx[k] > 0.5*Lbox: 
                    dx[k] = Lbox - dx[k]
                d2 += dx[k]*dx[k]
            mwd = galaxy_mass[j] / d2
            if mwd > max_mwd:
                    max_mwd = mwd
                    max_index = j
            if d2 < apert2:
                apert_HImass[j] += HImass[i]
                apert_H2mass[j] += H2mass[i]
        galaxy_HImass[max_index] += HImass[i]

    return


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def compute_selfshield(caesar_obj,grpids,rho_thresh):
    # Use Rahmati+13 to get self-shielded HI fractions, and Leroy+08 to get H2 fractions
    from caesar.property_manager import get_property

    cdef double gamma_HI
    from .treecool_data import UVB
    uvb = UVB['FG11']

    if np.log10(caesar_obj.simulation.redshift + 1.0) > uvb['logz'][len(uvb['logz'])-1]:
        gamma_HI = 0.0
    else:
        gamma_HI = np.interp(np.log10(caesar_obj.simulation.redshift + 1.0),
                             uvb['logz'],uvb['gH0'])

    cdef:
        ## density thresholds in atoms/cm^3
        double    low_rho_thresh = 0.001       # atoms/cm^3
        double    rho_th      = rho_thresh
        double    XH          = caesar_obj.simulation.XH
        double    FSHIELD     = 0.99
        double    redshift    = caesar_obj.simulation.redshift
        ## Leroy et al 2008, Fig17 (THINGS) Table 6 ##
        double    P0BLITZ     = 1.7e4
        double    ALPHA0BLITZ = 0.8
        ## Popping+09 constants (equation 7)
        int    i
        double beta, xi, nHss, fgamma_HI, C, Rmol, fHI, fH2, cold_phase_massfrac 
        double a  = 7.982e-11                  # cm^3/s
        double b  = 0.7480
        double T0 = 3.148                      # K
        double T1 = 7.036e5                    # K
        double sigHI     = 3.27e-18 * (1.0+redshift)**(-0.2)
        double fbaryon   = caesar_obj.simulation.omega_baryon / caesar_obj.simulation.omega_matter
        double nHss_part = 6.73e-3 * (sigHI/2.49e-18)**(-2./3.) * (fbaryon / 0.17)**(-1./3.)
        ## gas quantities
        float[:]   gmass = caesar_obj.data_manager.mass[grpids]
        float[:]   gnh = caesar_obj.data_manager.gnh[grpids]
        float[:]   gtemp = caesar_obj.data_manager.gT[grpids]
        int         npart = len(gmass)
        # things to compute
        float[:]   gfHI = caesar_obj.data_manager.gfHI[grpids]
        float[:]   gfH2 = np.zeros(npart,dtype=MY_DTYPE)

    # determine HI, H2 fractions for all particles
    for i in prange(npart,nogil=True):
        fHI = gfHI[i]
        fH2 = 0.0
        ## low density non-self shielded gas
        if gnh[i] < rho_th:
            ## Rahmati+13 equations 2, 1
            nHss      = nHss_part * (gtemp[i] * 1.0e-4)**0.17 * (gamma_HI * 1.0e12)**(2./3.)
            if nHss-1. < 1.e-4: continue  # no significant self-shielding adjustment needed; skip
            fgamma_HI = 0.98 * (1.0 + (gnh[i] / nHss)**(1.64))**(-2.28) + 0.02 * (1.0 + gnh[i] / nHss)**(-0.84)
            ### Popping+09 equations 3, 7, 4
            beta     = a / (c_sqrt(gtemp[i]/T0) *
                            (1.0 + c_sqrt(gtemp[i]/T0))**(1.0-b) *
                            (1.0 + c_sqrt(gtemp[i]/T1))**(1.0+b))   # cm^3/s
            ## Popping+09 equations 6, 5
            C = gnh[i] * beta / (gamma_HI * fgamma_HI)
            fHI = (2.0 * C + 1.0 - c_sqrt((2.0*C+1.0)*(2.0*C+1.0) - 4.0 * C * C)) / (2.0*C)

        if gnh[i] >= rho_th:  # dense gas
            cold_phase_massfrac = (1.0e8 - gtemp[i])/1.0e8
            Rmol = (gnh[i] * gtemp[i] / P0BLITZ)**ALPHA0BLITZ
            fHI  = FSHIELD * cold_phase_massfrac / (1.0 + Rmol)
            fH2  = FSHIELD - fHI

        gfHI[i]    = fHI 
        gfH2[i]    = fH2 * XH * gmass[i]

    caesar_obj.data_manager.gfHI[grpids] = caesar_obj.yt_dataset.arr(gfHI, '')
    caesar_obj.data_manager.gfH2[grpids] = caesar_obj.yt_dataset.arr(gfH2, '')

    memlog('Done assigning HI and H2 masses to particles')
    return 

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def get_aperture_masses(galaxies,aperture=30):
    ''' Compute aperture masses in various quantities.  The aperture should be specified
    in the same units as the galaxy positions in data_manager, usually ckpc '''

    from caesar.group import collate_group_ids
    from caesar.property_manager import has_property,ptype_ints

    _, grpids, gid_bins = collate_group_ids(galaxies.obj.halo_list,'all',galaxies.obj.simulation.ntot)

    # set up mass computation
    galpos = np.asarray([i.pos for i in galaxies.obj.galaxy_list], dtype=np.float64)
    galmass = np.asarray([i.masses['total'] for i in galaxies.obj.galaxy_list], dtype=np.float64)
    ptype_array = []
    for pt in ['gas','star','dm']:  # this is the ordering for ptypes to be computed
        ptype_array.append(ptype_ints[pt])
    ptype_array = np.array(ptype_array,dtype=np.int32)

    cdef:
        ## global quantities
        int         nhalo = len(galaxies.obj.halo_list)
        int         ngal = len(galaxies.obj.galaxy_list)
        int         npart = len(grpids)
        long int[:] hid_bins = gid_bins   # starting indexes of particle IDs in each halo
        double[:,:] galaxy_pos = galpos
        double[:]   galaxy_mass = galmass
        float[:,:] ppos = galaxies.obj.data_manager.pos[grpids]
        float[:]   pmass = galaxies.obj.data_manager.mass[grpids]
        int[:]      ptype = galaxies.obj.data_manager.ptype[grpids]
        int[:]      galaxy_indexes = np.zeros(ngal,dtype=np.int32)
        int[:]      galind_bins = np.zeros(nhalo+1,dtype=np.int32)
        ## general variables
        int         ih,ig,istart,iend,igstart,igend
        double      Lbox = galaxies.obj.simulation.boxsize.d
        double      apert2 = aperture*aperture
        int[:]      ptypes = ptype_array
        long int[:] gas_index = np.zeros(npart,dtype=np.int64)
        ## things to compute
        double[:]   galaxy_gmass = np.zeros(ngal)
        double[:]   galaxy_smass = np.zeros(ngal)
        double[:]   galaxy_dmmass = np.zeros(ngal)

    memlog('Doing aperture mass calculation for %d galaxies in %d halos'%(ngal,nhalo))
    # set up list of galaxies to process, associated to halos
    ngal = 0
    for ih in range(nhalo):
        for ig in range(len(galaxies.obj.halo_list[ih].galaxy_index_list)):
            galaxy_indexes[ngal+ig] = galaxies.obj.halo_list[ih].galaxy_index_list[ig]
        ngal += len(galaxies.obj.halo_list[ih].galaxy_index_list)
        galind_bins[ih+1] = ngal
    assert ngal==len(galaxies.obj.galaxy_list),"Assertion failed in galaxy counts: %d != %d"%(ngal,len(galaxies.obj.galaxy_list))

    # associate gas indexes to overall indexes, so we can properly index HImass and H2mass
    ih = 0
    for ig in range(npart):
        if ptype[ig] == ptypes[0]: 
            gas_index[ig] = ih
            ih += 1

    #for ih in prange(nhalo,nogil=True):
    for ih in range(nhalo):
        istart = hid_bins[ih]
        iend = hid_bins[ih+1]
        igstart = galind_bins[ih]
        igend = galind_bins[ih+1]
        if igstart < igend:
            get_galaxy_aperture_masses(igstart, igend, istart, iend, galaxy_pos, pmass, ppos, ptype, Lbox, galaxy_gmass, galaxy_smass, galaxy_dmmass, ptypes, apert2, gas_index)

    # fill galaxy and halo lists
    apert_str = '%dkpc'%aperture
    for ig in range(ngal):
        galaxies.obj.galaxy_list[ig].masses['gas_%s'%(apert_str)] = galaxies.obj.yt_dataset.quan(galaxy_gmass[ig], galaxies.obj.units['mass'])
        galaxies.obj.galaxy_list[ig].masses['stellar_%s'%(apert_str)] = galaxies.obj.yt_dataset.quan(galaxy_smass[ig], galaxies.obj.units['mass'])
        galaxies.obj.galaxy_list[ig].masses['dm_%s'%(apert_str)] = galaxies.obj.yt_dataset.quan(galaxy_dmmass[ig], galaxies.obj.units['mass'])
        #if ig < 10: print(ig,np.log10(galaxies.obj.galaxy_list[ig].masses['stellar']),galaxies.obj.galaxy_list[ig].sfr, np.log10(galaxies.obj.galaxy_list[ig].masses['stellar_30kpc']), np.log10(galaxies.obj.galaxy_list[ig].masses['dm_30kpc']))
    memlog('filled galaxy_lists')

    return 

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef void get_galaxy_aperture_masses(int igstart, int igend, int istart, int iend, double[:,:] galaxy_pos, float[:] pmass, float[:,:] ppos, int[:] ptype, float Lbox, double[:] galaxy_gmass, double[:] galaxy_smass, double[:] galaxy_dmmass, int[:] ptypes, float apert2, long int[:] gas_index) nogil:
    """Function to compute aperture masses for galaxies within halos.
    Note that this only looks at mass within a galaxy's halo, so it may miss some mass
    particularly for satellites on the outskirts of the halo.
    Should be good for centrals, though!

    """

    cdef int i,j,k
    cdef double d2
    cdef double dx[3]

    for i in range(istart,iend):
        for j in range(igstart,igend):
            d2 = 0.0
            for k in range(3):
                dx[k] = c_fabs(ppos[i,k] - galaxy_pos[j,k])
                if dx[k] > 0.5*Lbox: 
                    dx[k] = Lbox - dx[k]
                d2 += dx[k]*dx[k]
            if d2 < apert2:
                if ptype[i] == ptypes[0]: galaxy_gmass[j] += pmass[i]
                if ptype[i] == ptypes[1]: galaxy_smass[j] += pmass[i]
                if ptype[i] == ptypes[2]: galaxy_dmmass[j] += pmass[i]
                d2 = apert2

    return

