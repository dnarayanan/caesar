import numpy as np
from yt.funcs import mylog
from yt.units.yt_array import YTQuantity
cimport numpy as np
cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

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

'''
# Romeel's try -- TOO SLOW
@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def periodic_accel_(
        double mass,
        np.ndarray[np.float64_t,ndim=1]   x0,
        np.ndarray[np.float64_t,ndim=1]   x1,
        double boxsize
    ):
    cdef double d2=0.,dx[3]
    for i in range(3):
        dx[i] = abs(x0[i]-x1[i])
        if dx[i] > 0.5*boxsize: dx[i] = boxsize-dx[i]
        d2 += dx[i]*dx[i]
    return mass / d2

def get_HI_masses(obj):
    """ Compute HI masses in each halo and all its member galaxies, taking 
        HI fractions directly from snapshot.  Assigns particle's HI to galaxies based
        on whichever galaxy in the halo is highest in Mgal/R^2 for that particle."""
    from caesar.property_manager import get_property, has_property

    mylog.info('Calculating HI masses')
    nHlim_SF = 0.13  # in atom/cm^3. above this, HI frac is set to 1-fH2
    nHlim_HI = 0.001  #in atom/cm^3. below this, HI frac is set to zero for counting purposes
    for g in obj.galaxies:
        g.masses['HI'] = obj.yt_dataset.quan(0.0, obj.units['mass'])  # initialize

    ## loop over halos
    from yt.extern.tqdm import tqdm
    cdef double accel
    for h in tqdm(obj.halos,desc='Halo:'):
        halo_glist = obj.data_manager.glist[h.glist]  # particle IDs in halo
        gmass = obj.data_manager.mass[halo_glist]  # masses of halo particles
        gpos = obj.data_manager.pos[halo_glist]  # pos of halo particles
        gnH = get_property(obj, 'rho', 'gas').in_cgs().d[halo_glist] * obj.simulation.XH / 1.67262178e-24
        gfHI = get_property(obj, 'nh', 'gas')[halo_glist] # HI frac of halo particles
        gfH2 = get_property(obj, 'fH2', 'gas')[halo_glist] # H2 frac of halo particles
        gfHI = np.where(gnH>nHlim_SF,1.-gfH2,gfHI)
        gfHI = np.where(gnH<nHlim_HI,0.,gfHI)
        #if h.GroupID<3: print('gfHI:',h.GroupID,gfHI[gnH>0.13]+gfH2[gnH>0.13])
        gfHI = gfHI * gmass
        h.masses['HI'] = obj.yt_dataset.quan(np.sum(gfHI), obj.units['mass'])

        ## load galaxies.  if no galaxies, we're done
        halo_gals = [obj.galaxies[i] for i in h.galaxy_index_list]
        if len(halo_gals) == 0:
            continue

        ## for dense gas (with H2), we can add to its own galaxy
        #MHI = np.zeros(len(halo_gals))
        #for ih,hgal in enumerate(halo_gals):
        #    galmass = obj.data_manager.mass[hgal.glist]
        #    galfHI = get_property(obj, 'nh', 'gas')[hgal.glist] # HI frac of halo particles
        #    galfH2 = get_property(obj, 'fH2', 'gas')[hgal.glist] # H2 frac of halo particles
        #    MHI[ih] = np.sum(galfHI[galfH2>0]*galmass[galfH2>0])

        galpos = [hg.pos.d for hg in halo_gals]
        galmass = [hg.mass.d for hg in halo_gals]
        from scipy.spatial.distance import cdist
        invdist2 = np.reciprocal(cdist(gpos,galpos,'sqeuclidean'))
        #if h.GroupID < 3: print(np.shape(invdist2),np.shape(galpos),len(galmass))
        MHI = np.zeros(len(halo_gals))
        for ig in range(len(halo_gals)):
            MHI[ig] = np.sum([gfHI[i] for i in range(len(gfHI)) if ig == np.argmax(galmass*invdist2[i])])

        ## loop over particles in halo, assign to galaxies, add up HI masses
        #indices = np.zeros(len(gmass),dtype=np.int)
        #for ip in range(len(gmass)):
        #    if gfHI[ip] == 0.: ip = 0
        #    else: indices[ip] = np.argmax( [periodic_accel_(hgal.mass.d,hgal.pos.d,gpos[ip],obj.simulation.boxsize.d) for hgal in halo_gals] )   # index of galaxy that particle feels greatest acceleration from
        for i in range(len(halo_gals)):
            if h.GroupID < 3: 
                print(i,h.galaxy_index_list[i],np.log10(MHI[i]+1.),np.log10(halo_gals[i].mass),np.log10(h.masses['total'].d),np.log10(h.masses['HI'].d+1.))
            obj.galaxies[h.galaxy_index_list[i]].masses['HI'] = obj.yt_dataset.quan(MHI[i], obj.units['mass'])
    return 
'''

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
