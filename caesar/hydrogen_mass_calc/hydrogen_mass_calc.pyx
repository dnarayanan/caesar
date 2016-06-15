import numpy as np
from yt.funcs import mylog
from yt.units.yt_array import YTQuantity
cimport numpy as np
cimport cython

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

    tmp_str = 'Calculating HI/H2 masses for'
    cdef bint all_gas = 0
    if 'calculate_H_for_all_gas' in kwargs and kwargs['calculate_H_for_all_gas']:
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
    cdef np.int32_t[:] halo_glist   = np.array(obj.global_particle_lists.halo_glist,dtype=np.int32)
    cdef np.int32_t[:] galaxy_glist = np.array(obj.global_particle_lists.galaxy_glist,dtype=np.int32)

    ## gas properties
    from caesar.property_manager import get_property, has_property
    cdef np.float64_t[:,:] gpos  = obj.data_manager.pos[obj.data_manager.glist]
    cdef np.float64_t[:]   gmass = obj.data_manager.mass[obj.data_manager.glist]
    cdef np.float64_t[:]   grhoH = get_property(obj, 'rho', 'gas').in_cgs().d * XH / proton_mass
    cdef np.float64_t[:]   gtemp = obj.data_manager.gT.to('K').d
    cdef np.float64_t[:]    gsfr = obj.data_manager.gsfr.d
    cdef np.float64_t[:]     gnh = get_property(obj, 'nh', 'gas').d
    cdef np.float64_t[:]    gfH2
        
    cdef int i
    cdef int nhalos    = len(obj.halos)
    cdef int ngalaxies = len(obj.galaxies)
    cdef int ngas      = len(gmass)
        
    cdef bint H2_data_present = 0
    if has_property(obj, 'gas', 'fH2'):
        gfH2  = get_property(obj, 'fH2', 'gas').d
        H2_data_present = 1
    else:
        mylog.warning('Could not locate molecular fraction data. Estimating via Leroy+08')

    cdef np.ndarray[np.float64_t,ndim=1] HImass = np.zeros(ngas)
    cdef np.ndarray[np.float64_t,ndim=1] H2mass = np.zeros(ngas)

    cdef double fHI, fH2
    cdef double cold_phase_massfrac, Rmol
    
    ## calculate HI & H2 mass for each valid gas particle
    for i in range(0,ngas):

        ## skip if not assigned to a halo
        if not all_gas and halo_glist[i] < 0:
            continue

        ## skip if density is too low
        if not all_gas and grhoH[i] < low_rho_thresh:
            continue

        fHI = gnh[i]
        fH2 = 0.0

        ## low density non-self shielded gas
        if grhoH[i] < rho_thresh:
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

        ## high density gas when no H2 data is present
        ## estimate H2 via Leroy+08
        elif not H2_data_present:
            cold_phase_massfrac = (1.0e8 - gtemp[i])/1.0e8
            Rmol = (grhoH[i] * gtemp[i] / P0BLITZ)**ALPHA0BLITZ
            fHI  = FSHIELD * cold_phase_massfrac / (1.0 + Rmol)
            fH2  = FSHIELD - fHI

        ## high density gas when H2 data is present
        else:
            fH2 = gfH2[i]
            fHI = 1.0 - fH2            
            
            if fHI < 0.0:
                fHI = 0.0

        HImass[i]    = fHI * gmass[i]
        H2mass[i]    = fH2 * gmass[i]

    ## tally halo HI & H2 masses
    cdef np.float64_t[:] halo_HImass = np.zeros(nhalos)
    cdef np.float64_t[:] halo_H2mass = np.zeros(nhalos)
    for i in range(0,ngas):
        if halo_glist[i] < 0:
            continue
        halo_HImass[halo_glist[i]] += HImass[i]
        halo_H2mass[halo_glist[i]] += H2mass[i]

    for i in range(0,nhalos):
        obj.halos[i].masses['HI'] = obj.yt_dataset.quan(0.0, obj.units['mass'])
        obj.halos[i].masses['H2'] = obj.yt_dataset.quan(0.0, obj.units['mass'])
        
    ## assign halo HI & H2 masses to respective halos
    for i in range(0,nhalos):
        obj.halos[i].masses['HI'] += obj.yt_dataset.quan(halo_HImass[i], obj.units['mass'])
        obj.halos[i].masses['H2'] += obj.yt_dataset.quan(halo_H2mass[i], obj.units['mass'])

    if len(obj.galaxies) == 0:
        return HImass,H2mass
        
    ## galaxy calc
    cdef np.ndarray[np.float64_t,ndim=2] galaxy_pos  = np.array([s.pos for s in obj.galaxies])
    cdef np.ndarray[np.float64_t,ndim=1] galaxy_mass = np.array([s.masses['total'] for s in obj.galaxies])

    cdef np.float64_t[:] galaxy_HImass = np.zeros(ngalaxies)
    cdef np.float64_t[:] galaxy_H2mass = np.zeros(ngalaxies)

    cdef list halos = obj.halos
    cdef object h

    cdef int gas_galaxy_index, max_index, n_internal_galaxies,ngas_internal
    cdef double d2
    cdef double boxsize = obj.simulation.boxsize.d
    cdef double halfbox = obj.simulation.boxsize.d / 2.0

    for h in halos:

        ## check if any galaxies are contained
        if len(h.galaxy_index_list) == 0:
            continue

        internal_galaxy_pos  = galaxy_pos[h.galaxy_index_list]
        internal_galaxy_mass = galaxy_mass[h.galaxy_index_list]
        internal_galaxy_index_list = np.array(h.galaxy_index_list,dtype=np.int32)
        internal_glist = np.array(h.glist,dtype=np.int32)

        n_internal_galaxies  = internal_galaxy_mass.shape[0]
        ngas_internal        = len(h.glist)

        assign_halo_gas_to_galaxies(internal_galaxy_pos,
                                    internal_galaxy_mass,
                                    internal_glist,
                                    internal_galaxy_index_list,
                                    galaxy_glist,
                                    grhoH, gpos,
                                    galaxy_HImass, galaxy_H2mass,
                                    HImass, H2mass,
                                    low_rho_thresh,
                                    boxsize,halfbox)

    for i in range(0,ngalaxies):
        obj.galaxies[i].masses['HI'] = obj.yt_dataset.quan(0.0, obj.units['mass'])
        obj.galaxies[i].masses['H2'] = obj.yt_dataset.quan(0.0, obj.units['mass'])
        
    ## assign galaxy HI & H2 masses to respective galaxies
    for i in range(0,ngalaxies):
        obj.galaxies[i].masses['HI'] += obj.yt_dataset.quan(galaxy_HImass[i], obj.units['mass'])
        obj.galaxies[i].masses['H2'] += obj.yt_dataset.quan(galaxy_H2mass[i], obj.units['mass'])

    return HImass,H2mass

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def assign_halo_gas_to_galaxies(
        ## internal to the halo
        np.ndarray[np.float64_t,ndim=2]   internal_galaxy_pos,
        np.ndarray[np.float64_t,ndim=1]   internal_galaxy_mass,
        np.ndarray[np.int32_t,ndim=1]     internal_glist,
        np.ndarray[np.int32_t,ndim=1]     internal_galaxy_index_list,
        ## global values
        np.int32_t[:]     galaxy_glist,
        np.float64_t[:]   grhoH,
        np.float64_t[:,:] gpos,
        np.float64_t[:]   galaxy_HImass,
        np.float64_t[:]   galaxy_H2mass,
        np.float64_t[:]   HImass,
        np.float64_t[:]   H2mass,
        double low_rho_thresh,
        double boxsize,
        double halfbox
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
    cdef int ngas_internal       = internal_glist.shape[0]
    cdef int n_internal_galaxies = internal_galaxy_mass.shape[0]

    cdef int i,j,k,gas_galaxy_index,max_index
    cdef double d2,mwd,max_mwd

    for j in range(0,ngas_internal):
        i = internal_glist[j]
        if grhoH[i] < low_rho_thresh:
            continue

        gas_galaxy_index = galaxy_glist[i]
        max_index = 0
        max_mwd   = 0.0

        if gas_galaxy_index < 0:
            for k in range(0,n_internal_galaxies):
                d2 = ( periodic(gpos[i,0] - internal_galaxy_pos[k,0], halfbox, boxsize)**2 +
                       periodic(gpos[i,1] - internal_galaxy_pos[k,1], halfbox, boxsize)**2 +
                       periodic(gpos[i,2] - internal_galaxy_pos[k,2], halfbox, boxsize)**2 )
                mwd = internal_galaxy_mass[k] / d2
                if mwd > max_mwd:
                    max_mwd   = mwd
                    max_index = k
            gas_galaxy_index = internal_galaxy_index_list[max_index]

        galaxy_HImass[gas_galaxy_index] += HImass[i]
        galaxy_H2mass[gas_galaxy_index] += H2mass[i]


cdef double periodic(double x, double halfbox, double boxsize):
    if x < -halfbox:
        x += boxsize
    if x >  halfbox:
        x -= boxsize
    return x
