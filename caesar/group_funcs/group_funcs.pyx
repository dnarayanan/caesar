import six
import numpy as np
cimport numpy as np
import sys
cimport cython
from cython.parallel import prange, threadid
from caesar.utils import memlog
from yt.funcs import mylog
from caesar.property_manager import MY_DTYPE

""" ================================================ """
""" IMPORT C LIBRARY ROUTINES NEEDED FOR COMPUTATION """
""" ================================================ """
from libc.stdio cimport printf, fflush, stderr, stdout
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt as c_sqrt, fabs as c_fabs, sin as c_sin, cos as c_cos, atan2 as c_atan2, acos as c_acos, log10 as c_log10
cdef extern from "math.h":
    double sqrt(double x)
    double M_PI
cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size, int(*compar)(const_void *, const_void *)) nogil

ctypedef struct part_struct:  # structure to hold particle info for particles within single group
    float m  # mass
    float r  # radius
    float v[3]  # velocities (for vel disp)
    float x[3]  # positions with respect to center
    int t  # type
    #long long i  # index, for sorting purposes

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef int isin(int val, int[:] arr) nogil:
    cdef int i
    cdef int size = len(arr)
    for i in range(size):
        if arr[i] == val:
            return 1
    return 0

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef int mycmp(const_void * pa, const_void * pb) nogil:  # qsort comparison function
    cdef float a = ((<part_struct *>pa).r)
    cdef float b = ((<part_struct *>pb).r)
    if a < b:
        return -1
    elif a > b:
        return 1
    else:
        return 0

""" ======================================================= """
""" AUXILIARY ROUTINES TO COMPUTE SPECIFIC GROUP PROPERTIES """
""" ======================================================= """
@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef void nogil_CoM_quants(int ig, float[:,:] pos, float[:,:] vel, float[:] mass, float[:] pot, int[:] ptype, int[:] group_ptypes, float[:] grp_mtot, long long istart, long long iend, int ndim, float Lbox, float[:,:] grp_pos, float[:,:] grp_vel, float[:,:] grp_minpotpos, float[:,:] grp_minpotvel) nogil:
    """ Computes center-of-mass position and velocity, as well as minimum potential position.

    ig: group index
    pos, vel, mass, pot: positions, velocities, masses, potentials of particles in group
    grp_mtot: total mass of group
    istart,iend: starting, ending indexes for group particles
    ndim: number of dimensions, usually 3
    Lbox: periodic box size in units of pos
    grp_pos, grp_vel: quantities to be computed

    Returns minpotpart, index of particle at minimum of potential
    """
    cdef int ip
    cdef long long i, minpotpart = -1
    cdef float minpot = 1.e30
    cdef float[3] mypos

    for ip in range(ndim):
        for i in range(istart,iend):
            if isin(ptype[i], group_ptypes): # only selected types for calculation
                # handle periodicity by keeping all particles close to the first particle
                mypos[ip] = pos[i,ip]
                if mypos[ip] - pos[istart,ip] > 0.5*Lbox:
                    mypos[ip] -= Lbox
                if mypos[ip] - pos[istart,ip] < -0.5*Lbox:
                    mypos[ip] += Lbox
                grp_pos[ig,ip] += mass[i] * mypos[ip]
                grp_vel[ig,ip] += mass[i] * vel[i,ip]
        grp_pos[ig,ip] /= grp_mtot[ig]
        grp_vel[ig,ip] /= grp_mtot[ig]
        # if the CoM pos ends up outside the box, periodically wrap it back in
        if grp_pos[ig,ip] > Lbox:
            grp_pos[ig,ip] -= Lbox
        if grp_pos[ig,ip] < -0:
            grp_pos[ig,ip] += Lbox
    for i in range(istart,iend):
        if isin(ptype[i], group_ptypes): # only selected types for calculation
            if pot[i] < minpot:
                minpotpart = i
                minpot = pot[i]
    for ip in range(ndim):
        grp_minpotpos[ig,ip] = pos[minpotpart,ip]  # position of minimum potential particle, of any type
        grp_minpotvel[ig,ip] = vel[minpotpart,ip]  # velocity of minimum potential particle, of any type

    return


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef void nogil_load_partinfo(float[:] mass, float[:,:] pos, float[:,:] vel, int[:] ptype, float[:] cent_pos, float[:] cent_vel, part_struct *pinfo, float Lbox, long long istart, long long iend, int ndim) nogil:
    """ Computes center-of-mass position and velocity, as well as minimum potential position.

    ig: group index
    pos, vel, mass, pot: positions, velocities, masses, potentials of particles in group
    grp_mtot: total mass of group
    Lbox: periodic box size in units of pos
    istart,iend: starting, ending indexes for group particles
    ndim: number of dimensions, usually 3
    pinfo: particle info structure to be filled
    """
    cdef int ip
    cdef long long i,j
    cdef float dx[3]

    for i in range(istart,iend):
        j = i - istart
        pinfo[j].r = 0.
        for ip in range(ndim):
            # handle periodicity by keeping all particles close to the central position
            dx[ip] = pos[i,ip] - cent_pos[ip]
            if dx[ip] < -0.5*Lbox:
                dx[ip] = Lbox + dx[ip]
            if dx[ip] > 0.5*Lbox:
                dx[ip] = Lbox - dx[ip]
            pinfo[j].r += dx[ip]*dx[ip]
            pinfo[j].v[ip] = vel[i,ip] - cent_vel[ip]  # velocity wrt cent_vel
            pinfo[j].x[ip] = dx[ip]   # position relative to cent_pos
        pinfo[j].r = c_sqrt(pinfo[j].r)
        pinfo[j].m = mass[i]
        pinfo[j].t = ptype[i]
        #pinfo[j].i = i

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef float nogil_half_mass_radius(part_struct *pinfo, float mtarget, int ip, int[:] group_ptypes, int npart) nogil:
    """Computes mass-enclosing radius for a set of particles.

    pinfo: struct holding mass, radii, ptypes of particles
    mtarget: target mass to accumulate to (e.g. 0.5*grp_mtot for half-mass)
    ip: particle type. ip=len(group_ptypes) does all baryons, ip=len(group_ptypes)+1 does all particles
        NOTE: This assumes that "baryons" means all particle types other than 1 and 2
    group_ptypes: particle types in this dataset
    npart: total number of particles in this group


    Returns the radius of particle where the cumulative mass exceeds the target mass
    """
    cdef int i,ngp
    cdef double cumulative_mass = 0.0
    cdef float r = 0.0

    ngp = len(group_ptypes)
    for i in range(npart):
        if (ip==ngp+1) | ((ip==ngp) & (pinfo[i].t!=1) & (pinfo[i].t!=2)) | ((ip<ngp) & (pinfo[i].t==group_ptypes[ip])):
            cumulative_mass += pinfo[i].m
            if cumulative_mass >= mtarget:
                r = pinfo[i].r
                break
    return r

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef float nogil_velocity_dispersions(part_struct *pinfo, int ip, int[:] group_ptypes, int npart, int ndim) nogil:
    """Computes mass-weighted velocity dispersion around center-of-mass velocity for given
    particle type(s).

    pinfo: struct holding mass, radii, ptypes of particles
    mtarget: target mass to accumulate to (e.g. 0.5*grp_mtot for half-mass)
    ip: particle type. ip=len(group_ptypes) does all baryons, ip=len(group_ptypes)+1 does all particles
        NOTE: This assumes that "baryons" means all particle types other than 1 and 2
    group_ptypes: particle types in this dataset
    npart: total number of particles in this group

    Returns the velocity dispersion
    """
    cdef int i, idim, ngp, npt=0
    cdef double mtot=0., vdisp = 0.
    cdef double vcom[3]

    # first find center-of-mass velocity for this particle type
    for idim in range(ndim):
        vcom[idim] = 0.
    ngp = len(group_ptypes)
    for i in range(npart):
        if (ip==ngp+1) | ((ip==ngp) & (pinfo[i].t!=1) & (pinfo[i].t!=2)) | ((ip<ngp) & (pinfo[i].t==group_ptypes[ip])):
            mtot += pinfo[i].m
            for idim in range(ndim):
                vcom[idim] += pinfo[i].m * pinfo[i].v[idim]
            npt += 1
    if npt < 3:  # not enough particle to get a dispersion
        return 0.0
    for idim in range(ndim):
        vcom[idim] /= mtot
    # compute dispersion around CoM velocity
    for i in range(npart):
        if (ip==ngp+1) | ((ip==ngp) & (pinfo[i].t!=1) & (pinfo[i].t!=2)) | ((ip<ngp) & (pinfo[i].t==group_ptypes[ip])):
            for idim in range(ndim):
                vdisp += (pinfo[i].v[idim] - vcom[idim])**2
    vdisp = c_sqrt(vdisp/npt)
    return vdisp

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef void nogil_angular_quants(part_struct *pinfo, int npart, int ip, int[:] group_ptypes, float[:] L) nogil:
    """ Compute angular quantities associated with rotating the galaxy along its angular mom vector

    pinfo: struct holding mass, radii, ptypes of particles
    npart: total number of particles in this group
    ip: particle type. ip=len(group_ptypes) does all baryons, ip=len(group_ptypes)+1 does all particles
        NOTE: This assumes that "baryons" means all particle types other than 1 and 2
    group_ptypes: particle types in this dataset
    L: 3-vector of angular momenta, plus components 4, 5, 6, 7 hold ALPHA, BETA, kinematic bulge-to-total, kappa_rot (from Sales+11)

    No return value; computed quantities loaded into L.
    """

    cdef int i,idim,ngp,npt=0
    cdef float p[3]
    cdef float x[3]
    cdef float v[3]
    cdef float e[3]
    cdef double Lmag,phi,theta,jx,jy,jz,rx,ry,rz,v2,vphi
    cdef double krot=0., ktot=0.
    cdef double m_tot=0., m_counterrot=0., jtot=0.

    # first find center-of-mass velocity for this particle type
    ngp = len(group_ptypes)
    for idim in range(7):
        L[idim] = 0.   # this stores Lx, Ly, Lz, ALPHA, BETA, BoverT, kappa_rot

    # count particles of this type to see if we have enough
    rz = 0.
    for i in range(npart):
        if (ip==ngp+1) | ((ip==ngp) & (pinfo[i].t!=1) & (pinfo[i].t!=2)) | ((ip<ngp) & (pinfo[i].t==group_ptypes[ip])):
            npt += 1
        rz += pinfo[i].x[0]
    if npt < 3:  # not enough particles of this type to compute angular quants
        return

    for i in range(npart):
        # select particle type to do: default is to do all
        if (ip==ngp+1) | ((ip==ngp) & (pinfo[i].t!=1) & (pinfo[i].t!=2)) | ((ip<ngp) & (pinfo[i].t==group_ptypes[ip])):
            npt += 1
            # load info for desired particle type
            for idim in range(3):
                p[idim] = pinfo[i].m * pinfo[i].v[idim]  # Note: pinfo.x and .v are w.r.t. group center
                x[idim] = pinfo[i].x[idim]
            # compute angular momentum vector
            L[0] += x[1]*p[2] - x[2]*p[1]
            L[1] += x[2]*p[0] - x[0]*p[2]
            L[2] += x[0]*p[1] - x[1]*p[0]

    # compute rotation angles, which rotates the galaxy to line up the z-dir with L
    Lmag = c_sqrt(L[0]*L[0]+L[1]*L[1]+L[2]*L[2])
    phi = c_atan2(L[1],L[0])
    theta = c_acos(L[2]/Lmag)
    e[0] = c_sin(theta) * c_cos(phi)
    e[1] = c_sin(theta) * c_sin(phi)
    e[2] = c_cos(theta)
    L[3] = c_atan2(L[1],L[2])  # this is ALPHA (yaw)
    nogil_rotator(e, L[3], 0.0)
    L[4] = c_atan2(e[0],e[2])  # this is BETA (pitch)

    # compute bulge-to-total ratio based on kinematic decomposition
    for i in range(npart):
        # select particle type to do: default is to do all
        if (ip==ngp+1) | ((ip==ngp) & (pinfo[i].t!=1) & (pinfo[i].t!=2)) | ((ip<ngp) & (pinfo[i].t==group_ptypes[ip])):
            # load info for desired particle type
            v2 = 0.
            for idim in range(3):
                p[idim] = pinfo[i].m * pinfo[i].v[idim]  # Note: pinfo.x and .v are w.r.t. group center
                x[idim] = pinfo[i].x[idim]
                v2 += pinfo[i].v[idim]*pinfo[i].v[idim]
            #nogil_rotator(x,L[3],L[4])  # rotate positions and momenta to align with L
            #nogil_rotator(p,L[3],L[4])
            jx = x[1]*p[2] - x[2]*p[1]
            jy = x[2]*p[0] - x[0]*p[2]
            jz = x[0]*p[1] - x[1]*p[0]
            # for bulge-to-total, add up mass of particle rotating against L
            if jx*L[0] + jy*L[1] + jz*L[2] < 0:
                m_counterrot += pinfo[i].m
            m_tot += pinfo[i].m
            # to compute kappa_rot, need j along L direction = j dot L / |L|
            jz = (jx*L[0] + jy*L[1] + jz*L[2]) / Lmag
            # also need distance of particle to L vector rz = r X L / |L|
            rx = x[1]*L[2] - x[2]*L[1]
            ry = x[2]*L[0] - x[0]*L[2]
            rz = x[0]*L[1] - x[1]*L[0]
            rz = c_sqrt(rx*rx+ry*ry+rz*rz)/Lmag
            # compute kappa_rot, which is fraction of KE in ordered rotation (Sales+11 eq 1)
            if rz>0: krot += 0.5*(jz/rz)**2/pinfo[i].m
            ktot += 0.5*pinfo[i].m*v2

    # bulge_to_total is defined as twice the fraction of counter-rotating mass
    L[5] = <float>(2.*m_counterrot / m_tot)
    # if L[5] > 1.: L[5] = 1.
    L[6] = <float>(krot / ktot)  # kappa_rot
    #if L[5] > 1 and ip == 0:
    #    printf("TROUBLE? npart=%d BoverT=%g  kappa_rot=%g L=%g\n",npt,L[5],L[6],Lmag)

    return

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef void nogil_virial_quants(part_struct *pinfo, double[:] Densities, int npart, int nDens, float[:] collectRadii, float[:] collectMasses) nogil:
    """Get virial mass and radius at desired set of Densities

    pinfo: struct holding mass, radii of particles
    Densities: Array of density values you want: e.g rho200, rhovirial, ... IN ASCENDING ORDER
    collectRadii, collectMasses: Empty array to hold radii, masses; same length as Densities
    """
    cdef int i,j
    cdef float volume, density
    cdef float PiFac = 4./3.*M_PI
    cdef double *mcum

    mcum = <double *> malloc(npart*sizeof(double))
    mcum[0] = pinfo[0].m
    for i in range(1,npart):
        mcum[i] = mcum[i-1] + pinfo[i].m
    for i in range(1,npart):
        if pinfo[i].r == 0.:
            continue
        volume = PiFac*pinfo[i].r*pinfo[i].r*pinfo[i].r
        density = mcum[i]/volume
        for j in range(nDens):
            if density > Densities[j]:
                collectRadii[j] = pinfo[i].r
                collectMasses[j] = mcum[i]
    #if npart > 5000: printf("%g %g %g %g %g %g %g\n",pinfo[npart-1].r,mcum[npart-1]/(PiFac*pinfo[npart-1].r*pinfo[npart-1].r*pinfo[npart-1].r),collectRadii[0],collectRadii[1],collectRadii[2],c_log10(collectMasses[0]),c_log10(collectMasses[1]),c_log10(collectMasses[2]))
    return

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef void nogil_rotator(float *vector, float ALPHA, float BETA) nogil:
    """Rotate a vector through rotator angles ALPHA, BETA, which are the yaw and pitch angles.
    See https://en.wikipedia.org/wiki/Rotation_matrix

    vector : 3-dim vector you want to rotate (usually pos's or vel's)
    ALPHA : Angle to rotate around first.
    BETA : Angle to rotate around second.
    """

    cdef int i,j
    cdef float c, s
    cdef float vcopy[3]

    # set up rotation matrix
    for i in range(3):
        vcopy[i] = vector[i]
    c = c_cos(ALPHA)
    s = c_sin(ALPHA)

    if ALPHA != 0:  # this rotates around x-axis ("yaw"), so x doesn't change
        vector[1] = c*vcopy[1] - s*vcopy[2]
        vector[2] = s*vcopy[1] + c*vcopy[2]

    c = c_cos(BETA)
    s = c_sin(BETA)

    if BETA != 0:  # now rotate around y-axis ("pitch")
        vector[0] = c*vcopy[0] - s*vcopy[2]
        vector[2] = s*vcopy[0] + c*vcopy[2]

    return

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef void nogil_radial_quants(int ig, long long istart, long long iend, int ndim, float[:] mass, float[:,:] pos, float[:,:] vel, int[:] ptype, float[:] cent_pos, float[:] cent_vel, float Lbox, int nptypes, int[:] group_ptypes, int gtflag, double[:] Densities, int nDens, float[:,:] grp_mass, float[:,:] grp_R20, float[:,:] grp_Rhalf, float[:,:] grp_R80, float[:,:] grp_vdisp, float[:,:,:] grp_L, float[:,:] grp_rvir, float[:,:] grp_mvir) nogil:
    """Compute radial quantities """

    cdef int ip, nparticles = iend-istart
    cdef float mtarget
    cdef part_struct *grp_partinfo

    # get radius of each particle, sort by radii
    grp_partinfo = <part_struct *> malloc((nparticles)*sizeof(part_struct))  # allocate particle info
    nogil_load_partinfo(mass, pos, vel, ptype, cent_pos, cent_vel, grp_partinfo, Lbox, istart, iend, ndim)
    qsort(<void*>grp_partinfo, <size_t>(nparticles), sizeof(part_struct), mycmp)

    # calculate radii, velocity dispersions, angular quantities
    for ip in range(nptypes+2):
        # compute total mass of a given ptype in order to set target
        if ip == nptypes+1:  # last value stores radii for all particles together
            mtarget = 0.
            for i in range(nptypes):
                mtarget += grp_mass[ig,i]
        elif ip == nptypes:  # second-to-last value stores baryonic radii
            mtarget = 0.
            for i in range(nptypes):
                if group_ptypes[i] == 1 or group_ptypes[i] == 2: continue  # skip DM particles
                mtarget += grp_mass[ig,i]
        else:
            mtarget = grp_mass[ig,ip]
        # compute radii for this ptype(s), and 20%, 50%, and 80% of total mass
        grp_R20[ig,ip] = nogil_half_mass_radius(grp_partinfo, 0.2*mtarget, ip, group_ptypes, nparticles)
        grp_Rhalf[ig,ip] = nogil_half_mass_radius(grp_partinfo, 0.5*mtarget, ip, group_ptypes, nparticles)
        grp_R80[ig,ip] = nogil_half_mass_radius(grp_partinfo, 0.8*mtarget, ip, group_ptypes, nparticles)
        # compute velocity dispersions for this ptype(s)
        grp_vdisp[ig,ip] = nogil_velocity_dispersions(grp_partinfo, ip, group_ptypes, nparticles, ndim)
        # calculate angular quantities for this ptypes(s)
        nogil_angular_quants(grp_partinfo, nparticles, ip, group_ptypes, grp_L[ig,ip])

        # calculate virial quantities
        if gtflag == 1:  # only calculate these for halos
            nogil_virial_quants(grp_partinfo, Densities, nparticles, nDens, grp_rvir[ig], grp_mvir[ig])

    #if ig < 5: printf("%d %q %q %g %g %g %g %g %g\n",ig,istart,iend,c_log10(grp_mass[ig,0]),c_log10(grp_mass[ig,1]),grp_Rhalf[ig,0],grp_Rhalf[ig,1],grp_vdisp[ig,1],grp_L[ig,1,5])

    free(grp_partinfo)


""" ============================================================ """
""" THESE ARE THE MAIN ROUTINE TO CALCULATE ALL GROUP PROPERTIES """
""" ============================================================ """

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)


def get_group_dust_properties(group,grp_list):
    # collect particle IDs
    from caesar.group import collate_group_ids
    from caesar.property_manager import ISM_NH_THRESHOLD

    ngroup, grpids, did_bins = collate_group_ids(grp_list,'dust',group.nparttype['dust'])

    cdef:
        ## gas quantities
        long long[:] hid_bins = did_bins   # starting indexes of particle IDs in each group
        float[:]   dm = group.obj.data_manager.mass[grpids]

        # general variables#
        int ng = ngroup
        int my_nproc = group.nproc
        int ig
        long long i,istart,iend
 #       double XH = group.obj.simulation.XH
 #       float ism_thresh = ISM_NH_THRESHOLD

        # Things to compute
        float[:]   grp_mass = np.zeros(ngroup,dtype=MY_DTYPE)  # total mass

    for ig in prange(ng,nogil=True,schedule='dynamic',num_threads=my_nproc):
        istart = hid_bins[ig]
        iend = hid_bins[ig+1]
        for i in range(istart,iend):
            grp_mass[ig] += dm[i]

    for ig in range(ng):
        grp_list[ig].masses['dust'] = group.obj.yt_dataset.quan(grp_mass[ig], group.obj.units['mass'])

    return


def get_group_gas_properties(group,grp_list):
    # collect particle IDs
    from caesar.group import collate_group_ids
    from caesar.property_manager import ISM_NH_THRESHOLD

    ngroup, grpids, gid_bins = collate_group_ids(grp_list,'gas',group.nparttype['gas'])

    cdef:
        ## gas quantities
        long long[:] hid_bins = gid_bins   # starting indexes of particle IDs in each group
        float[:]   gm = group.obj.data_manager.mass[grpids]
        float[:]   gnh = group.obj.data_manager.gnh[grpids]
        float[:]   gsfr = group.obj.data_manager.gsfr[grpids]
        float[:]   gZ = group.obj.data_manager.gZ[grpids]
        float[:]   gtemp = group.obj.data_manager.gT[grpids]
        float[:]   gfH2 = group.obj.data_manager.gfH2[grpids]
        float[:]   gfHI = group.obj.data_manager.gfHI[grpids]
        float[:]   mdust = group.obj.data_manager.dustmass[grpids]
        # general variables
        int ng = ngroup
        int my_nproc = group.nproc
        int ig
        long long i,istart,iend
        double XH = group.obj.simulation.XH
        float ism_thresh = ISM_NH_THRESHOLD
        # Things to compute
        float[:]   grp_mass = np.zeros(ngroup,dtype=MY_DTYPE)  # total mass
        float[:]   grp_mH2 = np.zeros(ngroup,dtype=MY_DTYPE)  # H2 mass
        float[:]   grp_mHI = np.zeros(ngroup,dtype=MY_DTYPE)  # H1 mass
        float[:]   grp_mdust = np.zeros(ngroup,dtype=MY_DTYPE)  # dust mass
        float[:]   grp_mism = np.zeros(ngroup,dtype=MY_DTYPE)  # mass of gas with nH<ISM_NH_THRESHOLD
        float[:]   grp_sfr = np.zeros(ngroup,dtype=MY_DTYPE)  # SFR
        float[:]   grp_Zm = np.zeros(ngroup,dtype=MY_DTYPE)  # mass-weighted metallicity
        float[:]   grp_Zsfr = np.zeros(ngroup,dtype=MY_DTYPE)  # SFR-weighted metallicity
        float[:]   grp_Tm = np.zeros(ngroup,dtype=MY_DTYPE)  # mass-weighted temperature
        float[:]   grp_Tcgm = np.zeros(ngroup,dtype=MY_DTYPE)  # mass-weighted CGM temperature (excluding SF gas)
        float[:]   grp_Zcgm = np.zeros(ngroup,dtype=MY_DTYPE)  # mass-weighted CGM metallicity (excluding SF gas)
        float[:]   grp_TZcgm = np.zeros(ngroup,dtype=MY_DTYPE)  # metal-weighted CGM temperature (excluding SF gas)
        float[:]   grp_ZTcgm = np.zeros(ngroup,dtype=MY_DTYPE)  # T-weighted CGM metallicity (excluding SF gas)

    for ig in prange(ng,nogil=True,schedule='dynamic',num_threads=my_nproc):
        istart = hid_bins[ig]
        iend = hid_bins[ig+1]
        for i in range(istart,iend):
            grp_mass[ig] += gm[i]
            grp_mH2[ig] += XH*gm[i]*gfH2[i]
            grp_mHI[ig] += XH*gm[i]*gfHI[i]
            if gnh[i] >= ism_thresh:
                grp_mism[ig] += gm[i]
                grp_mdust[ig] += mdust[i]
            grp_sfr[ig] += gsfr[i]
            grp_Zm[ig] += gZ[i]*gm[i]
            grp_Zsfr[ig] += gZ[i]*gsfr[i]
            grp_Tm[ig] /= gm[i]*gtemp[i]
            if gnh[i] < ism_thresh:
                grp_Zcgm[ig] += gm[i]*gZ[i]
                grp_Tcgm[ig] += gm[i]*gtemp[i]
                grp_ZTcgm[ig] += gm[i]*gtemp[i]*gZ[i]
                grp_TZcgm[ig] += gm[i]*gtemp[i]*gZ[i]
        grp_Zm[ig] /= grp_mass[ig]
        if grp_sfr[ig]>0:
            grp_Zsfr[ig] /= grp_sfr[ig]
        if grp_mass[ig] - grp_mism[ig] > 0:
            grp_ZTcgm[ig] /= grp_Tcgm[ig]
            grp_TZcgm[ig] /= grp_Zcgm[ig]
            grp_Zcgm[ig] /= grp_mass[ig] - grp_mism[ig];
            grp_Tcgm[ig] /= grp_mass[ig] - grp_mism[ig];

    for ig in range(ng):
        grp_list[ig].sfr = group.obj.yt_dataset.quan(grp_sfr[ig], '%s/%s' % (group.obj.units['mass'],group.obj.units['time']))
        grp_list[ig].masses['H2'] = group.obj.yt_dataset.quan(grp_mH2[ig], group.obj.units['mass'])
        grp_list[ig].masses['HI'] = group.obj.yt_dataset.quan(grp_mHI[ig], group.obj.units['mass'])
        grp_list[ig].metallicities = dict(
            mass_weighted = group.obj.yt_dataset.quan(grp_Zm[ig], ''),
            sfr_weighted  = group.obj.yt_dataset.quan(grp_Zsfr[ig], ''),
            mass_weighted_cgm = group.obj.yt_dataset.quan(grp_Zcgm[ig], ''),
            temp_weighted_cgm = group.obj.yt_dataset.quan(grp_ZTcgm[ig], '')
        )
        grp_list[ig].temperatures = dict(
            mass_weighted = group.obj.yt_dataset.quan(grp_Tm[ig], group.obj.units['temperature']),
            mass_weighted_cgm = group.obj.yt_dataset.quan(grp_Tcgm[ig], group.obj.units['temperature']),
            metal_weighted_cgm = group.obj.yt_dataset.quan(grp_TZcgm[ig], group.obj.units['temperature'])
        )
        grp_list[ig].masses['dust'] = group.obj.yt_dataset.quan(grp_mdust[ig], group.obj.units['mass'])

    return


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def get_group_star_properties(group,grp_list):
    # collect particle IDs
    from caesar.group import collate_group_ids
    ngroup, grpids, gid_bins = collate_group_ids(grp_list,'star',group.nparttype['star'])

    cdef:
        ## star quantities
        long long[:] hid_bins = gid_bins   # starting indexes of particle IDs in each group
        float[:]   sm = group.obj.data_manager.mass[grpids]
        float[:]   sZ = group.obj.data_manager.sZ[grpids]
        float[:]   sage = group.obj.data_manager.age[grpids]
        # general variables
        int ng = ngroup
        int my_nproc = group.nproc
        int ig
        long long i,istart,iend
        # Things to compute
        float[:]   grp_mass = np.zeros(ngroup,dtype=MY_DTYPE)  # total mass
        float[:]   grp_Zm = np.zeros(ngroup,dtype=MY_DTYPE)  # mass-weighted metallicity
        float[:]   grp_age = np.zeros(ngroup,dtype=MY_DTYPE)  # mass-weighted mean age in Gyr
        float[:]   grp_ageZ = np.zeros(ngroup,dtype=MY_DTYPE)  # metal-weighted mean age in Gyr
        float[:]   grp_sfr100 = np.zeros(ngroup,dtype=MY_DTYPE)  # SFR over last 100 Myr in Mo/yr

    for ig in prange(ng,nogil=True,schedule='dynamic',num_threads=my_nproc):
        istart = hid_bins[ig]
        iend = hid_bins[ig+1]
        for i in range(istart,iend):
            grp_mass[ig] += sm[i]
            grp_Zm[ig] += sZ[i]*sm[i]
            grp_age[ig] += sage[i]*sm[i]
            grp_ageZ[ig] += sage[i]*sm[i]*sZ[i]
            if sage[i] < 0.1:  # last 100 Myr
                grp_sfr100[ig] += sm[i]
        if grp_mass[ig] > 0:
            grp_ageZ[ig] /= grp_Zm[ig]
            grp_Zm[ig] /= grp_mass[ig]
            grp_age[ig] /= grp_mass[ig]

    for ig in range(ng):
        grp_list[ig].metallicities['stellar'] = group.obj.yt_dataset.quan(grp_Zm[ig], '')
        grp_list[ig].sfr_100 = group.obj.yt_dataset.quan(grp_sfr100[ig]/100.e6, 'Msun/yr')
        grp_list[ig].ages = dict(
            mass_weighted = group.obj.yt_dataset.quan(grp_age[ig], 'Gyr'),
            metal_weighted = group.obj.yt_dataset.quan(grp_ageZ[ig], 'Gyr')
        )

    return

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def get_group_bh_properties(group,grp_list):

    # collect particle IDs
    from caesar.group import collate_group_ids
    ngroup, grpids, gid_bins = collate_group_ids(grp_list,'bh',group.nparttype['bh'])

    cdef:
        ## bh quantities
        long long[:] hid_bins = gid_bins   # starting indexes of particle IDs in each group
        float[:]   bhmass = group.obj.data_manager.bhmass[grpids]
        float[:]   bhmdot = group.obj.data_manager.bhmdot[grpids]
        # general variables
        int ng = ngroup
        int my_nproc = group.nproc
        int ig
        long long i, imax, istart,iend
        float bhmax
        # Things to compute
        float[:]   bhm = np.zeros(ngroup,dtype=MY_DTYPE)  # max BH mass in group
        float[:]   bhrate = np.zeros(ngroup,dtype=MY_DTYPE)  # accretion rate of highest mass BH


    for ig in prange(ng,nogil=True,schedule='dynamic',num_threads=my_nproc):
        istart = hid_bins[ig]
        iend = hid_bins[ig+1]
        imax = -1
        bhmax = 0.0
        for i in range(istart,iend):
            if bhmass[i] > bhmax:
                bhmax = bhmass[i]
                imax = i
        if imax >= 0:
            bhm[ig] = bhmass[imax]
            bhrate[ig] = bhmdot[imax]

    from astropy import constants as const
    FRAD = 0.1  # assume 10% radiative efficiency
    edd_factor = (4 * np.pi * const.G * const.m_p / (FRAD * const.c * const.sigma_T)).to('1/yr').value
    for ig in range(ng):
        grp_list[ig].masses['bh'] = group.obj.yt_dataset.quan(bhm[ig], group.obj.units['mass'])
        grp_list[ig].bhmdot = group.obj.yt_dataset.quan(bhrate[ig], 'Msun/yr')
        if bhm[ig] > 0:
            grp_list[ig].bh_fedd = group.obj.yt_dataset.quan(bhrate[ig] / (edd_factor * bhm[ig]), '')
        else:
            grp_list[ig].bh_fedd = group.obj.yt_dataset.quan(0.0, '')

    return

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def get_group_overall_properties(group,grp_list):
    """Calculate physical properties of a set of objects in a fof6d group.
    Computes properties, assigns to Caesar object, and fills the associated caesar
        object list (e.g. halo_list/galaxy_list/cloud_list).  No return value.

    Parameters
    ----------
    group : fof6d instance (see fof6d.py) holding the set of objects to process
    grp_list: list of groups (e.g. Halo/Galaxy/Cloud) to process

    """

    from caesar.group import MINIMUM_DM_PER_HALO,MINIMUM_STARS_PER_GALAXY,MINIMUM_GAS_PER_CLOUD
    from caesar.property_manager import ptype_ints
    from caesar.group import group_types, collate_group_ids, list_types

    # collect particle IDs.  need to concatenate into a single array for cython.
    ngroup, grpids, gid_bins = collate_group_ids(grp_list,'all',group.nparttot)
    memlog('Calculating properties for %d %s (nproc=%d)'%(ngroup,group_types[group.obj_type],group.nproc))

    # collect all the particle type integers for particles in this group
    # NOTE: this routine assumes gadget numbering: 0=gas, 1=DM, 2=DM2, 3=dust, 4=star, 5=BH
    pt_ints = []
    for p in group.obj.data_manager.ptypes:
        if (group.obj_type in ['galaxy', 'cloud']):
            if p not in ['dm','dm2','dm3']:  # not DM informaiton for galaxies, see aperture calculation later
                pt_ints.append(ptype_ints[p])
        else:
            pt_ints.append(ptype_ints[p])

    cdef:
        ## global quantities
        long long[:] hid_bins = gid_bins   # starting indexes of particle IDs in each group
        float[:,:] pos = group.obj.data_manager.pos[grpids]
        float[:,:] vel = group.obj.data_manager.vel[grpids]
        float[:]   mass = group.obj.data_manager.mass[grpids]
        float[:]   pot = group.obj.data_manager.pot[grpids]
        int[:]      ptype = group.obj.data_manager.ptype[grpids]
        int[:]      group_ptypes = np.asarray(pt_ints,dtype=np.int32)
        int         nptypes = len(group_ptypes)
        # general variables
        int ng = ngroup
        int ndim = len(pos.T)
        int imin = -1
        int my_nproc = group.nproc
        double[:] Densities = group.obj.simulation.Densities.in_units(group.obj.units['mass']+'/'+group.obj.units['length']+'**3')
        int nDens = len(Densities)
        float Lbox = group.obj.simulation.boxsize.d
        bint use_pot = group.obj.load_pot
        int minpotpart,gtflag=-1
        int j,ig,ip,ir,binary
        long long i,istart,iend
        float[3] dx
        float mtarget
        float r200_fact, G_in_simunits

        # things to compute
        float[:]   grp_mtot = np.zeros(ngroup,dtype=MY_DTYPE)  # total masses
        float[:,:] grp_mass = np.zeros((ngroup,nptypes),dtype=MY_DTYPE)  # masses in the various types
        # int[:,:]   grp_count = np.zeros((ngroup,nptypes),dtype=np.int32)  # part counts in the various types
        float[:,:] grp_pos = np.zeros((ngroup,ndim),dtype=MY_DTYPE)  # CoM positions
        float[:,:] grp_vel = np.zeros((ngroup,ndim),dtype=MY_DTYPE)  # CoM velocities
        float[:,:] grp_minpotpos = np.zeros((ngroup,ndim),dtype=MY_DTYPE)  # position of minimum potential
        float[:,:] grp_minpotvel = np.zeros((ngroup,ndim),dtype=MY_DTYPE)  # velocity of minimum potential
        float[:,:] grp_R20 = np.zeros((ngroup,nptypes+2),dtype=MY_DTYPE)  # 80% mass-enclosing radius
        float[:,:] grp_Rhalf = np.zeros((ngroup,nptypes+2),dtype=MY_DTYPE)  # half-mass radius
        float[:,:] grp_R80 = np.zeros((ngroup,nptypes+2),dtype=MY_DTYPE)  # 80% mass-enclosing radius
        float[:,:] grp_vdisp = np.zeros((ngroup,nptypes+2),dtype=MY_DTYPE)  # velocity dispersions
        float[:,:,:] grp_L = np.zeros((ngroup,nptypes+2,7),dtype=MY_DTYPE)  # holds angular quants (Lx,Ly,Lz,ALPHA,BETA,B/T,kappa_rot)
        float[:,:] grp_mvir = np.zeros((ngroup,nDens),dtype=MY_DTYPE)  # virial masses like M500, M2500, ...
        float[:,:] grp_rvir = np.zeros((ngroup,nDens),dtype=MY_DTYPE)  # corresponding radii

    # preliminary stuff to set up calculation of properties
    if group.obj_type == 'halo': gtflag = 1
    elif group.obj_type == 'galaxy': gtflag = 2
    elif group.obj_type == 'cloud': gtflag = 3
    else: sys.exit('Group type %s not recognized'%group.obj_type)

    ## loop over objects, calculate properties for each object
    for ig in prange(ng,nogil=True,schedule='dynamic',num_threads=my_nproc):
    #for ig in range(ng):
        istart = hid_bins[ig]
        iend = hid_bins[ig+1]

        # compute masses and particle counts
        for ip in range(nptypes):
            for i in range(istart,iend):
                if ptype[i] == group_ptypes[ip]:
                    grp_mass[ig,ip] += mass[i]
                    # grp_count[ig,ip] += 1
            grp_mtot[ig] += grp_mass[ig,ip]

        # Center of mass quantities
        nogil_CoM_quants(ig, pos, vel, mass, pot, ptype, group_ptypes, grp_mtot, istart, iend, ndim, Lbox, grp_pos, grp_vel, grp_minpotpos, grp_minpotvel)

        # Compute other quantities that require radially sorted particle list
        if gtflag == 1 and use_pot: # if halo, use min potential for halo center
            nogil_radial_quants(ig, istart, iend, ndim, mass, pos, vel, ptype, grp_minpotpos[ig], grp_minpotvel[ig], Lbox, nptypes, group_ptypes, gtflag, Densities, nDens, grp_mass, grp_R20, grp_Rhalf, grp_R80, grp_vdisp, grp_L, grp_rvir, grp_mvir)
        else:
            nogil_radial_quants(ig, istart, iend, ndim, mass, pos, vel, ptype, grp_pos[ig], grp_vel[ig], Lbox, nptypes, group_ptypes, gtflag, Densities, nDens, grp_mass, grp_R20, grp_Rhalf, grp_R80, grp_vdisp, grp_L, grp_rvir, grp_mvir)

    # assign quantities to groups, with units
    from caesar.property_manager import has_ptype
    L_units = 'Msun * kpccm * km/s'
    r200_fact = (200*group.obj.simulation.Om_z*1.3333333*np.pi*group.obj.simulation.critical_density.in_units('Msun/kpccm**3'))**(-1./3.)
    G_in_simunits = group.obj.simulation.G.to('(km**2 * kpc)/(Msun * s**2)')  # so we get vcirc in km/s
    ds = group.obj.yt_dataset
    if not group.obj.load_pot:
        mylog.warning('Potential not found in snapshot: minpotpos/vel not computed, halo radial quantities taken around CoM')
    for ig in range(ng):
        mygroup = grp_list[ig]
        mygroup.masses['total'] = group.obj.yt_dataset.quan(grp_mtot[ig], group.obj.units['mass'])
        mbaryon = 0.
        for ip,p in enumerate(group.obj.data_manager.ptypes):
            if has_ptype(group.obj,p):
                mygroup.masses[list_types[p]] = group.obj.yt_dataset.quan(grp_mass[ig,ip], group.obj.units['mass'])
                if p is not 'dm' and p is not 'dm2' and p is not 'dm3':
                    mbaryon += grp_mass[ig,ip]
        mygroup.masses['baryon'] = group.obj.yt_dataset.quan(mbaryon, group.obj.units['mass'])
        mygroup.pos = group.obj.yt_dataset.arr(grp_pos[ig], group.obj.units['length'])
        mygroup.vel = group.obj.yt_dataset.arr(grp_vel[ig], group.obj.units['velocity'])
        if group.obj.load_pot:
            mygroup.minpotpos = group.obj.yt_dataset.arr(grp_minpotpos[ig], group.obj.units['length'])
            mygroup.minpotvel = group.obj.yt_dataset.arr(grp_minpotvel[ig], group.obj.units['velocity'])
        for ip in range(nptypes+2):
            if ip == nptypes+1:
                mygroup.radii['total_r20'] = group.obj.yt_dataset.quan(grp_R20[ig,ip], group.obj.units['length'])
                mygroup.radii['total_half_mass'] = group.obj.yt_dataset.quan(grp_Rhalf[ig,ip], group.obj.units['length'])
                mygroup.radii['total_r80'] = group.obj.yt_dataset.quan(grp_R80[ig,ip], group.obj.units['length'])
                mygroup.velocity_dispersions['total'] = group.obj.yt_dataset.quan(grp_vdisp[ig,ip], group.obj.units['velocity'])
                mygroup.rotation['total_L'] = group.obj.yt_dataset.arr( [grp_L[ig,ip,0],grp_L[ig,ip,1],grp_L[ig,ip,2]], L_units)
                mygroup.rotation['total_ALPHA'] = group.obj.yt_dataset.quan(grp_L[ig,ip,3],'')
                mygroup.rotation['total_BETA'] = group.obj.yt_dataset.quan(grp_L[ig,ip,4],'')
                mygroup.rotation['total_BoverT'] = group.obj.yt_dataset.quan(grp_L[ig,ip,5],'')
                mygroup.rotation['total_kappa_rot'] = group.obj.yt_dataset.quan(grp_L[ig,ip,6],'')
            elif ip == nptypes:
                mygroup.radii['baryon_r20'] = group.obj.yt_dataset.quan(grp_R20[ig,ip], group.obj.units['length'])
                mygroup.radii['baryon_half_mass'] = group.obj.yt_dataset.quan(grp_Rhalf[ig,ip], group.obj.units['length'])
                mygroup.radii['baryon_r80'] = group.obj.yt_dataset.quan(grp_R80[ig,ip], group.obj.units['length'])
                mygroup.velocity_dispersions['baryon'] = group.obj.yt_dataset.quan(grp_vdisp[ig,ip], group.obj.units['velocity'])
                mygroup.rotation['baryon_L'] = group.obj.yt_dataset.arr( [grp_L[ig,ip,0],grp_L[ig,ip,1],grp_L[ig,ip,2]], L_units)
                mygroup.rotation['baryon_ALPHA'] = group.obj.yt_dataset.quan(grp_L[ig,ip,3],'')
                mygroup.rotation['baryon_BETA'] = group.obj.yt_dataset.quan(grp_L[ig,ip,4],'')
                mygroup.rotation['baryon_BoverT'] = group.obj.yt_dataset.quan(grp_L[ig,ip,5],'')
                mygroup.rotation['baryon_kappa_rot'] = group.obj.yt_dataset.quan(grp_L[ig,ip,6],'')
            else:
                if has_ptype(group.obj,group.obj.data_manager.ptypes[ip]):
                    name = list_types[group.obj.data_manager.ptypes[ip]]+'_r20'
                    mygroup.radii[name] = group.obj.yt_dataset.quan(grp_R20[ig,ip], group.obj.units['length'])
                    name = list_types[group.obj.data_manager.ptypes[ip]]+'_half_mass'
                    mygroup.radii[name] = group.obj.yt_dataset.quan(grp_Rhalf[ig,ip], group.obj.units['length'])
                    name = list_types[group.obj.data_manager.ptypes[ip]]+'_r80'
                    mygroup.radii[name] = group.obj.yt_dataset.quan(grp_R80[ig,ip], group.obj.units['length'])
                    mygroup.velocity_dispersions[list_types[group.obj.data_manager.ptypes[ip]]] = group.obj.yt_dataset.quan(grp_vdisp[ig,ip], group.obj.units['velocity'])
                    name = list_types[group.obj.data_manager.ptypes[ip]]
                    mygroup.rotation[name+'_L'] = group.obj.yt_dataset.arr( [grp_L[ig,0,ip],grp_L[ig,1,ip],grp_L[ig,2,ip]], L_units)
                    mygroup.rotation[name+'_ALPHA'] = group.obj.yt_dataset.quan(grp_L[ig,ip,3],'')
                    mygroup.rotation[name+'_BETA'] = group.obj.yt_dataset.quan(grp_L[ig,ip,4],'')
                    mygroup.rotation[name+'_BoverT'] = group.obj.yt_dataset.quan(grp_L[ig,ip,5],'')
                    mygroup.rotation[name+'_kappa_rot'] = group.obj.yt_dataset.quan(grp_L[ig,ip,6],'')

        # some additional halo quantities to store
        if mygroup.obj_type is 'halo':
            mygroup.virial_quantities['r200'] = group.obj.yt_dataset.quan(r200_fact * grp_mtot[ig]**(1./3.), group.obj.units['length'])  # effective R200 calculated for total (FOF) mass.
            mygroup.virial_quantities['circular_velocity'] = group.obj.yt_dataset.quan(np.sqrt(G_in_simunits * grp_mtot[ig] / mygroup.virial_quantities['r200']), group.obj.units['velocity'])  # sqrt(GM_FOF/R_200)
            mygroup.virial_quantities['temperature'] = group.obj.yt_dataset.quan(3.6e5 * (mygroup.virial_quantities['circular_velocity'] / 100.0)**2, 'K')  # eq 4 of Mo et al 2002 (K)
            #angular_momentum = group.obj.yt_dataset.quan(np.linalg.norm(grp_L[ig,0,:3]), L_units)
            mygroup.virial_quantities['spin_param'] = np.linalg.norm(mygroup.rotation['total_L']) / (1.4142135623730951 * mygroup.masses['total'] * mygroup.virial_quantities['circular_velocity'] * mygroup.virial_quantities['r200'])
            for ir,rtype in enumerate(['200c','500c','2500c']):  # these should match the ones in simulation.Densities
                mygroup.virial_quantities['r'+rtype] = group.obj.yt_dataset.quan(grp_rvir[ig,ir], group.obj.units['length'])
                mygroup.virial_quantities['m'+rtype] = group.obj.yt_dataset.quan(grp_mvir[ig,ir], group.obj.units['mass'])

        if ig < 0:
            print('NEW: %g %g %g %g %g %g %g %g %g %g'%(mygroup.radii['total_r20'],mygroup.radii['total_half_mass'],mygroup.radii['total_r80'],mygroup.radii['baryon_r20'],mygroup.radii['baryon_half_mass'],mygroup.radii['baryon_r80'],mygroup.rotation['stellar_BoverT'],mygroup.rotation['gas_BoverT'],mygroup.rotation['stellar_kappa_rot'],mygroup.rotation['gas_kappa_rot']))
            print('OLD: %g %g %g %g'%(grp_list[ig].radii['total_half_mass'],grp_list[ig].radii['baryon_half_mass'],grp_list[ig].radii['gas_half_mass'],grp_list[ig].radii['stellar_half_mass']))

        grp_list[ig] = mygroup
        #if ig < 5: print('%d: tot %g dm %g gas %g star %g bh %g [%g %g %g] sig %g %g %g'%(ig,np.log10(mygroup.masses['total']),np.log10(mygroup.masses['dm']),np.log10(mygroup.masses['gas']),np.log10(mygroup.masses['stellar']),np.log10(mygroup.masses['bh']),mygroup.pos[0],mygroup.pos[1],mygroup.pos[2],mygroup.velocity_dispersions['gas'],mygroup.velocity_dispersions['stellar'],mygroup.velocity_dispersions['dm']))

    memlog('Computed properties for %d %s'%(group.counts[group.obj_type],group_types[group.obj_type]))


""" ========================================================== """
""" OLD CYTHON ROUTINES FOR GROUP PROPERTY CALCULATIONS        """
""" ========================================================== """

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def append_bh(
        double boxsize,
        double gal_R,
        np.ndarray[np.float64_t, ndim=1] gal_mass,
        np.ndarray[np.float64_t, ndim=2] gal_pos,
        np.ndarray[np.float64_t, ndim=2] p_pos,
        np.ndarray[np.int32_t, ndim=1] p_group_id
):
    """Append the particles to the galaxies
    Parameters
    ----------
    boxsize : double
    gal_mass: np.ndarray([1,2,3,...], dtype=np.float64)
        containing the masses of galaxies
    gal_r   : np.ndarray([1,2,3,...], dtype=np.float64)
        containing the size (radius) of the galaxies
    gal_pos : np.ndarray((N,3), dtype=np.float64)
        Position of the galaxies
    p_pos  : np.ndarray((Np,3), dtype=np.float64)
        Position of the particles
    p_group_id: np.ndarray(Np, dtype=np.int32)
        initialised with -1
    """

    cdef int i,j
    cdef int ngal = np.shape(gal_pos)[0]
    cdef int npart= len(p_group_id)
    cdef double dx = 0.0
    cdef double dy = 0.0
    cdef double dz = 0.0
    cdef double halfbox = boxsize / 2.0
    cdef double r2, rold
    cdef double gal_R2 = gal_R*gal_R

    for i in range(ngal):
        for j in range(npart):
            dx    = periodic(gal_pos[i,0] - p_pos[j,0], halfbox, boxsize)
            dy    = periodic(gal_pos[i,1] - p_pos[j,1], halfbox, boxsize)
            dz    = periodic(gal_pos[i,2] - p_pos[j,2], halfbox, boxsize)
            r2    = dx*dx + dy*dy + dz*dz
            #if (r2 > gal_R[i]*gal_R[i]): continue
            if (r2 > gal_R2): continue

            if p_group_id[j]==-1:
                p_group_id[j] = i
            else:
                dx    = periodic(gal_pos[p_group_id[j],0] - p_pos[j,0], halfbox, boxsize)
                dy    = periodic(gal_pos[p_group_id[j],1] - p_pos[j,1], halfbox, boxsize)
                dz    = periodic(gal_pos[p_group_id[j],2] - p_pos[j,2], halfbox, boxsize)
                r2old = dx*dx + dy*dy + dz*dz
                if (gal_mass[i]/r2) > (gal_mass[p_group_id[j]]/r2old):
                    p_group_id[j] = i


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def get_periodic_r(
        double boxsize,
        np.ndarray[np.float64_t, ndim=1] center,
        np.ndarray[np.float64_t, ndim=2] pos,
        np.ndarray[np.float64_t, ndim=1] r
):
    """Get periodic radii.

    Parameters
    ----------
    boxsize : double
        The size of your domain.
    center : np.ndarray([x,y,z])
        Position in which to calculate the radius from.
    pos : np.ndarray
        Nx3 numpy array containing the positions of particles.
    r : np.array
        Empty array to fill with radius values.

    """
    cdef int i
    cdef int n = len(r)
    cdef double dx = 0.0
    cdef double dy = 0.0
    cdef double dz = 0.0
    cdef double halfbox = boxsize / 2.0

    for i in range(0,n):
        dx = periodic(center[0] - pos[i,0], halfbox, boxsize)
        dy = periodic(center[1] - pos[i,1], halfbox, boxsize)
        dz = periodic(center[2] - pos[i,2], halfbox, boxsize)

        r[i] = sqrt(dx*dx + dy*dy + dz*dz)

cdef double periodic(double x, double halfbox, double boxsize):
    if x < -halfbox:
        x += boxsize
    if x > halfbox:
        x -= boxsize
    return x

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def get_virial_mr(
        np.ndarray[np.float64_t, ndim=1] Densities,
        np.ndarray[np.float64_t, ndim=1] mass,
        np.ndarray[np.float64_t, ndim=1] r,
        np.ndarray[np.float64_t, ndim=1] collectRadii,
        np.ndarray[np.float64_t, ndim=1] collectMasses
):
    """Get virial mass and radius.

    Parameters
    ----------
    Density: array
        Different densities you are interested in: e.g rho200, rhovirial, ...
        They have to be in ascending order.
    r : array
        Particle radii inward
    mass: array
        Cumulative Particle masses inward
    collectRadii: array
        Empty array to contain the radii
        Should be the same size as the Densities
    """
    cdef int i
    cdef int j = len(Densities)
    cdef int k = 0
    cdef int n = len(r)
    cdef double volume, density
    cdef double PiFac = 4./3.*M_PI

    for i in range(0,n):
        volume = PiFac*r[i]*r[i]*r[i]
        density = mass[i]/volume
        while density > Densities[k]:
            collectRadii[k] = r[i]
            collectMasses[k] = mass[i]
            k += 1
            if k == j: return
    return


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def rotator(
        np.ndarray[np.float64_t, ndim=2] vals,
        np.ndarray[np.float64_t, ndim=2] Rx,
        np.ndarray[np.float64_t, ndim=2] Ry,
        double ALPHA, double BETA
):
    """Rotate a number of vectors around ALPHA, BETA

    Parameters
    ----------
    vals : np.ndarray
        Nx3 np.ndarray of values you want to rotate.
    Rx : np.ndarray
        3x3 array used for the first rotation about ALPHA.
        The dot product is taken against each value:
        vals[i] = np.dot(Rx, vals[i])
    Ry : np.ndarray
        3x3 array used for the second rotation about BETA
        The dot product is taken against each value:
        vals[i] = np.dot(Ry, vals[i])
    ALPHA : double
        Angle to rotate around first.
    BETA : double
        Angle to rotate around second.

    Notes
    -----
    This is typically called from :func:`utils.rotator`.

    """

    cdef int i
    cdef int n = len(vals)

    cdef double alpha_c = Rx[1,1]
    cdef double alpha_s = Rx[2,1]

    cdef double beta_c = Ry[0,0]
    cdef double beta_s = Ry[2,0]

    cdef double x, y, z

    for i in range(0,n):
        if ALPHA != 0:
            y = vals[i,1]
            z = vals[i,2]

            vals[i,1] = alpha_c * y - alpha_s * z
            vals[i,2] = alpha_s * y + alpha_c * z

        if BETA != 0:
            x = vals[i,0]
            z = vals[i,2]

            vals[i,0] = beta_c * x - beta_s * z
            vals[i,2] = beta_s * x + beta_c * z

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def get_half_mass_radius(
        np.ndarray[np.float64_t, ndim=1] mass,
        np.ndarray[np.float64_t, ndim=1] radii,
        np.ndarray[np.int32_t, ndim=1] ptype,
        double half_mass,
        int binary
):
    """Get half mass radius for a set of particles.

    Parameters
    ----------
    mass : np.ndarray
        Masses of particles.
    radii : np.ndarray
        Radii of particles.
    ptype : np.ndarray
        Array of integers containing the particle types.
    half_mass : double
        Half mass value to accumulate to.
    binary : int
        Integer used to select particle types.  For example,
        if you are interested in particle types 0 and 3 this
        value would be 2^0+2^3=9.

    """
    cdef int i
    cdef int n = len(mass)
    cdef double cumulative_mass = 0.0
    cdef double r = 0.0

    for i in range(0,n):
        if ((1<<ptype[i]) & (binary)) > 0:
            cumulative_mass += mass[i]
            if cumulative_mass >= half_mass:
                r = radii[i]
                break

    return r


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def get_full_mass_radius(
        np.ndarray[np.float64_t, ndim=1] radii,
        np.ndarray[np.int32_t, ndim=1] ptype,
        int binary
):
    """Get full mass radius for a set of particles.

    Parameters
    ----------
    radii : np.ndarray[::-1]
        Radii of particles
    ptype : np.ndarray[::-1]
        Array of integers containing the particle types.
    binary : int
        Integer used to select particle types.  For example,
        if you are interested in particle types 0 and 3 this
        value would be 2^0+2^3=9.

    Notes
    -----
    This function iterates forward through the array, so it
    is advisable to reverse the radii & ptype arrays before
    passing them via np.ndarray[::-1].

    """
    cdef int i
    cdef int n = len(radii)
    cdef double r = 0.0

    for i in range(0,n):
        if ((1<<ptype[i]) & (binary)) > 0:
            r = radii[i]
            break

    return r
