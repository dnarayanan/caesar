import numpy as np
cimport numpy as np
cimport cython

from libc.stdio cimport printf
cdef extern from "math.h":
    double sqrt(double x)
    double M_PI

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
