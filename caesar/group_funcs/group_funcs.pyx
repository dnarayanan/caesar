import numpy as np
cimport numpy as np
cimport cython

cdef extern from "math.h":
    double sqrt(double x)    

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def get_periodic_r(
        double boxsize,
        np.ndarray[np.float64_t, ndim=1] center,
        np.ndarray[np.float64_t, ndim=2] pos,
        np.ndarray[np.float64_t, ndim=1] r
):

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
def rotator(
        np.ndarray[np.float64_t, ndim=2] vals,
        np.ndarray[np.float64_t, ndim=2] Rx,
        np.ndarray[np.float64_t, ndim=2] Ry,        
        double ALPHA, double BETA
):

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
