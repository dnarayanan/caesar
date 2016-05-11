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


