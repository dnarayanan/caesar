## SHELL FILES FOR DOCUMENTATION PURPOSES ##


def get_periodic_r(
        double boxsize,
        np.ndarray[np.float64_t, ndim=1] center,
        np.ndarray[np.float64_t, ndim=2] pos,
        np.ndarray[np.float64_t, ndim=1] r
):
    """Get periodic r values."""
    pass


def rotator(
        np.ndarray[np.float64_t, ndim=2] vals,
        np.ndarray[np.float64_t, ndim=2] Rx,
        np.ndarray[np.float64_t, ndim=2] Ry,        
        double ALPHA, double BETA
):
    """Do some rotating."""
    pass

def get_half_mass_radius(
        np.ndarray[np.float64_t, ndim=1] mass,
        np.ndarray[np.float64_t, ndim=1] radii,
        np.ndarray[np.int32_t, ndim=1] ptype,
        double half_mass,
        int binary
):
    """Get half mass radius."""
    pass

def get_full_mass_radius(
        np.ndarray[np.float64_t, ndim=1] radii,
        np.ndarray[np.int32_t, ndim=1] ptype,
        int binary
):
    """Get full mass radius."""
    pass
