## SHELL FILES FOR DOCUMENTATION PURPOSES           ##
## if you want to see the source please examine     ##
## caesar/group_funcs/group_funcs.pyx               ##


def get_periodic_r(
        boxsize,
        center,
        pos,
        r
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
    pass


def rotator(
        vals,
        Rx,
        Ry,        
        ALPHA,
        BETA
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
    pass


def get_half_mass_radius(
        mass,
        radii,
        ptype,
        half_mass,
        binary
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
    pass

def get_full_mass_radius(
        radii,
        ptype,
        binary
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
    pass
