import numpy as np
    
def rotator(vals, ALPHA=0, BETA=0):
    """Rotate particle set around given angles.

    Parameters
    ----------
    vals : np.array
        a Nx3 array typically consisting of
        either positions or velocities.
    ALPHA : float, optional
        First angle to rotate about
    BETA : float, optional
        Second angle to rotate about

    Examples
    --------
    >>> rotated_pos = rotator(positions, 32.3, 55.2)

    """    
    c  = np.cos(ALPHA)
    s  = np.sin(ALPHA)
    Rx = np.array([
        [1.0,0.0,0.0],
        [0.0,  c, -s],
        [0.0,  s,  c]
    ])

    c  = np.cos(BETA)
    s  = np.sin(BETA)
    Ry = np.array([
        [  c,0.0, -s],
        [0.0,1.0,0.0],
        [  s,0.0,  c]
    ])
    
    # one value to rotate
    if len(np.shape(vals)) == 1:    
        if ALPHA != 0:
            vals = np.dot(Rx, vals)
        if BETA != 0:
            vals = np.dot(Ry, vals)

    # rotating many values
    else:
        from .group_funcs import rotator as rotator_cython
        rotator_cython(vals, Rx, Ry, ALPHA, BETA)
        
    return vals


def calculate_local_densities(obj, group_list):
    """Calculate the local number and mass density of objects.

    Parameters
    ----------
    obj : SPHGR object
    group_list : list
        List of objects to perform this operation on.

    """    
    from caesar.periodic_kdtree import PeriodicCKDTree

    pos  = np.array([i.pos for i in group_list])
    mass = np.array([i.masses['total'] for i in group_list])
    box  = obj.simulation.boxsize
    box  = np.array([box,box,box])
    
    TREE = PeriodicCKDTree(box, pos)

    search_radius = obj.simulation.search_radius
    search_volume = 4.0/3.0 * np.pi * search_radius**3

    for group in group_list:
        inrange = TREE.query_ball_point(group.pos, search_radius.d)
        total_mass = obj.yt_dataset.quan(np.sum(mass[inrange]), obj.units['mass'])
        group.local_mass_density   = total_mass / search_volume
        group.local_number_density = float(len(inrange)) / search_volume
