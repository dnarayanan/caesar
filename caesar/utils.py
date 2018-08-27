import numpy as np
from yt.funcs import mylog

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
    if len(group_list) == 0:
        return
    
    try:
        from scipy.spatial import KDTree, cKDTree
        from caesar.periodic_kdtree import PeriodicCKDTree
        #mylog.info('Calculating local densities')
    except:
        mylog.warning('Could not import scipy.spatial! '   \
                      'Please install scipy to allow for ' \
                      'local density calculations.')
        return

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


def info_printer(obj, group_type, top):
    """General method to print data.
    
    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Main CAESAR object.
    group_type : {'halo','galaxy','cloud'}
        Type of group to print data for.
    top : int
        Number of objects to print.

    """
    from caesar.group import group_types
    if group_type == 'halo':
        group_list = obj.halos
    elif group_type == 'galaxy':
        group_list = obj.galaxies
    elif group_type == 'cloud':
        group_list = obj.clouds
        
    nobjs = len(group_list)
    if top > nobjs:
        top = nobjs

    if obj.simulation.cosmological_simulation:
        time = 'z=%0.3f' % obj.simulation.redshift
    else:
        time = 't=%0.3f' % obj.simulation.time

    output  = '\n'
    output += '## Largest %d %s\n' % (top, group_types[group_type])
    if hasattr(obj, 'data_file'): output += '## from: %s\n' % obj.data_file
    output += '## %d @ %s' % (nobjs, time)
    output += '\n\n'

    cnt = 1
    if group_type == 'halo':
        output += ' ID    Mdm       Mstar     Mgas      r         fgas   nrho\t|  CentralGalMstar\n'
        #         ' 0000  4.80e+09  4.80e+09  4.80e+09  7.64e-09  0.000  7.64e-09\t|  7.64e-09'
        output += ' ---------------------------------------------------------------------------------\n'
        for o in group_list:
            cgsm = -1
            if hasattr(o,'central_galaxy'): cgsm = o.central_galaxy.masses['stellar']
            output += ' %04d  %0.2e  %0.2e  %0.2e  %0.2e  %0.3f  %0.2e\t|  %0.2e \n' % \
                      (o.GroupID, o.masses['dm'], o.masses['stellar'],
                       o.masses['gas'],o.radii['total'], o.gas_fraction,
                       o.local_number_density, cgsm)
            cnt += 1
            if cnt > top: break
    elif group_type == 'galaxy':
        output += ' ID    Mstar     Mgas      SFR       r         fgas   nrho      Central\t|  Mhalo     HID\n'
        output += ' ----------------------------------------------------------------------------------------\n'
        #         ' 0000  4.80e+09  4.80e+09  4.80e+09  7.64e-09  0.000  7.64e-09  False
        for o in group_list:
            phm, phid = -1, -1
            if o.halo is not None: phm, phid = o.halo.masses['total'], o.halo.GroupID
            output += ' %04d  %0.2e  %0.2e  %0.2e  %0.2e  %0.3f  %0.2e  %s\t|  %0.2e  %d \n' % \
                      (o.GroupID, o.masses['stellar'], o.masses['gas'],
                       o.sfr, o.radii['total'], o.gas_fraction,
                       o.local_number_density, o.central,
                       phm, phid)
            cnt += 1
            if cnt > top: break
    elif group_type == 'cloud':
        output += ' ID    Mstar     Mgas      SFR       r         fgas   nrho      Central\t|  Mhalo     HID\n'
        output += ' ----------------------------------------------------------------------------------------\n'
        #         ' 0000  4.80e+09  4.80e+09  4.80e+09  7.64e-09  0.000  7.64e-09  False
        for o in group_list:
            phm, phid = -1, -1
            output += ' %04d  %0.2e  %0.2e  %0.2e  %0.2e  %0.3f  %0.2e  %s\t|  %0.2e  %d \n' % \
                      (o.GroupID, o.masses['stellar'], o.masses['gas'],
                       o.sfr, o.radii['total'], o.gas_fraction,
                       o.local_number_density, o.central,
                       phm, phid)
            cnt += 1
            if cnt > top: break


            
    print(output)
