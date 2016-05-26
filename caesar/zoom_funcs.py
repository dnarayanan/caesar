import numpy as np
from yt.funcs import mylog

def write_IC_mask(group, ic_ds, filename, search_factor, print_extents=True):
    """Write MUSIC initial condition mask to disk.

    Parameters
    ----------
    group : :class:`group.Group`
        Group we are querying.
    ic_ds : yt dataset
        The initial condition dataset via ``yt.load()``.
    filename : str
        The filename of which to write the mask to.  If a full path is
        not supplied then it will be written in the current directory.
    search_factor : float, optional
        How far from the center to select DM particles 
        (defaults to 2.5)
    print_extents : bool, optional
        Print MUSIC extents for cuboid after mask creation
    
    Examples
    --------
    >>> import yt
    >>> import caesar
    >>>
    >>> snap = 'my_snapshot.hdf5'
    >>> ic   = 'IC.dat'
    >>>
    >>> ds    = yt.load(snap)
    >>> ic_ds = yt.load(ic)
    >>>
    >>> obj = caesar.load('caesar_my_snapshot.hdf5', ds)
    >>> obj.galaxies[0].write_IC_mask(ic_ds, 'mymask.txt')
    
    """    
    ic_dmpos = get_IC_pos(group, ic_ds, search_factor=search_factor,
                          return_mask=True)
    mylog.info('Writing IC mask to %s' % filename)
    f = open(filename, 'w')
    for i in range(0, len(ic_dmpos)):
        f.write('%e %e %e\n' % (ic_dmpos[i,0], ic_dmpos[i,1], ic_dmpos[i,2]))
    f.close()

    def get_extents(dim):
        vals = np.sort(ic_dmpos[:,dim])
        break_point = -1.0
        for i in range(1,len(vals)):
            ddim = vals[i] - vals[i-1]
            if ddim > 0.1:
                break_point = vals[i-1] + (ddim/2.0)
                break
        if break_point != -1:
            l_indexes = np.where(icdmpos[:,dim] < break_point)[0]
            ic_dmpos[l_indexes, dim] += 1.0

        dmin = np.min(ic_dmpos[:,dim])
        dmax = np.max(ic_dmpos[:,dim])
        dext = dmax - dmin

        return dmin,dext

    if print_extents:
        xcen, xext = get_extents(0)
        ycen, yext = get_extents(1)
        zcen, zext = get_extents(2)

        mylog.info('MUSIC cuboid settings:')
        mylog.info('ref_center = %0.3f,%0.3f,%0.3f' % (xcen,ycen,zcen))
        mylog.info('ref_extent = %0.3f,%0.3f,%0.3f' % (xext,yext,zext))

        if xext >= 0.5 or yext >= 0.5 or zext >= 0.5:
            mylog.warning('REGION EXTENDS MORE THAN HALF OF YOUR VOLUME')
            
        
def get_IC_pos(group, ic_ds, search_factor=2.5, return_mask=False):
    """Get the initial dark matter positions of a ``CAESAR`` halo.

    If called on a galaxy, it will return the IC DM positions of the
    parent halo.
    
    Parameters
    ----------
    group : :class:`group.Group`
        Group we are querying.
    ic_ds : yt dataset
        The initial condition dataset via ``yt.load()``
    search_factor : float, optional
        How far from the center to select DM particles (defaults to 2.5).
    return_mask : bool, optional
        Return initial condition positions from 0-->1 rather than raw 
        data.  Useful for writing a MUSIC mask file.
    
    Returns
    -------
    ic_dmpos : np.ndarray
        DM positions of this object in the initial condition file.
    
    """    
    from caesar.property_getter import ptype_aliases, get_property, DatasetType
    from caesar.periodic_kdtree import PeriodicCKDTree

    ic_ds_type = ic_ds.__class__.__name__
    if ic_ds_type not in ptype_aliases:
        raise NotImplementedError('%s not yet supported' % ic_ds_type)
    if group.obj.yt_dataset.domain_width[0].d != ic_ds.domain_width[0].d:
        raise Exception('IC and SNAP boxes do not match! (%f vs %f)' %
                        (ic_ds.domain_width[0].d,
                         group.obj.yt_dataset.domain_width[0].d))
    if str(ic_ds.length_unit) != str(group.obj.yt_dataset.length_unit):
        raise Exception('LENGTH UNIT MISMATCH! '\
                        'This may arise from loading the snap/IC '\
                        'incorrectly and WILL cause problems with '\
                        'the matching process. (%s vs %s)' %
                        (str(ic_ds.length_unit), str(group.obj.yt_dataset.length_unit)))
        
    if group.obj_type == 'halo':
        obj = group
    elif group.obj_type == 'galaxy':
        if group.halo is None:
            mylog.warning('Galaxy %d has no halo!' % group.GroupID)
            return
        obj = group.halo

    search_params = dict(
        pos = obj.pos.in_units('code_length').d,
        r   = obj.radii['total'].in_units('code_length').d * search_factor,
    )
        
    box    = ic_ds.domain_width[0].d
    bounds = np.array([box,box,box])

    dmpids = get_property(obj.obj, 'pid', 'dm').d
    dmpos  = get_property(obj.obj, 'pos', 'dm').d
    
    dm_TREE = PeriodicCKDTree(bounds, dmpos)

    valid = dm_TREE.query_ball_point(search_params['pos'], search_params['r'])
    search_params['ids'] = dmpids[valid]

    ic_ds_type = DatasetType(ic_ds)
    ic_dmpos   = ic_ds_type.get_property('dm', 'pos').d
    ic_dmpids  = ic_ds_type.get_property('dm', 'pid').d

    matches  = np.in1d(ic_dmpids, search_params['ids'], assume_unique=True)
    nmatches = len(np.where(matches)[0])
    nvalid   = len(valid)
    if nmatches != nvalid:
        raise Exception('Could not match all particles! '\
                        'Only %0.2f%% particles matched.' %
                        (float(nmatches)/float(nvalid) * 100.0))

    mylog.info('MATCHED %d particles from %s %d in %s' %
               (nmatches, obj.obj_type, obj.GroupID, ic_ds.basename))
    mylog.info('Returning %0.2f%% of the total DM from the sim' %
               (float(nmatches)/float(len(ic_dmpids)) * 100.0))
    
    matched_pos = ic_dmpos[matches]

    if return_mask:
        matched_pos /= box

    return matched_pos


def construct_lowres_tree(group, lowres):
    """Construct a periodic KDTree for low-resolution particles.

    Parameters
    ----------
    group : :class:`group.Group`
        Group we are querying.
    lowres : list
        Particle types to be considered low-resolution.  Typically
        [2,3,5]

    Notes
    -----
    Assigns the dict ``_lowres`` to the :class:`main.CAESAR` object.

    """
    obj = group.obj
    if hasattr(obj, '_lowres') and obj._lowres['ptypes'] == lowres:
        return
    if not hasattr(obj, '_ds_type'):
        raise Exception('No yt_dataset assigned!')

    mylog.info('Gathering low-res particles and constructing tree')
    from caesar.property_getter import get_property
    
    lr_pos  = np.empty((0,3))
    lr_mass = np.empty(0)

    pos_unit  = group.pos.units
    mass_unit = group.masses['total'].units
    
    for p in lowres:
        ptype = 'PartType%d' % p
        if ptype in obj.yt_dataset.particle_fields_by_type:
            cur_pos  = obj._ds_type.dd[ptype, 'particle_position'].to(pos_unit)
            cur_mass = obj._ds_type.dd[ptype, 'particle_mass'].to(mass_unit)

            lr_pos  = np.append(lr_pos,  cur_pos.d, axis=0)
            lr_mass = np.append(lr_mass, cur_mass.d, axis=0)
    
    from caesar.periodic_kdtree import PeriodicCKDTree
    box    = obj.simulation.boxsize.to(pos_unit)
    bounds = np.array([box,box,box])

    obj._lowres = dict(
        TREE   = PeriodicCKDTree(bounds, lr_pos),
        MASS   = lr_mass,
        ptypes = lowres
    )


def all_object_contam_check(obj):
    if obj._ds_type.ds_type != 'GizmoDataset' and obj._ds_type.ds_type != 'GadgetHDF5Dataset':
        return
    if not 'lowres' in obj._kwargs or obj._kwargs['lowres'] is None:
        return        

    lowres = obj._kwargs['lowres']
    if not isinstance(lowres, list):
        raise Exception('lowres must be a list!')    
    
    mylog.info('Checking all objects for contamination.  Lowres Types: %s' % lowres)
    if hasattr(obj, 'halos'):
        for h in obj.halos:
            h.contamination_check(lowres, printer=False)
    if hasattr(obj, 'galaxies'):
        for g in obj.galaxies:
            if g.halo is None:
                g.contamination = None
            else:
                g.contamination = g.halo.contamination
        
