import os
import h5py
import numpy as np

from yt.extern import six
from yt.units.yt_array import YTQuantity, YTArray

blacklist = [
    'G', 'initial_mass',
    'valid', 'vel_conversion',
    'unbound_particles', '_units',
    'unit_registry_json',
    'unbound_indexes',
    'lists','dicts'
]

######################################################################

def _write_dataset(key, data, hd):
    hd.create_dataset(key, data=data)

def check_and_write_dataset(obj, key, hd):
    """General function for writing an HDF5 dataset.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Main caesar object to save.
    key : str
        Name of dataset to write.
    hd : h5py.Group
        Open HDF5 group.

    """
    if not hasattr(obj, key): return
    if isinstance(getattr(obj, key), int): return
    _write_dataset(key, getattr(obj, key), hd)

######################################################################

def serialize_list(obj_list, key, hd):
    """Function that serializes a index list (glist/etc) for objects.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Main caesar object.
    key : str
        Name of the index list.
    hd : h5py.Group
        Open HDF5 group.

    """
    if key in blacklist: return
    if not hasattr(obj_list[0], key): return
    data = _get_serialized_list(obj_list, key)
    _write_dataset(key, data, hd)

def _get_serialized_list(obj_list, key):
    tmp, index = [], 0
    for i in obj_list:
        current_list = getattr(i, key)
        n = len(current_list)
        tmp.extend(current_list)

        setattr(i, '%s_start' % key, index)
        index += n
        setattr(i, '%s_end'   % key, index)
    return tmp

######################################################################

def serialize_attributes(obj_list, hd, hd_dicts):
    """Function that goes through a list full of halos/galaxies/clouds and
    serializes their attributes.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Main caesar object.
    hd : h5py.Group
        Open HDF5 group for lists.
    hd_dicts : h5py.Group
        Open HDF5 group for dictionaries.

    """
    for k,v in six.iteritems(obj_list[0].__dict__):
        if k in blacklist: continue

        if isinstance(v, dict):
            _write_dict(obj_list, k, v, hd_dicts)
        else:
            _write_attrib(obj_list, k, v, hd)

def _write_attrib(obj_list, k, v, hd):
    unit = False
    if isinstance(v, YTQuantity):
        data = [getattr(i,k).d for i in obj_list]
        unit = True
    elif isinstance(v, YTArray):
        if np.shape(v)[0] == 3:
            data = np.vstack((getattr(i,k).d for i in obj_list))
        else:
            data = [getattr(i,k).d for i in obj_list]
        unit = True
    elif isinstance(v, np.ndarray) and np.shape(v)[0] == 3 and k is not 'bhlist' and k is not '_Group__glist':
        try:
            data = np.vstack((getattr(i,k) for i in obj_list))
        except:
            print 'Warning: saver unable to stack:',k,v,np.shape(v)
            return
    elif isinstance(v, (int, float, bool, np.number)):
        data = [getattr(i,k) for i in obj_list]
    else:
        return

    _write_dataset(k, data, hd)
    if unit:
        hd[k].attrs.create('unit', str(v.units).encode('utf8'))
            
def _write_dict(obj_list, k, v, hd):
    for kk,vv in six.iteritems(v):
        unit = False        
        if isinstance(vv, (YTQuantity, YTArray)):
            data = np.array([getattr(i,k)[kk].d for i in obj_list])
            unit = True
        else:
            data = np.array([getattr(i,k)[kk] for i in obj_list])            

        _write_dataset('%s.%s' % (k,kk), data, hd)
        if unit:
            hd['%s.%s' % (k,kk)].attrs.create('unit', str(vv.units).encode('utf8'))        

######################################################################

def serialize_global_attribs(obj, hd):
    """Function that goes through a caesar object and saves general 
    attributes.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Main caesar object.
    hd : h5py.File
        Open HDF5 dataset.

    """
    units = {}
    for k,v in six.iteritems(obj.__dict__):
        if k in blacklist: continue

        if isinstance(v, (YTQuantity, YTArray)):
            hd.attrs.create(k, v.d)
            units[k] = v.units
        elif isinstance(v, str):
            hd.attrs.create(k, v.encode('utf8'))
        elif isinstance(v, (int, float, bool, np.number)):
            hd.attrs.create(k, v)
        #else:
        #    print(k,type(v))

    if len(units) > 0:
        uhd = hd.create_group('global_attribute_units')
        for k,v in six.iteritems(units):
            uhd.attrs.create(k, str(v).encode('utf8'))
            
######################################################################
    
def save(obj, filename='test.hdf5'):
    """Function to save a CAESAR file to disk.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Main caesar object to save.
    filename : str, optional
        Filename of the output file.

    Examples
    --------
    >>> obj.save('output.hdf5')
    
    """
    from yt.funcs import mylog

    try:
        from caesar.__hg_version__ import hg_version
    except:
        hg_version = None

    if hg_version is None:
        hg_version = 'Unknown'
        
    if os.path.isfile(filename):
        mylog.warning('%s already present, overwriting!' % filename)
        os.remove(filename)
    mylog.info('Writing %s' % filename)
        
    outfile = h5py.File(filename, 'w')
    outfile.attrs.create('caesar', 315)
    
    
    unit_registry = obj.yt_dataset.unit_registry.to_json()
    outfile.attrs.create('unit_registry_json', unit_registry.encode('utf8'))

    serialize_global_attribs(obj, outfile)
    obj.simulation._serialize(obj, outfile)

    if hasattr(obj, 'halos') and obj.nhalos > 0:
        hd   = outfile.create_group('halo_data')
        hdd  = hd.create_group('lists')
        hddd = hd.create_group('dicts')

        # gather
        index_lists = ['galaxy_index_list', 'dmlist', 'glist', 'slist']
        if obj.data_manager.blackholes:
            index_lists.append('bhlist')
        if obj.data_manager.dust:
            index_lists.append('dlist')

        #write        
        for vals in index_lists:
            serialize_list(obj.halos, vals, hdd)
        serialize_attributes(obj.halos, hd, hddd)
  
    if hasattr(obj, 'galaxies') and obj.ngalaxies > 0:
        hd   = outfile.create_group('galaxy_data')
        hdd  = hd.create_group('lists')
        hddd = hd.create_group('dicts')

        # gather
        index_lists = ['glist', 'slist','cloud_index_list']
        if obj.data_manager.blackholes:
            index_lists.append('bhlist')
        if obj.data_manager.dust:
            index_lists.append('dlist')

        # write
        for vals in index_lists:
            serialize_list(obj.galaxies, vals, hdd)
        serialize_attributes(obj.galaxies, hd, hddd)


    if hasattr(obj, 'clouds') and obj.nclouds > 0:
        hd   = outfile.create_group('cloud_data')
        hdd  = hd.create_group('lists')
        hddd = hd.create_group('dicts')

        # gather
        index_lists = ['glist']
 
        # write
        for vals in index_lists:
            serialize_list(obj.clouds, vals, hdd)
        serialize_attributes(obj.clouds, hd, hddd)


        
    if hasattr(obj, 'global_particle_lists'):
        hd = outfile.create_group('global_lists')

        # gather
        global_index_lists = ['halo_dmlist','halo_glist','halo_slist',
                              'galaxy_glist','galaxy_slist','cloud_glist']
        if obj.data_manager.blackholes:
            global_index_lists.extend(['halo_bhlist','galaxy_bhlist'])
        if obj.data_manager.dust:
            global_index_lists.extend(['halo_dlist','galaxy_dlist'])

        # write
        for vals in global_index_lists:
            check_and_write_dataset(obj.global_particle_lists, vals, hd)
            
    outfile.close()
