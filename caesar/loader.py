import h5py
import numpy as np

from caesar.group import Halo, Galaxy, Cloud
from caesar.saver import blacklist

from yt.extern import six
from yt.units.yt_array import YTQuantity, YTArray, UnitRegistry

LOAD_OBJECT_LISTS = True

######################################################################

def restore_single_list(obj, group, key):
    """Function for restoring a single list.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Main CAESAR object.
    group : object
        Object we are restoring the list to.
    key : str
        Name of the list to restore.

    """
    if LOAD_OBJECT_LISTS: return
    infile = h5py.File(obj.data_file,'r')
    data   = infile['%s_data/lists/%s' % (group.obj_type, key)][:]
    infile.close()
    start  = getattr(group, '%s_start' % key)
    end    = getattr(group, '%s_end'   % key)
    setattr(group, key, data[start:end])    

######################################################################

def get_unit_quant(v, data):
    """Function for restoring CAESAR global attributes.

    Parameters
    ----------
    v : h5py object
        Object to check for unit attributes.
    data : object
        The actual data.

    Returns
    -------
    unit, quant : str, boolean
        Returns the unit string and a boolean.  The boolean says if 
        the returned value should be a quantity (True) or an array
        (False).

    """
    unit  = None
    quant = True
    if 'unit' in v.attrs:
        unit = v.attrs['unit'].decode('utf8')
        if len(np.shape(data)) > 1:
            quant = False
    return unit, quant
                
######################################################################

def restore_global_attributes(obj, hd):
    """Function for restoring CAESAR global attributes.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Main CAESAR object.
    hd : h5py.Group
        Open HDF5 dataset.

    """
    for k,v in six.iteritems(hd.attrs):
        if k in blacklist: continue
        setattr(obj, k, v)

    if 'global_attribute_units' in hd:
        uhd = hd['global_attribute_units']
        for k,v in six.iteritems(uhd.attrs):
            setattr(obj, k, YTQuantity(getattr(obj, k), v, registry=obj.unit_registry))

######################################################################

def restore_object_list(obj_list, key, hd):
    """Function for restoring halo/galaxy/cloud sublists.

    Parameters
    ----------
    obj_list : list
        List of objects we are restoring attributes to.
    key : str
        Name of list to restore.
    hd : h5py.Group
        Open HDF5 dataset.

    """
    if not LOAD_OBJECT_LISTS and key is not 'galaxy_index_list': return
    #if not LOAD_OBJECT_LISTS and key is not 'cloud_index_list': return
        
    if ('lists/%s' % key) not in hd: return
    if key in blacklist: return

    data = hd['lists/%s' % key][:]
    for i in obj_list:
        start = getattr(i, '%s_start' % key)
        end   = getattr(i, '%s_end'   % key)
        setattr(i, key, data[start:end])
        delattr(i, '%s_start' % key)
        delattr(i, '%s_end'   % key)

######################################################################                    

def restore_object_dicts(obj_list, hd, unit_reg):
    """Function for restoring halo/galaxy/cloud dictionary attributes.

    Parameters
    ----------
    obj_list : list
        List of objects we are restoring attributes to.
    hd : h5py.Group
        Open HDF5 dataset.
    unit_reg : yt unit registry
        Unit registry.    

    """
    if 'dicts' not in hd: return
    hdd = hd['dicts']
    for k,v in six.iteritems(hdd):
        data = np.array(v)

        unit, use_quant = get_unit_quant(v, data)               
        
        dict_name, dict_key = k.split('.')
        if dict_key in blacklist: continue
        for i in range(0,len(obj_list)):
            if not hasattr(obj_list[i], dict_name):
                setattr(obj_list[i], dict_name, {})
            cur_dict = getattr(obj_list[i], dict_name)

            if unit is not None:
                if use_quant:
                    cur_dict[dict_key] = YTQuantity(data[i], unit, registry=unit_reg)
                else:
                    cur_dict[dict_key] = YTArray(data[i], unit, registry=unit_reg)
            else:
                cur_dict[dict_key] = data[i]
            setattr(obj_list[i], dict_name, cur_dict)
            
######################################################################

def restore_object_attributes(obj_list, hd, unit_reg):
    """Function for restoring halo/galaxy/cloud attributes.

    Parameters
    ----------
    obj_list : list
        List of objects we are restoring attributes to.
    hd : h5py.Group
        Open HDF5 dataset.
    unit_reg : yt unit registry
        Unit registry.    

    """
    for k,v in six.iteritems(hd):
        if k in blacklist: continue
        if k == 'lists' or k == 'dicts': continue
        data = np.array(v)

        unit, use_quant = get_unit_quant(v, data)
            
        for i in range(0,len(obj_list)):
            if unit is not None:
                if use_quant:
                    setattr(obj_list[i], k, YTQuantity(data[i], unit, registry=unit_reg))
                else:
                    setattr(obj_list[i], k, YTArray(data[i], unit, registry=unit_reg))
            else:
                setattr(obj_list[i], k, data[i])

######################################################################

def load(filename, ds = None, obj = None, load_limit = None, LoadHalo=1):
    """Function to load a CAESAR object from disk.

    Parameters
    ----------
    filename : str
        Input file.
    ds : yt dataset, optional
        yt dataset to link to.
    obj : :class:`main.CAESAR`, optional
        For loading into an already created CAESAR object.
    LoadHalo: int
        Option to load halo information: 0=No; 1=Yes; 2=Yes, but not mass/radius info


    Examples
    --------
    >>> import caesar
    >>> import yt
    >>> ds  = yt.load('snapshot')
    >>> obj = caesar.load(ds)

    """
    from yt.funcs import mylog

    mylog.info('Reading %s' % filename)
    try:
        infile = h5py.File(filename, 'r')
    except IOError:
        raise

    if obj is None:
        import os
        from caesar.main import CAESAR
        obj = CAESAR()
        obj.data_file = os.path.abspath(filename)

    unit_registry_json = infile.attrs['unit_registry_json']
    obj.unit_registry  = UnitRegistry.from_json(unit_registry_json.decode('utf8'))
    
    restore_global_attributes(obj, infile)
    obj.simulation._unpack(obj, infile)
    
    # restore halo data
    if 'halo_data' in infile and LoadHalo:
        mylog.info('Restoring halo attributes')
        hd = infile['halo_data']
        obj.halos = []
        if load_limit is None:
            for i in range(0, obj.nhalos):
                obj.halos.append(Halo(obj))
        else:
            for i in range(0, min(load_limit, obj.nhalos)):
                obj.halos.append(Halo(obj))
        restore_object_attributes(obj.halos, hd, obj.unit_registry)
        restore_object_dicts(obj.halos, hd, obj.unit_registry)
        restore_object_list(obj.halos, 'galaxy_index_list', hd)

        if LoadHalo==1:
            #Compute the virial/200/500/2500 masses
            if hasattr(obj.simulation, 'Densities'):
                PiFac = 4./3.*np.pi
                h = obj.halos[0]
                for h in obj.halos:
                    h.masses['virial'] = obj.simulation.Densities[0]*PiFac*(h.radii['virial']*h.radii['virial']*h.radii['virial'])
                    h.masses['m200c'] = (obj.simulation.Densities[1]*PiFac*(h.radii['r200c']*h.radii['r200c']*h.radii['r200c'])).to('Msun')
                    h.masses['m500c'] = (obj.simulation.Densities[2]*PiFac*(h.radii['r500c']*h.radii['r500c']*h.radii['r500c'])).to('Msun')
                    h.masses['m2500c'] = (obj.simulation.Densities[3]*PiFac*(h.radii['r2500c']*h.radii['r2500c']*h.radii['r2500c'])).to('Msun')

        # optional
        for vals in ['dmlist', 'glist', 'slist', 'bhlist','dlist']:
            restore_object_list(obj.halos, vals, hd)
   
    # restore galaxy data
    if 'galaxy_data' in infile:
        mylog.info('Restoring galaxy attributes')
        hd = infile['galaxy_data']
        obj.galaxies = []
        if load_limit is None:
            for i in range(0, obj.ngalaxies):
                obj.galaxies.append(Galaxy(obj))
        else:
            for i in range(0, min(load_limit, obj.ngalaxies)):
                obj.galaxies.append(Galaxy(obj))

        restore_object_attributes(obj.galaxies, hd, obj.unit_registry)
        restore_object_dicts(obj.galaxies, hd, obj.unit_registry)
        restore_object_list(obj.galaxies, 'cloud_index_list', hd)

        # optional
        for vals in ['glist', 'slist', 'bhlist','dlist']:
            restore_object_list(obj.galaxies, vals, hd)



    if 'cloud_data' in infile:
        mylog.info('Restoring cloud attributes')
        hd = infile['cloud_data']
        obj.clouds = []
        if load_limit is None:
            for i in range(0, obj.nclouds):
                obj.clouds.append(Cloud(obj))
        else:
            for i in range(0, min(load_limit, obj.nclouds)):
                obj.clouds.append(Cloud(obj))

        restore_object_attributes(obj.clouds, hd, obj.unit_registry)
        restore_object_dicts(obj.clouds, hd, obj.unit_registry)

        # optional
        for vals in ['glist', 'slist', 'bhlist','dlist']:
            restore_object_list(obj.clouds, vals, hd)

    infile.close()

    if LoadHalo:
        obj._link_objects(load_limit=load_limit)

    if ds is not None:
        obj.yt_dataset = ds

    return obj

