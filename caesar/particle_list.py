
class ParticleList(object):
    """Class for a generic particle list.  Contains methods for 
    retrieving data from disk if it is not already loaded.
    """
    def __init__(self, name):
        self.name = name

    def __get__(self, instance, owner):
        if isinstance(getattr(instance, '_%s' % self.name), int):
            instance.restore_global_list(self.name)
        return getattr(instance, '_%s' % self.name)

    def __set__(self, instance, value):
        setattr(instance, '_%s' % self.name, value)

class ParticleListContainer(object):
    """Container that holds global particle/field index lists."""
    halo_dmlist   = ParticleList('halo_dmlist')
    halo_glist    = ParticleList('halo_glist')
    halo_slist    = ParticleList('halo_slist')
    halo_bhlist   = ParticleList('halo_bhlist')
    
    galaxy_glist  = ParticleList('galaxy_glist')
    galaxy_slist  = ParticleList('galaxy_slist')
    galaxy_bhlist = ParticleList('galaxy_bhlist')

    cloud_glist = ParticleList('cloud_glist')
    
    def __init__(self, obj):
        self.halo_dmlist   = 0
        self.halo_glist    = 0
        self.halo_slist    = 0
        self.halo_bhlist   = 0
        self.galaxy_glist  = 0
        self.galaxy_slist  = 0
        self.galaxy_bhlist = 0
        self.cloud_glist = 0
        
        self.obj = obj

    def restore_global_list(self, key):        
        if not hasattr(self.obj, 'data_file'):
            return

        import h5py
        import numpy as np
        print('restoring %s' % key)
        with h5py.File(self.obj.data_file, 'r') as infile:
            if 'global_lists/%s' % key in infile:
                data = np.array(infile['global_lists/%s' % key])
                setattr(self, key, data)
            else:
                from yt.funcs import mylog
                mylog.warning('%s global list not present.' % key)
                return 
