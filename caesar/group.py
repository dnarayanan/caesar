import six
import numpy as np

from .property_getter import ptype_ints

MINIMUM_STARS_PER_GALAXY = 32
MINIMUM_DM_PER_HALO      = 32

class Group(object):
    def __init__(self,obj):
        self.particle_indexes = []
        self.obj = obj

        self.masses = {}
        self.temperatures = {}
        self.radii = {}        

    @property
    def valid(self):
        valid = True
        if self.obj_type == 'halo' and len(self.dmlist) < MINIMUM_DM_PER_HALO:
            valid = False
        elif self.obj_type == 'galaxy' and len(self.slist) < MINIMUM_STARS_PER_GALAXY:
            valid = False
        return valid

    def _delete_attribute(self,a):
        if hasattr(self,a):
            delattr(self,a)
    
    def _cleanup(self):
        self._delete_attribute('particle_data')
        self._delete_attribute('particle_indexes')
    
    def _process_group(self,pdata):
        self._assign_particle_data(pdata)
        self._assign_local_indexes()

        if self.valid:
            self._calculate_masses()
            self._calculate_center_of_mass_quantities()
            self._assign_global_plists()
            
        self._cleanup()
        
    def _assign_particle_data(self,pdata):
        """ Use self.particle_indexes to assign group particle data """
        self.particle_data = {}
        for k,v in six.iteritems(pdata):
            self.particle_data[k] = v[self.particle_indexes]

    def _assign_local_indexes(self):
        """ glist/slist/dmlist indexes correspond to the LOCAL particle data """
        self.glist  = np.where(self.particle_data['ptype'] == ptype_ints['gas'])[0] 
        self.slist  = np.where(self.particle_data['ptype'] == ptype_ints['star'])[0]
        self.dmlist = np.where(self.particle_data['ptype'] == ptype_ints['dm'])[0]
        
    def _assign_global_plists(self):
        """ glist/slist/dmlist indexes correspond to the GLOBAL particle data """
        if isinstance(self.particle_indexes, list):
            self.particle_indexes = np.array(self.particle_indexes)
        self.glist  = self.particle_indexes[self.glist]
        self.slist  = self.particle_indexes[self.slist]
        self.dmlist = self.particle_indexes[self.dmlist]

    def _calculate_masses(self):
        """ calculate various masses """
        mass_total  = np.sum(self.particle_data['mass'])
        mass_dm     = np.sum(self.particle_data['mass'][self.dmlist])
        mass_gas    = np.sum(self.particle_data['mass'][self.glist])
        mass_star   = np.sum(self.particle_data['mass'][self.slist])
        mass_baryon = mass_gas + mass_star

        self.masses['total']   = self.obj.yt_dataset.quan(mass_total, self.obj.units['mass'])
        self.masses['dm']      = self.obj.yt_dataset.quan(mass_dm, self.obj.units['mass'])
        self.masses['gas']     = self.obj.yt_dataset.quan(mass_gas, self.obj.units['mass'])
        self.masses['stellar'] = self.obj.yt_dataset.quan(mass_star, self.obj.units['mass'])
        self.masses['baryon']  = self.obj.yt_dataset.quan(mass_baryon, self.obj.units['mass'])
            
    def _calculate_center_of_mass_quantities(self):
        """ calculate center-of-mass position and velocity """
        def get_center_of_mass_quantity(quantity):  ## REFACTOR ME TO BE MORE GENERIC WITH SHAPE
            val  = np.zeros(3)
            for i in range(0,3):
                val[i] = np.sum(self.particle_data['mass'] * self.particle_data[quantity][:,i])
            val /= self.masses['total'].d
            return val

        self.pos = self.obj.yt_dataset.arr(get_center_of_mass_quantity('pos'), self.obj.units['length'])
        self.vel = self.obj.yt_dataset.arr(get_center_of_mass_quantity('vel'), self.obj.units['velocity'])
        
class Galaxy(Group):
    obj_type = 'galaxy'    
    def __init__(self,obj):
        super(Galaxy, self).__init__(obj)
        self.central = False
        
class Halo(Group):
    obj_type = 'halo'    
    def __init__(self,obj):
        super(Halo, self).__init__(obj)
        self.child = False

def create_new_group(obj, group_type):
    if group_type == 'halo':
        return Halo(obj)
    elif group_type == 'galaxy':
        return Galaxy(obj)
