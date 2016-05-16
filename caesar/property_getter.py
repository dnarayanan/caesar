# RENAME TO PROPERTY_MANAGER?

import numpy as np
from yt.units.yt_array import YTArray

particle_data_aliases = {
    'pos':'particle_position',
    'vel':'particle_velocity',
    'rho':'density',
    'hsml':'smoothing_length',
    'sfr':'StarFormationRate',
    'mass':'particle_mass',
    'u':'thermal_energy',
    'temp':'Temperature',
    'temperature':'Temperature',
    'ne':'ElectronAbundance',
    'nh':'NeutralHydrogenAbundance',
    'pid':'particle_index',
    'fh2':'FractionH2',
    'metallicity':'metallicity',
    'age':'StellarFormationTime'
}

grid_gas_aliases = {
    'mass':'cell_mass',
    'rho':'density',
    'temp':'temperature',
    'metallicity':'metallicity',
    'pos':'x',
    'vel':'velocity_x',
}

ptype_ints = dict(
    gas=0,
    star=4,
    dm=1,
    bh=5
)

ptype_aliases = dict(
    GadgetHDF5Dataset = {'gas':'PartType0','star':'PartType4','dm':'PartType1','bh':'PartType5'},
    GadgetDataset     = {'gas':'Gas','star':'Stars','dm':'Halo'},
    TipsyDataset      = {'gas':'Gas','star':'Stars','dm':'DarkMatter'},
    EnzoDataset       = {'gas':'gas','star':'io','dm':'io'},
)

class DatasetType(object):
    def __init__(self, ds):
        self.ds      = ds
        self.ds_type = ds.__class__.__name__
        self.dd      = ds.all_data()

        if self.ds_type not in ptype_aliases.keys():
            raise NotImplementedError('%s not yet supported' % ds_type)

        self.ptype_aliases = ptype_aliases[self.ds_type]

        self.indexes = 'all'
        
        if self.ds_type == 'EnzoDataset':
            self.grid = True
        else:
            self.grid = False

    def has_ptype(self, requested_ptype):
        """ Returns True/False if requested ptype is present """
        requested_ptype = requested_ptype.lower()
        if requested_ptype in self.ptype_aliases.keys():
            ptype = self.ptype_aliases[requested_ptype]
            if requested_ptype == 'gas' and self.grid:
                for field in self.ds.derived_field_list:
                    if field[0] == ptype:
                        return True
            else:
                if ptype in self.ds.particle_fields_by_type:
                    return True
                for field in self.ds.derived_field_list:
                    if field[0] == ptype:
                        return True
        return False

    def get_ptype_name(self, requested_ptype):
        if not self.has_ptype(requested_ptype):
            raise NotImplementedError('Could not find %s ptype!' % requested_ptype)
        return self.ptype_aliases[requested_ptype.lower()]

    def get_property_name(self, requested_ptype, requested_prop):
        prop  = requested_prop.lower()
        ptype = requested_ptype.lower()
        if ptype == 'gas' and self.grid:
            if prop in grid_gas_aliases.keys():
                return grid_gas_aliases[prop]
        else:
            if prop in particle_data_aliases.keys():
                return particle_data_aliases[prop]
        return prop
            
    def has_property(self, requested_ptype, requested_prop):
        prop  = self.get_property_name(requested_ptype, requested_prop)
        ptype = self.get_ptype_name(requested_ptype)

        if ptype in self.ds.particle_fields_by_type:
            fields = self.ds.particle_fields_by_type[ptype]
            if prop in fields:
                return True
    
        fields = self.ds.derived_field_list
        for f in fields:
            if f[0] == ptype and f[1] == prop:
                return True

        return False


    def get_property(self, requested_ptype, requested_prop):
        if not self.has_ptype(requested_ptype):
            raise NotImplementedError('ptype %s not found!' % requested_ptype)
        if not self.has_property(requested_ptype, requested_prop):
            raise NotImplementedError('prop %s not found!' % requested_prop)

        ptype = self.get_ptype_name(requested_ptype)
        prop  = self.get_property_name(requested_ptype, requested_prop)

        if self.ds_type == 'EnzoDataset' and requested_ptype != 'gas':
            self._set_indexes_for_enzo(ptype, requested_ptype)
            
        if self.grid and (requested_prop == 'pos' or requested_prop == 'vel'):
            data = self._get_gas_grid_posvel(requested_prop)
        else:
            data = self.dd[ptype, prop]
            
        if not isinstance(self.indexes, str):
            data = data[self.indexes]
            self.indexes = 'all'
        return data
        
    def _get_gas_grid_posvel(self,request):
        if request == 'pos':
            x = self.dd['gas','x']
            y = self.dd['gas','y']
            z = self.dd['gas','z']
            return self.ds.arr(np.column_stack((x.d,y.d,z.d)), x.units)
        elif request == 'vel':
            vx = self.dd['gas','velocity_x']
            vy = self.dd['gas','velocity_y']
            vz = self.dd['gas','velocity_z']
            return self.ds.arr(np.column_stack((vx.d,vy.d,vz.d)), vx.units)
            
    def _set_indexes_for_enzo(self, proper_ptype, requested_ptype):
        ptype_vals = dict(gas=0, dm=1, star=2)
        if ptype_vals[requested_ptype] > 0:
            self.indexes = np.where(self.dd[proper_ptype, 'particle_type'] == ptype_vals[requested_ptype])[0]
            

"""    
def filter_enzo_results(obj, data, ptype, requested_ptype):
    ptype_val = -1
    if requested_ptype == 'gas':
        ptype_val = 0
    elif requested_ptype == 'dm':
        ptype_val = 1
    elif requested_ptype == 'star':
        ptype_val = 2

    if ptype_val > 0:
        indexes = np.where(obj._ds_type.dd[ptype, 'particle_type'] == ptype_val)[0]
        data = data[indexes]
    return data
"""

def has_ptype(obj, requested_ptype):
    return obj._ds_type.has_ptype(requested_ptype)
        
def has_property(obj, requested_ptype, requested_prop):
    return obj._ds_type.has_property(requested_ptype, requested_prop)

def get_property(obj, requested_prop, requested_ptype):
    ds_type = obj._ds_type    
    return obj._ds_type.get_property(requested_ptype, requested_prop)


def get_high_density_gas_indexes(obj):
    nH_thresh = 0.13

    rho  = get_property(obj, 'rho', 'gas').in_cgs()
    rho /= 1.67262178e-24      # to atoms/cm^3
    rho *= obj.simulation.XH   # to hydrogen atoms / cm^3

    indexes = np.where(rho >= nH_thresh)[0]
    return indexes

def get_particles_for_FOF(obj, ptypes, find_type=None):

    pos  = np.empty((0,3))
    vel  = np.empty((0,3))
    mass = np.empty(0)

    ptype   = np.empty(0,dtype=np.int32)
    indexes = np.empty(0,dtype=np.int32)
    
    for p in ptypes:
        if not has_ptype(obj, p):
            continue
        
        data = get_property(obj, 'pos', p).to(obj.units['length'])
        pos  = np.append(pos, data.d, axis=0)
        
        data = get_property(obj, 'vel', p).to(obj.units['velocity'])
        vel  = np.append(vel, data.d, axis=0)
        
        data = get_property(obj, 'mass', p).to(obj.units['mass'])
        mass = np.append(mass, data.d, axis=0)

        nparts = len(data)
        
        ptype   = np.append(ptype,   np.full(nparts, ptype_ints[p], dtype=np.int32), axis=0)
        indexes = np.append(indexes, np.arange(0, nparts, dtype=np.int32))

    return dict(pos=pos,vel=vel,mass=mass,ptype=ptype,indexes=indexes)
