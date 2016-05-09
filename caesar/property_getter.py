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

        if self.ds_type not in ptype_aliases.keys():
            raise NotImplementedError('%s not yet supported' % ds_type)

        self.ptype_aliases = ptype_aliases[self.ds_type]

        self.indexes = 'all'
        
        if self.ds_type == 'EnzoDataset':
            self.grid = True
        else:
            self.grid = False


    def check_for_field(self, prop, ptype):
        present = False
        fields = self.ds.particle_fields_by_type[ptype]
        if prop in fields:
            present = True
        else:
            fields = self.ds.derived_field_list
            for f in fields:
                if f[0] == ptype and f[1] == prop:
                    present = True
                    break
        return present
        
    def _get_particle_property_name(self, request):
        if request in particle_data_aliases.keys():
            prop = particle_data_aliases[request]
        else:
            prop = request
        return prop

    def _get_grid_property_name(self, request):
        if request in grid_gas_aliases.keys():
            prop = grid_gas_aliases[request]
        else:
            prop = request
        return prop

    def _get_ptype_name(self, request):
        if request in self.ptype_aliases.keys():
            return self.ptype_aliases[request]
        else:
            raise NotImplementedError('could not find %s ptype!' % request)

        
    def _set_indexes_for_enzo(self, proper_ptype, requested_ptype):
        ptype_vals = dict(gas=0, dm=1, star=2)
        if ptype_vals[requested_ptype] > 0:
            self.indexes = np.where(self.dd[proper_ptype, 'particle_type'] == ptype_vals[requested_ptype])[0]
            
    def get_property(self, requested_prop, requested_ptype, indexes='all'):
        self.indexes = indexes
        if not hasattr(self, 'dd'):
            self.dd = self.ds.all_data()

        ptype = self._get_ptype_name(requested_ptype)

        ## special enzo case
        if self.ds_type == 'EnzoDataset':
            self._set_indexes_for_enzo(ptype, requested_ptype)
        ##
        
        if requested_ptype == 'gas' and self.grid:
            prop = self._get_grid_property_name(requested_prop)
        else:
            prop = self._get_particle_property_name(requested_prop)

        if not self.check_for_field(prop, ptype):
            raise IOError('could not find %s for %s!' % (prop, ptype))

        data = self.dd[ptype, prop]

        if not isinstance(self.indexes, str):
            data = data[self.indexes]
        return data

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

def check_for_ptype(obj, requested_ptype):
    requested_ptype = requested_ptype.lower()
    ptype = obj._ds_type._get_ptype_name(requested_ptype)
    if ptype in obj._ds_type.ds.particle_fields_by_type:
        return True
    else:
        return False    

def get_property(obj, requested_prop, requested_ptype, indexes='all'):
    requested_prop  = requested_prop.lower()
    requested_ptype = requested_ptype.lower()

    ds_type = obj._ds_type
    
    return obj._ds_type.get_property(requested_prop, requested_ptype, indexes=indexes)


def get_high_density_gas_indexes(obj):
    nH_thresh = 0.13

    rho  = get_property(obj, 'rho', 'gas').in_cgs()
    rho /= 1.67262178e-24  # to atoms/cm^3
    rho *= 0.76            # to hydrogen atoms / cm^3

    indexes = np.where(rho >= nH_thresh)[0]
    return indexes

def get_particles_for_FOF(obj, ptypes, find_type):

    pos  = np.empty((0,3))
    vel  = np.empty((0,3))
    mass = np.empty(0)

    ptype   = np.empty(0,dtype=np.int32)
    indexes = np.empty(0,dtype=np.int32)
    
    for p in ptypes:
        if not check_for_ptype(obj, p):
            continue
        
        ind = 'all'
        if p == 'gas' and find_type == 'galaxy':
            ind = get_high_density_gas_indexes(obj)
            
        data = get_property(obj, 'pos', p, indexes=ind).to(obj.units['length'])
        pos  = np.append(pos, data.d, axis=0)
        
        data = get_property(obj, 'vel', p, indexes=ind).to(obj.units['velocity'])
        vel  = np.append(vel, data.d, axis=0)
        
        data = get_property(obj, 'mass', p, indexes=ind).to(obj.units['mass'])
        mass = np.append(mass, data.d, axis=0)

        nparts = len(data)
        
        ptype   = np.append(ptype,   np.full(nparts, ptype_ints[p], dtype=np.int32), axis=0)

        if isinstance(ind, str):
            i = np.arange(0, nparts, dtype=np.int32)
        else:
            i = ind.astype(np.int32)        
        indexes = np.append(indexes, i, axis=0) 

    return dict(pos=pos,vel=vel,mass=mass,ptype=ptype,indexes=indexes)
