import numpy as np

# Field name aliases
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
    'age':'StellarFormationTime',
    'dustmass':'Dust_Masses'
}

# Field name aliases for grid cells
grid_gas_aliases = {
    'mass':'cell_mass',
    'rho':'density',
    'temp':'temperature',
    'metallicity':'metallicity',
    'pos':'x',
    'vel':'velocity_x',
}

# Integer value for different particles/fields
ptype_ints = dict(
    gas=0,
    star=4,
    dm=1,
    bh=5
)

# Master dict which dictates supported dataset types. Within each dict
# the keys 'gas','star','dm','bh' should point to the corresponding
# yt field name.
ptype_aliases = dict(
    GadgetDataset     = {'gas':'Gas','star':'Stars','dm':'Halo'},
    GadgetHDF5Dataset = {'gas':'PartType0','star':'PartType4','dm':'PartType1','bh':'PartType5'},
    EagleDataset      = {'gas':'PartType0','star':'PartType4','dm':'PartType1','bh':'PartType5'},
    OWLSDataset       = {'gas':'PartType0','star':'PartType4','dm':'PartType1','bh':'PartType5'},
    GizmoDataset      = {'gas':'PartType0','star':'PartType4','dm':'PartType1','bh':'PartType5'},
    TipsyDataset      = {'gas':'Gas','star':'Stars','dm':'DarkMatter'},
    ARTDataset        = {'gas':'gas','star':'stars','dm':'darkmatter'},
    EnzoDataset       = {'gas':'gas','star':'io','dm':'io'},
    RAMSESDataset     = {'gas':'gas','star':'io','dm':'io'},    
)

# Specify which of the above datasets are grid based.
grid_datasets = [
    'EnzoDataset',
    'ARTDataset',
    'RAMSESDataset',
]

class DatasetType(object):
    """Class to help check for, or load data from different dataset 
    types.

    Parameters
    ----------
    ds : yt dataset
        yt dataset loaded via yt.load().

    """
    def __init__(self, ds):
        self.ds      = ds
        self.ds_type = ds.__class__.__name__
        self.dd      = ds.all_data()

        if self.ds_type not in ptype_aliases.keys():
            raise NotImplementedError('%s not yet supported' % self.ds_type)

        self.ptype_aliases = ptype_aliases[self.ds_type]

        self.indexes = 'all'
        
        if self.ds_type in grid_datasets:
            self.grid = True
        else:
            self.grid = False

    def has_ptype(self, requested_ptype):
        """Returns True/False if requested ptype is present.

        Parameters
        ----------
        requested_ptype : str
            Typically 'gas','dm','star','bh'

        Returns
        -------
        bolean

        """
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
        """Gets the correct ptype name for this dataset.

        Parameters
        ----------
        requested_ptype : str
            Typically 'gas','dm','star','bh'

        Returns
        -------
        str
            The proper ptype name for a given dataset.

        """        
        if not self.has_ptype(requested_ptype):
            raise NotImplementedError('Could not find %s ptype!' % requested_ptype)
        return self.ptype_aliases[requested_ptype.lower()]

    def get_property_name(self, requested_ptype, requested_prop):
        """Gets the correct property/field name for this dataset.

        Parameters
        ----------
        requested_ptype : str
            Typically 'gas','dm','star','bh'
        requested_prop : str
            Requested property/field.

        Returns
        -------
        str
            The proper property name for a given dataset.

        """        
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
        """Returns True/False if requested property/field is present.

        Parameters
        ----------
        requested_ptype : str
            Typically 'gas','dm','star','bh'
        requested_prop : str
            Requested property/field.

        Returns
        -------
        boolean
            True if property/field is present, False otherwise.

        """
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
        """Returns the requested property if present.

        Parameters
        ----------
        requested_ptype : str
            Typically 'gas','dm','star','bh'
        requested_prop : str
            Requested property/field.

        Returns
        -------
        np.ndarray
            The requested property values.

        """
        if not self.has_ptype(requested_ptype):
            raise NotImplementedError('ptype %s not found!' % requested_ptype)
        if not self.has_property(requested_ptype, requested_prop):
            raise NotImplementedError('prop %s not found for %s!' % (requested_prop, requested_ptype))

        ptype = self.get_ptype_name(requested_ptype)
        prop  = self.get_property_name(requested_ptype, requested_prop)

        # Correct for special cases of grid code indexes
        if self.ds_type == 'EnzoDataset' and requested_ptype != 'gas':
            self._set_indexes_for_enzo(ptype, requested_ptype)
        if self.ds_type == 'RAMSESDataset' and requested_ptype != 'gas':
            self._set_indexes_for_ramses(ptype, requested_ptype)

        if (self.grid and requested_ptype == 'gas' and
            (requested_prop == 'pos' or requested_prop == 'vel')):
            data = self._get_gas_grid_posvel(requested_prop)
        else:
            data = self.dd[ptype, prop]
            
        if not isinstance(self.indexes, str):
            data = data[self.indexes]
            self.indexes = 'all'
        return data


    def _get_gas_grid_posvel(self,request):
        """Return a typical Nx3 array for gas grid positions."""
        if request == 'pos':
            rx,ry,rz = 'x','y','z'
        elif request == 'vel':
            rx,ry,rz = 'velocity_x','velocity_y','velocity_z'
        xval = self.dd['gas',rx]
        yval = self.dd['gas',ry]
        zval = self.dd['gas',rz]
        return self.ds.arr(np.column_stack((xval.d,yval.d,zval.d)), xval.units)
            
    def _set_indexes_for_enzo(self, proper_ptype, requested_ptype):
        """Extract the correct particle type (star/dm) for Enzo."""
        ptype_vals = dict(gas=0, dm=1, star=2)
        if ptype_vals[requested_ptype] > 0:
            self.indexes = np.where(self.dd[proper_ptype, 'particle_type'] == ptype_vals[requested_ptype])[0]
    def _set_indexes_for_ramses(self, proper_ptype, requested_ptype):
        """Extract the correct particle type (star/dm) for Ramses."""
        if requested_ptype == 'gas': return
        if self.has_property(requested_ptype, 'particle_age'):            
            if requested_ptype == 'dm':
                self.indexes = np.where(self.dd[proper_ptype, 'particle_age'] == 0)[0]
            elif requested_ptype == 'star':
                self.indexes = np.where(self.dd[proper_ptype, 'particle_age'] != 0)[0]
                

def has_ptype(obj, requested_ptype):
    """Helper function to check if ptype/field is present.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Main caesar object.
    requested_ptype : str
        Requested ptype ('gas','star','dm','bh')

    Returns
    -------
    boolean
        True if ptype present, False otherwise.

    """
    return obj._ds_type.has_ptype(requested_ptype)
        
def has_property(obj, requested_ptype, requested_prop):
    """Helper function to check if property is present.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Main caesar object.
    requested_ptype : str
        Requested ptype ('gas','star','dm','bh')
    requested_prop : str
        Requested property name

    Returns
    -------
    boolean
        True if property/field is present, False otherwise.

    """
    return obj._ds_type.has_property(requested_ptype, requested_prop)

def get_property(obj, requested_prop, requested_ptype):
    """Helper function to return a property.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Main caesar object.
    requested_prop : str
        Requested property name
    requested_ptype : str
        Requested ptype ('gas','star','dm','bh')

    Returns
    -------
    np.ndarray
        The requested property for the requested particle/field type.

    """
    ds_type = obj._ds_type    
    return obj._ds_type.get_property(requested_ptype, requested_prop)


def get_high_density_gas_indexes(obj):
    """Returns the indexes of gas with densities above 0.13 protons/cm^3.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Main caesar object.

    Returns
    -------
    np.ndarray
        Index array of high density gas.

    """
    nH_thresh = 0.13

    rho  = get_property(obj, 'rho', 'gas').in_cgs()
    rho /= 1.67262178e-24      # to atoms/cm^3
    rho *= obj.simulation.XH   # to hydrogen atoms / cm^3

    indexes = np.where(rho >= nH_thresh)[0]
    return indexes

def get_particles_for_FOF(obj, ptypes, find_type=None):
    """This function concats all of the valid particle/field types
    into pos/vel/mass/ptype/index arrays for use throughout the 
    analysis.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Main caesar object.
    ptypes : list
        List containing which ptypes to concat.
    find_type : str, optional
        Depreciated.

    Returns
    -------
    dict
        Dictionary containing the keys 'pos','vel','mass','ptype','indexes'.

    """    
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
