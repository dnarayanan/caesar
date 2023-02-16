import numpy as np
from caesar.utils import memlog

MY_DTYPE = np.float32
ISM_NH_THRESHOLD = 0.13  # in protons per cm^3

# Field name aliases
particle_data_aliases = {
    'pos':'particle_position',
    'vel':'particle_velocity',
    'pot':'Potential',
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
    'met_tng':'GFM_Metallicity',  # for Illustris/TNG
    'aform':'StellarFormationTime',
    'aform_tng':'GFM_StellarFormationTime',  # for Illustris/TNG
    'bhmdot':'BH_Mdot',
    'bhmass':'BH_Mass',
    'haloid':'HaloID',
    'dustmass':'Dust_Masses'
}

# Field name aliases for SwiftDataset
particle_data_aliases_swift = {
    'pos':'Coordinates',
    'vel':'Velocities',
    'pot':'Potentials',
    'rho':'Densities',
    'hsml':'SmoothingLengths',
    'sfr':'StarFormationRates',
    'mass':'Masses',
    'temp':'Temperatures',
    'temperature':'Temperatures',
    'ne':'ElectronNumberDensities',
    'nh':'AtomicHydrogenMasses',
    'pid':'ParticleIDs',
    'fh2':'MolecularHydrogenMasses',
    'metallicity':'MetalMassFractions',  
    'aform':'BirthScaleFactors',  
    'bhmdot':'AccretionRates',
    'bhmass':'SubgridMasses',
    'haloid':'FOFGroupIDs',  
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
    dust=3,
    dm3=3,
    star=4,
    dm=1,
    dm2=2,
    bh=5
)

# Master dict which dictates supported dataset types. Within each dict
# the keys 'gas','star','dm','dm2','bh','dust' should point to the corresponding
# yt field name.
ptype_aliases = dict(
    GadgetDataset     = {'gas':'Gas','star':'Stars','dm':'Halo','dm2':'Bulge','dm3':'Disk','bh':'Bndry'},
    GadgetHDF5Dataset = {'gas':'PartType0','star':'PartType4','dm':'PartType1','bh':'PartType5','dm2':'PartType2','dm3':'PartType3'},
    EagleDataset      = {'gas':'PartType0','star':'PartType4','dm':'PartType1','bh':'PartType5','dm2':'PartType2','dm3':'PartType3'},
    OWLSDataset       = {'gas':'PartType0','star':'PartType4','dm':'PartType1','bh':'PartType5','dm2':'PartType2','dm3':'PartType3'},
    GizmoDataset      = {'gas':'PartType0','star':'PartType4','dm':'PartType1','dm2':'PartType2', 'bh':'PartType5','dust':'PartType3','dm3':'PartType3'},
    ArepoHDF5Dataset  = {'gas':'PartType0','star':'PartType4','dm':'PartType1','dm2':'PartType2', 'bh':'PartType5','tracer':'PartType3'},
    SwiftDataset  = {'gas':'PartType0','star':'PartType4','dm':'PartType1','dm2':'PartType2', 'bh':'PartType5','tracer':'PartType3'},
	# comment by Qi: maybe add a [Simba's offspring]-Dataset
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
            if self.ds_type == 'SwiftDataset':
                if prop in particle_data_aliases_swift.keys():
                    return particle_data_aliases_swift[prop]
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
        try:
            ptype = self.get_ptype_name(requested_ptype)
        except NotImplementedError:
            return False

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
            raise NotImplementedError('property %s not found for %s!' % (requested_prop, requested_ptype))

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
            if self.ds_type == 'GizmoDataset' or self.ds_type == 'GadgetDataset' or self.ds_type == None:  # Note yt doesn't work properly for the particle IDs for Gadget snapshot, additional twist is needed in lower function and pygadgetreader
                data = self._get_simba_property(requested_ptype,requested_prop)
            else:
                data = self.dd[ptype, prop].astype(MY_DTYPE)

        #if not isinstance(self.indexes, str):
        #    data = data[self.indexes]
        #    self.indexes = 'all'
        return data

    def _get_simba_property(self,ptype,prop):
        try:
            import pygadgetreader as pygr
        except:
            return self.dd[ptype, prop].astype(MY_DTYPE)
        snapfile = ('%s/%s'%(self.ds.directory,self.ds.basename))
        # set up units coming out of pygr
        prop_unit = {'mass':'Msun', 'pos':'kpccm', 'vel':'km/s', 'pot':'Msun * kpccm**2 / s**2', 'rho':'g / cm**3', 'sfr':'Msun / yr', 'u':'K', 'Dust_Masses':'Msun', 'bhmass':'Msun', 'bhmdot':'Msun / yr', 'hsml':'kpccm'}

        # damn you little h!
        if prop == 'mass' or prop == 'pos':
            hfact = 1./self.ds.hubble_constant
        elif prop == 'rho':
            hfact = self.ds.hubble_constant**2
        else:
            hfact = 1

        # deal with differences in pygr vs. yt/caesar naming
        if ptype == 'bh': ptype = 'bndry'
        if prop == 'temperature': prop = 'u'
        if prop == 'haloid' or prop == 'dustmass' or prop == 'aform' or prop == 'bhmass' or prop == 'bhmdot': prop = self.get_property_name(ptype, prop)
        if ptype == 'dm2': ptype = 'disk'
        if ptype == 'dm3': ptype = 'bulge'

        # read in the data
        if (self.ds_type == 'GadgetDataset'):  #need to retweek the names for G2.
            if prop == 'metallicity':
                prop =  'Z'
            if prop == 'aform' or prop == 'StellarFormationTime':
                prop = 'age'
        data = pygr.readsnap(snapfile, prop, ptype, units=1, suppress=1) * hfact

        # set to doubles
        if prop == 'HaloID' or prop == 'haloid':
            data = data.astype(np.uint32)
        elif prop == 'particle_index' or prop == 'pid':  # this fixes a bug in our Gizmo, that HaloID is output as a float!
            data = data.astype(np.int64)
        else:
            data = data.astype(np.float32)

        if prop in prop_unit.keys():
            data = self.ds.arr(data, prop_unit[prop])
        else:
            data = self.ds.arr(data, '')
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


def get_high_density_gas_indexes(obj, nH_thresh=ISM_NH_THRESHOLD, return_flag=0):
    """Returns the indexes of gas with densities specified nH threshold in protons/cm^3.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Main caesar object.

    Returns
    -------
    np.ndarray
        Index array of high density gas.

    """
    rho  = get_property(obj, 'rho', 'gas').in_cgs()
    rho /= 1.67262178e-24      # to atoms/cm^3
    rho *= obj.simulation.XH   # to hydrogen atoms / cm^3

    if return_flag == 1:
        return np.where(rho >= nH_thresh, 1, 0)
    else:
        return np.where(rho >= nH_thresh)[0]

def get_particles_for_FOF(obj, ptypes, select='all', my_dtype=MY_DTYPE):
    """This function concats all of the valid particle/field types
    into pos/vel/mass/ptype/index arrays for use throughout the 
    analysis.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Main caesar object.
    ptypes : list
        List containing which ptypes to concat.
    select : a list of length len(ptypes) containing numpy arrays, where
        only particles with array values>=0 will be selected

    Returns
    -------
    dict
        Dictionary containing the keys 'pos','vel','mass','ptype','indexes'.

    """    
    # check if potential exists in snapshot for all particle types
    obj.load_pot = True
    for ip,p in enumerate(ptypes):
        if not has_ptype(obj, p):
            continue
        if not has_property(obj, p, 'pot'):
            obj.load_pot = False

    pos  = np.empty((0,3),dtype=MY_DTYPE)
    vel  = np.empty((0,3),dtype=MY_DTYPE)
    mass = np.empty(0,dtype=MY_DTYPE)
    pot = np.empty(0,dtype=MY_DTYPE)
    if obj.load_haloid:
        haloid  = np.empty(0, dtype=np.int64)

    ptype   = np.empty(0,dtype=np.int32)
    indexes = np.empty(0,dtype=np.int64)

    for ip,p in enumerate(ptypes):
        if not has_ptype(obj, p):
            continue
     
        if p == 'bh':
            if has_property(obj, 'bh', 'bhmass'):
                count = len(get_property(obj, 'bhmass', p))
            else:
                count = len(get_property(obj, 'mass', p))
        else:
            count = len(get_property(obj, 'mass', p))
        if isinstance(select,str) and select == 'all': 
            flag = [True]*count
        else:
            flag = (select[ip]>=0)

        data = get_property(obj, 'pos', p).to(obj.units['length'])[flag]
        pos  = np.append(pos, data.d, axis=0)
        
        data = get_property(obj, 'vel', p).to(obj.units['velocity'])[flag]
        vel  = np.append(vel, data.d, axis=0)
        
        if p == 'bh':
            if has_property(obj, 'bh', 'bhmass'):
                data = get_property(obj, 'bhmass', 'bh').to(obj.units['mass'])[flag]
                # obj.yt_dataset.arr(get_property(obj, 'bhmass', 'bh').d[flag]*1e10, 'Msun/h').to(obj.units['mass'])
            else:
                data = get_property(obj, 'mass', 'bh').to(obj.units['mass'])[flag]
                # obj.yt_dataset.arr(get_property(obj, 'mass', 'bh').d[flag]*1e10, 'Msun/h').to(obj.units['mass'])
        else:
            data = get_property(obj, 'mass', p).to(obj.units['mass'])[flag]
        mass = np.append(mass, data.d, axis=0)

        if obj.load_pot:
            data = get_property(obj, 'pot', p)[flag]
            pot = np.append(pot, data.d, axis=0)
        else:
            pot = np.append(pot, np.zeros(count,dtype=MY_DTYPE), axis=0)

        if obj.load_haloid:
            data = get_property(obj, 'haloid', p)[flag]
            haloid = np.append(haloid, data.d.astype(np.int64), axis=0)

        nparts = len(data)

        ptype   = np.append(ptype,   np.full(nparts, ptype_ints[p], dtype=np.int32), axis=0)
        indexes = np.append(indexes, np.arange(0, count, dtype=np.int64)[flag])

    if obj.load_haloid:
        return dict(pos=pos,vel=vel,pot=pot,mass=mass,haloid=haloid,ptype=ptype,indexes=indexes)
    else: return dict(pos=pos,vel=vel,pot=pot,mass=mass,ptype=ptype,indexes=indexes)

def get_haloid(obj, ptypes, offset=-1):
    """This function returns a list of HaloID numpy arrays from the snapshot, 
    corresponding to the HaloID for the particle types in ptypes.
    The list of arrays will be of the length of ptypes.

    The Gadget/Gizmo convention is that the snapshot has HaloID=0 for particles not in a halo,
    whereas yt/caesar use -1 for this.  So we add offset=-1 to each haloid, but this can be changed.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Main caesar object.
    ptypes : list
        List containing which ptypes to concat.
    select : str, optional

    Returns
    -------
    list of array of haloids for each particle type in ptypes

    """    
    haloid = []

    obj.npartsnap = 0  # number of particles in snapshot
    for p in ptypes:
        if has_ptype(obj, p): 
            data = (get_property(obj, 'haloid', p).d.astype(np.int64) + offset)
            obj.npartsnap += len(data)
        else: 
            data = []
        haloid.append(data)
    haloid = np.asarray(haloid)

    return haloid

