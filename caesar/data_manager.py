import numpy as np

from caesar.property_manager import ptype_ints, get_particles_for_FOF, get_property, has_property

class DataManager(object):
    """Class to handle the initial IO and data storage for the duration of
    a CAESAR run.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Main CAESAR object.

    """    
    def __init__(self, obj):
        self.obj = obj
        self.blackholes = False
        self._pdata_loaded = False
        self._determine_ptypes()

    def _member_search_init(self):
        """Method to run all required methods for member_search()"""
        self._determine_ptypes()
        self.load_particle_data()
        self._assign_particle_counts()
        self._load_gas_data()
        
    def _determine_ptypes(self):
        """Determines what particle/field types to collect."""
        self.ptypes = ['gas','star']
        if 'blackholes' in self.obj._kwargs and self.obj._kwargs['blackholes']:
            from yt.funcs import mylog
            mylog.warning('Enabling black holes')
            self.ptypes.append('bh')
            self.blackholes = True
        self.ptypes.append('dm')

        #if self.obj._ds_type.grid:
        #    self.ptypes.remove('gas')
        #print self.ptypes
        
    def load_particle_data(self):
        """Loads positions, velocities, masses, particle types, and indexes.
        Assigns a global glist, slist, dmlist, and bhlist used
        throughout the group analysis.  Finally assigns
        ngas/nstar/ndm/nbh values."""
        if self._pdata_loaded:
            return
        
        pdata      = get_particles_for_FOF(self.obj, self.ptypes)
        self.pos   = pdata['pos']
        self.vel   = pdata['vel']
        self.mass  = pdata['mass']
        self.ptype = pdata['ptype']
        self.index = pdata['indexes']
        pdata      = None

        self._assign_local_lists()
        self._check_for_lowres_dm()
        
        self._pdata_loaded = True

    def _assign_local_lists(self):
        """Assigns local lists."""
        self.glist  = np.where(self.ptype == ptype_ints['gas'])[0]
        self.slist  = np.where(self.ptype == ptype_ints['star'])[0]
        self.dmlist = np.where(self.ptype == ptype_ints['dm'])[0]        
        self.bhlist = np.where(self.ptype == ptype_ints['bh'])[0]

    def _reset_dm_indexes(self):
        """Reset the dark matter index list after we detect a zoom."""
        self.index[self.dmlist] = np.arange(0,len(self.dmlist), dtype=np.int32)

    def _check_for_lowres_dm(self):
        """Check and account for low-resolution dark matter in non 
        Gadget/Gizmo simulations."""
        gadget_list = ['GadgetDataset','GadgetHDF5Dataset',
                       'EagleDataset','OWLSDataset','GizmoDataset']
        if self.obj._ds_type.ds_type in gadget_list:
            return  # lowres particles for gadget are a diff type
        
        dmmass = self.mass[self.dmlist]
        unique = np.unique(dmmass)
        if len(unique) > 1:
            from yt.funcs import mylog
            mylog.info('Found %d DM species, assuming a zoom' % len(unique))
            minmass = np.min(unique)
            lowres  = np.where(dmmass > minmass)[0]
            self.ptype[self.dmlist[lowres]] = 2  ## arbitrary
            self._assign_local_lists()
            self._reset_dm_indexes()
            
    def _assign_particle_counts(self):
        """Assign particle counts."""
        self.obj.simulation.ngas  = len(self.glist)
        self.obj.simulation.nstar = len(self.slist)
        self.obj.simulation.ndm   = len(self.dmlist)
        self.obj.simulation.nbh   = len(self.bhlist)


    def _load_gas_data(self):
        """If gas is present loads gas SFR/Metallicity/Temperatures."""
        if self.obj.simulation.ngas == 0:
            return
        
        sfr_unit = '%s/%s' % (self.obj.units['mass'], self.obj.units['time'])

        sfr = self.obj.yt_dataset.arr(np.zeros(self.obj.simulation.ngas), sfr_unit)
        gZ  = self.obj.yt_dataset.arr(np.zeros(self.obj.simulation.ngas), '')        
        gT  = self.obj.yt_dataset.arr(np.zeros(self.obj.simulation.ngas), self.obj.units['temperature'])
            
        if has_property(self.obj, 'gas', 'sfr'):
            sfr = get_property(self.obj, 'sfr', 'gas').to(sfr_unit)

        if has_property(self.obj, 'gas', 'metallicity'):            
            gZ  = get_property(self.obj, 'metallicity', 'gas')

        if has_property(self.obj, 'gas', 'temperature'):
            gT  = get_property(self.obj, 'temperature', 'gas').to(self.obj.units['temperature'])

        self.gsfr = sfr
        self.gZ   = gZ
        self.gT   = gT
