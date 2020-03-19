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
        self.dust = False
        self._pdata_loaded = False
        self._determine_ptypes()

    def _member_search_init(self):
        """Method to run all required methods for member_search()"""
        self._determine_ptypes()
        self.load_particle_data()
        self._assign_particle_counts()
        self._load_gas_data()
        self._load_star_data()
        if self.blackholes:
            self._load_bh_data()
        
    def _determine_ptypes(self):
        """Determines what particle/field types to collect."""
        self.ptypes = ['gas','star']
        if 'blackholes' in self.obj._kwargs and self.obj._kwargs['blackholes']:
            if 'PartType5' in self.obj._ds_type.ds.particle_fields_by_type:
                if 'BH_Mdot' in self.obj._ds_type.ds.particle_fields_by_type['PartType5'] or 'StellarFormationTime' in self.obj._ds_type.ds.particle_fields_by_type['PartType5']:
                    from yt.funcs import mylog
                    mylog.warning('Enabling black holes')
                    self.ptypes.append('bh')
                    self.blackholes = True
            else:
                mylog.warning('You have enabled black holes, but no BH particle type is in the simulation snapshot')
        if 'dust' in self.obj._kwargs and self.obj._kwargs['dust']:
            from yt.funcs import mylog
            mylog.warning('Enabling active dust particles')
            self.ptypes.append('dust')
            self.dust = True
        self.ptypes.append('dm')

        #if self.obj._ds_type.grid:
        #    self.ptypes.remove('gas')
        #print self.ptypes
        
    def load_particle_data(self):
        """Loads positions, velocities, masses, particle types, and indexes.
        Assigns a global glist, slist, dlist, dmlist, and bhlist used
        throughout the group analysis.  Finally assigns
        ngas/nstar/ndm/nbh values."""
        if self._pdata_loaded:
            return
        
        pdata      = get_particles_for_FOF(self.obj, self.ptypes)
        self.pos   = pdata['pos']
        self.vel   = pdata['vel']
        self.pot   = pdata['pot']
        self.mass  = pdata['mass']
        self.ptype = pdata['ptype']
        self.index = pdata['indexes']
        if ('fof_from_snap' in self.obj._kwargs and self.obj._kwargs['fof_from_snap']==1):
            self.haloid = pdata['haloid']
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
        self.dlist = np.where(self.ptype == ptype_ints['dust'])[0]

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
        self.obj.simulation.ndust   = len(self.dlist)


    def _load_gas_data(self):
        """If gas is present loads gas SFR/Metallicity/Temperatures."""
        if self.obj.simulation.ngas == 0:
            return
        
        sfr_unit = '%s/%s' % (self.obj.units['mass'], self.obj.units['time'])
        dustmass_unit = '%s' % (self.obj.units['mass'])

        sfr = self.obj.yt_dataset.arr(np.zeros(self.obj.simulation.ngas), sfr_unit)
        gZ  = self.obj.yt_dataset.arr(np.zeros(self.obj.simulation.ngas), '')        
        gT  = self.obj.yt_dataset.arr(np.zeros(self.obj.simulation.ngas), self.obj.units['temperature'])
        dustmass = self.obj.yt_dataset.arr(np.zeros(self.obj.simulation.ngas),'')
        #dustmass = self.obj.yt_dataset.arr(np.zeros(self.obj.simulation.ngas), '')#dustmass_unit)
            
        if has_property(self.obj, 'gas', 'sfr'):
            sfr = get_property(self.obj, 'sfr', 'gas').to(sfr_unit)

        if has_property(self.obj, 'gas', 'metallicity'):            
            gZ  = get_property(self.obj, 'metallicity', 'gas')

        if has_property(self.obj, 'gas', 'temperature'):
            gT  = get_property(self.obj, 'temperature', 'gas').to(self.obj.units['temperature'])

        
        if has_property(self.obj, 'gas', 'dustmass'):
            dustmass = get_property(self.obj,'dustmass','gas')
            #dustmass = get_property(self.obj,'dustmass','gas'))#.to(dustmass_unit)


        self.gsfr = sfr
        self.gZ   = gZ
        self.gT   = gT
        self.dustmass = self.obj.yt_dataset.arr(dustmass,'code_mass').in_units('Msun')

    def _load_star_data(self):
        """If star is present load Metallicity if present"""
        if self.obj.simulation.nstar == 0:
            return

        if has_property(self.obj, 'star', 'metallicity'):
            self.sZ  = get_property(self.obj, 'metallicity', 'star')



    def _load_bh_data(self):
        """If blackholes are present, loads BH_Mdot"""
        from yt.funcs import mylog
        if has_property(self.obj, 'bh', 'bhmass'):
            self.bhmass     = self.obj.yt_dataset.arr(get_property(self.obj, 'bhmass', 'bh').d*1e10,
                                                      'Msun/h').to(self.obj.units['mass'])#I don't know how to convert this automatically
            self.use_bhmass = True
            mylog.info('BH_Mass available, units=1e10 Msun/h')
            mylog.info('Using BH_Mass instead of BH particle masses')
        else:
            mylog.info('Using BH particle mass')

        if has_property(self.obj, 'bh', 'bhmdot'):
            #units mutlitplied by ((All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR))
            bhmdot_unit = '10.22465727143273*Msun/h/yr'
            #bhmdot_unit = '15.036260693283424*Msun/yr'
            #bhmdot_unit = '%s/%s' %(self.obj.units['mass'], self.obj.units['time'])

            bhmdot      = get_property(self.obj, 'bhmdot', 'bh').d #of course  it is dimentionless
            bhmdot      = self.obj.yt_dataset.arr(bhmdot, bhmdot_unit).to('%s/%s' %(self.obj.units['mass'], self.obj.units['time']))
            self.bhmdot = bhmdot
            mylog.info('BH_Mdot available, units=%s'%bhmdot_unit)
        else: mylog.warning('BH_Mdot not available')



