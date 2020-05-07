import numpy as np
from yt.funcs import mylog

from caesar.property_manager import ptype_ints, get_particles_for_FOF, get_property, has_property
from caesar.utils import memlog
from caesar.property_manager import MY_DTYPE

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

    def _member_search_init(self, select='all'):
        """Collect particle information for member_search()"""
        memlog('Initializing member search, loading particles')
        self._determine_ptypes()
        self.load_particle_data(select=select)
        memlog('Loaded particle data')
        self._assign_particle_counts()
        if select is 'all': self._load_gas_data()
        else: self._load_gas_data(select=select[self.ptypes.index('gas')])
        if select is 'all': self._load_star_data()
        else: self._load_star_data(select=select[self.ptypes.index('star')])
        if self.blackholes:
            if select is 'all': self._load_bh_data()
            else: self._load_bh_data(select=select[self.ptypes.index('bh')])
        memlog('Loaded baryon data')
        
    def _determine_ptypes(self):
        """Determines what particle/field types to collect."""
        self.ptypes = ['gas','star']
        #if 'blackholes' in self.obj._kwargs and self.obj._kwargs['blackholes']:
        self.blackholes = self.dust = self.dm2 = False
        if hasattr(self.obj,'_ds_type'):
            if 'PartType5' in self.obj._ds_type.ds.particle_fields_by_type:
                if 'BH_Mdot' in self.obj._ds_type.ds.particle_fields_by_type['PartType5'] or 'StellarFormationTime' in self.obj._ds_type.ds.particle_fields_by_type['PartType5']:
                    self.ptypes.append('bh')
                    self.blackholes = True
            else:
                memlog('No black holes found')
        if hasattr(self.obj,'_kwargs') and 'dust' in self.obj._kwargs and self.obj._kwargs['dust']:
            mylog.warning('Enabling active dust particles')
            self.ptypes.append('dust')
            self.dust = True
        self.ptypes.append('dm')
        if hasattr(self.obj,'_kwargs') and 'dm2' in self.obj._kwargs and self.obj._kwargs['dm2']:
            self.ptypes.append('dm2')
            self.dm2 = True

        #if self.obj._ds_type.grid:
        #    self.ptypes.remove('gas')
        #print self.ptypes
        
    def load_particle_data(self, select=None):
        """Loads positions, velocities, masses, particle types, and indexes.
        Assigns a global glist, slist, dlist, dmlist, and bhlist used
        throughout the group analysis.  Finally assigns
        ngas/nstar/ndm/nbh values."""
        if self._pdata_loaded:
            return
        
        pdata      = get_particles_for_FOF(self.obj, self.ptypes, select)
        self.pos   = pdata['pos']
        self.vel   = pdata['vel']
        self.pot   = pdata['pot']
        self.mass  = pdata['mass']
        self.ptype = pdata['ptype']
        self.indexes = pdata['indexes']
        if hasattr(self.obj,'_kwargs') and ('haloid' in self.obj._kwargs and 'snap' in self.obj._kwargs['haloid']):
            self.haloid = pdata['haloid']
        pdata      = None

        self._assign_local_lists()
        self._check_for_lowres_dm()
        
        if select is None: self._pdata_loaded = True

    def _assign_local_lists(self):
        """Assigns local lists."""
        self.glist  = np.where(self.ptype == ptype_ints['gas'])[0]
        self.slist  = np.where(self.ptype == ptype_ints['star'])[0]
        self.dmlist = np.where(self.ptype == ptype_ints['dm'])[0]        
        self.dm2list = np.where(self.ptype == ptype_ints['dm2'])[0]
        self.bhlist = np.where(self.ptype == ptype_ints['bh'])[0]
        self.dlist = np.where(self.ptype == ptype_ints['dust'])[0]

    def _reset_dm_indexes(self):
        """Reset the dark matter index list after we detect a zoom."""
        self.indexes[self.dmlist] = np.arange(0,len(self.dmlist), dtype=np.int32)

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
        self.obj.simulation.ndm2  = len(self.dm2list)
        self.obj.simulation.nbh   = len(self.bhlist)
        self.obj.simulation.ndust = len(self.dlist)
        self.obj.simulation.ntot  = self.obj.simulation.ngas+self.obj.simulation.nstar+self.obj.simulation.ndm+self.obj.simulation.ndm2+self.obj.simulation.nbh+self.obj.simulation.ndust


    def _load_gas_data(self,select='all'):
        """If gas is present loads gas SFR, metallicities, temperatures, nH.
           If select is not 'all', return all particles with select>=0 """

        if self.obj.simulation.ngas == 0:
            return
        
        sfr_unit = '%s/%s' % (self.obj.units['mass'], self.obj.units['time'])
        dustmass_unit = '%s' % (self.obj.units['mass'])
        gnh_unit = '1/%s**3' % (self.obj.units['length'])

        sfr = self.obj.yt_dataset.arr(np.zeros(self.obj.simulation.ngas,dtype=MY_DTYPE), sfr_unit)
        gZ  = self.obj.yt_dataset.arr(np.zeros(self.obj.simulation.ngas,dtype=MY_DTYPE), '')        
        gT  = self.obj.yt_dataset.arr(np.zeros(self.obj.simulation.ngas,dtype=MY_DTYPE), self.obj.units['temperature'])
        gnh  = self.obj.yt_dataset.arr(np.zeros(self.obj.simulation.ngas,dtype=MY_DTYPE), gnh_unit)
        dustmass = self.obj.yt_dataset.arr(np.zeros(self.obj.simulation.ngas,dtype=MY_DTYPE),'')
        gfHI  = self.obj.yt_dataset.arr(np.zeros(self.obj.simulation.ngas,dtype=MY_DTYPE), '')        
        gfH2  = self.obj.yt_dataset.arr(np.zeros(self.obj.simulation.ngas,dtype=MY_DTYPE), '')        
        ghsml  = self.obj.yt_dataset.arr(np.zeros(self.obj.simulation.ngas,dtype=MY_DTYPE), self.obj.units['length'])        
        #dustmass = self.obj.yt_dataset.arr(np.zeros(self.obj.simulation.ngas), '')#dustmass_unit)
            
        if select is 'all': 
            flag = [True]*self.obj.simulation.ngas
        else:
            flag = (select>=0)

        if has_property(self.obj, 'gas', 'sfr'):
            sfr = get_property(self.obj, 'sfr', 'gas')[flag].to(sfr_unit)

        if has_property(self.obj, 'gas', 'metallicity'):            
            gZ  = get_property(self.obj, 'metallicity', 'gas')[flag]
        elif has_property(self.obj, 'gas', 'met_tng'):
            gZ  = get_property(self.obj, 'met_tng', 'gas')[flag]  # for Illustris, array of mets
        else:
            mylog.warning('Metallicity not found: setting all gas to solar=0.0134')
            gZ = 0.0134*np.ones(self.obj.simulation.nstar,dtype=MY_DTYPE)

        if has_property(self.obj, 'gas', 'nh'):            
            gfHI  = get_property(self.obj, 'nh', 'gas')[flag]

        if has_property(self.obj, 'gas', 'fh2'):            
            gfH2  = get_property(self.obj, 'fh2', 'gas')[flag]
        else:
            mylog.warning('H2 fractions not found in snapshot -- will compute later')

        if has_property(self.obj, 'gas', 'temperature'):
            gT  = get_property(self.obj, 'temperature', 'gas')[flag].to(self.obj.units['temperature'])

        if has_property(self.obj, 'gas', 'hsml'):
            ghsml  = get_property(self.obj, 'hsml', 'gas')[flag].to(self.obj.units['length'])

        if has_property(self.obj, 'gas', 'rho'):
            from astropy import constants as const
            from yt import YTQuantity
            redshift = self.obj.simulation.redshift
            m_p = YTQuantity.from_astropy(const.m_p)
            gnh  = get_property(self.obj, 'rho', 'gas')[flag].in_cgs() *0.76*(1+redshift)**3/m_p.in_cgs()
 
        if has_property(self.obj, 'gas', 'dustmass'):
            dustmass = get_property(self.obj,'dustmass','gas')[flag]
        else:
            mylog.warning('Dust masses not found in snapshot')


        self.gsfr = sfr
        self.gZ   = gZ
        self.gT   = gT
        self.gnh   = gnh
        self.gfHI   = gfHI
        self.gfH2   = gfH2
        self.hsml   = ghsml
        self.dustmass = self.obj.yt_dataset.arr(dustmass,'code_mass').in_units('Msun')
        self.dustmass.dtype = MY_DTYPE

    def _load_star_data(self, select='all'):
        """If star is present load Metallicity if present"""
        if self.obj.simulation.nstar == 0:
            return

        if select is 'all': 
            flag = [True]*self.obj.simulation.nstar
        else:
            flag = (select>=0)

        if has_property(self.obj, 'star', 'metallicity'):
            self.sZ  = get_property(self.obj, 'metallicity', 'star')[flag]
        elif has_property(self.obj, 'star', 'met_tng'):  # try Illustris/TNG alias
            self.sZ  = get_property(self.obj, 'met_tng', 'star')[flag]  
            #self.sZ  = np.sum(self.sZ.T[2:],axis=0)  # first two are H,He; the rest sum to give metallicity
            #self.sZ[self.sZ<0] = 0.  # some (very small) negative values, set to 0
        else:
            mylog.warning('Metallicity not found: setting all stars to solar=0.0134')
            self.sZ = 0.0134*np.ones(self.obj.simulation.nstar,dtype=MY_DTYPE)

        ds = self.obj.yt_dataset
        if has_property(self.obj, 'star', 'aform'):
            self.age  = get_property(self.obj, 'aform', 'star')[flag]  # a_exp at time of formation
        elif has_property(self.obj, 'star', 'aform_tng'):  # try Illustris/TNG alias
            self.age  = get_property(self.obj, 'aform_tng', 'star')[flag]  
            self.age  = abs(self.age)  # some negative values here too; not sure what to do?
        else:
            self.age = np.zeros(self.obj.simulation.nstar,dtype=MY_DTYPE)
            mylog.warning('Stellar age not found -- photometry will be incorrect!')
        if ds.cosmological_simulation:
            from yt.utilities.cosmology import Cosmology
            co = Cosmology(hubble_constant=ds.hubble_constant, omega_matter=ds.omega_matter, omega_lambda=ds.omega_lambda)
            self.age = (ds.current_time - co.t_from_z(1./self.age-1.)).in_units('Gyr').astype(MY_DTYPE)  # age at time of snapshot 

    def _load_bh_data(self, select='all'):
        """If blackholes are present, loads BH_Mdot"""

        if select is 'all': 
            flag = [True]*self.obj.simulation.nbh
        else:
            flag = (select>=0)

        if has_property(self.obj, 'bh', 'bhmass'):
            self.bhmass     = self.obj.yt_dataset.arr(get_property(self.obj, 'bhmass', 'bh').d[flag]*1e10, 'Msun/h').to(self.obj.units['mass'])  # I don't know how to convert this automatically
            self.use_bhmass = True
        else:
            mylog.warning('No black holes found')
            self.use_bhmass = False

        if has_property(self.obj, 'bh', 'bhmdot') and self.use_bhmass:
            #units mutlitplied by ((All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR))
            bhmdot_unit = '10.22465727143273*Msun/h/yr'
            #bhmdot_unit = '15.036260693283424*Msun/yr'
            #bhmdot_unit = '%s/%s' %(self.obj.units['mass'], self.obj.units['time'])

            bhmdot      = get_property(self.obj, 'bhmdot', 'bh').d[flag] #of course  it is dimentionless
            bhmdot      = self.obj.yt_dataset.arr(bhmdot, bhmdot_unit).to('%s/%s' %(self.obj.units['mass'], self.obj.units['time']))
            self.bhmdot = bhmdot
            #mylog.info('BH_Mdot available, units=%s'%bhmdot_unit)
        else: 
            if self.use_bhmass: 
                mylog.warning('Black holes are there, but BH_Mdot not available!')

    def _photometry_init(self):
        """Collect particle information for photometry"""
        from caesar.property_manager import get_property, ptype_ints

        memlog('Loading gas and star particles for photometry')
        self._determine_ptypes()

        self.pos  = np.empty((0,3),dtype=MY_DTYPE)
        self.vel  = np.empty((0,3),dtype=MY_DTYPE)
        self.mass = np.empty(0,dtype=MY_DTYPE)
        self.ptype   = np.empty(0,dtype=np.int32)
        for ip,p in enumerate(['gas','star']):
            data = get_property(self.obj, 'pos', p).to(self.obj.units['length'])
            self.pos  = np.append(self.pos, data.d, axis=0)
            data = get_property(self.obj, 'vel', p).to(self.obj.units['velocity'])
            self.vel  = np.append(self.vel, data.d, axis=0)
            data = get_property(self.obj, 'mass', p).to(self.obj.units['mass'])
            self.mass = np.append(self.mass, data.d, axis=0)
            self.ptype   = np.append(self.ptype, np.full(len(data), ptype_ints[p], dtype=np.int32), axis=0)
        self._assign_local_lists()
        self._assign_particle_counts()
        memlog('Loaded particle data')

        self._load_gas_data()
        self._load_star_data()
        memlog('Loaded gas and star data')
        


