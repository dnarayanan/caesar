import six
import numpy as np

from caesar.property_manager import ptype_ints
from caesar.group_funcs import get_periodic_r,get_virial_mr

MINIMUM_STARS_PER_GALAXY = 24  # set a bit below 32 so we capture all galaxies above a given Mstar, rather than a given Nstar.
MINIMUM_DM_PER_HALO      = 64
MINIMUM_GAS_PER_CLOUD = 16

group_types = dict(
    halo='halos',
    galaxy='galaxies',
    cloud='clouds'
)

info_blacklist = [
    '_glist','glist_end','glist_start',
    '_slist','slist_end','slist_start',
    '_dmlist','dmlist_end','dmlist_start',
    '_dlist','dlist_end','dlist_start',
    'obj', 'halo', 'galaxies','clouds', 'satellites',
    'galaxy_index_list_end', 'galaxy_index_list_start','cloud_index_list_end','cloud_index_list_start']

category_mapper = dict(
    mass = 'masses',
    radius = 'radii',
    sigma = 'velocity_dispersions',
    metallicity = 'metallicities',
    temperature = 'temperatures',
)

class GroupProperty(object):
    """Class to return default values for the quantities held in 
    the category_mapper dictionaries."""
    def __init__(self, source_dict, name):
        self.name = name
        self.source_dict = source_dict
    def __get__(self, instance, owner):
        key = instance.obj._default_returns[instance.obj_type][self.name]
        return getattr(instance, self.source_dict)[key]
    def __set__(self, instance, value):
        pass

class GroupList(object):
    """Class to hold particle/field index lists."""
    def __init__(self, name):
        self.name = name
    def __get__(self, instance, owner):
        if not hasattr(instance, '_%s' % self.name) or \
           isinstance(getattr(instance, '_%s' % self.name), int):
            from caesar.loader import restore_single_list
            restore_single_list(instance.obj, instance, self.name)
        return getattr(instance, '_%s' % self.name)
    def __set__(self, instance, value):
        setattr(instance, '_%s' % self.name, value)


class Group(object):
    """Parent class for halo and galaxy and halo objects."""
    glist = GroupList('glist')
    slist = GroupList('slist')    

    mass        = GroupProperty(category_mapper['mass'],   'mass')
    radius      = GroupProperty(category_mapper['radius'], 'radius')
    sigma       = GroupProperty(category_mapper['sigma'],  'sigma')
    temperature = GroupProperty(category_mapper['temperature'], 'temperature')
    metallicity = GroupProperty(category_mapper['metallicity'], 'metallicity')
    
    def __init__(self,obj):
        self.obj = obj

        self.masses = {}
        self.radii = {} 
        self.temperatures = {}
        self.spin_param = 0.0
    def _append_global_index(self, i):
        if not hasattr(self, 'global_indexes'):
            self.global_indexes = []
        self.global_indexes.append(i)

    @property
    def _valid(self):
        """Check against the minimum number of particles to see if
        this object is 'valid'."""
        if self.obj_type == 'halo' and self.ndm < MINIMUM_DM_PER_HALO:
            return False
        elif self.obj_type == 'galaxy' and self.nstar < MINIMUM_STARS_PER_GALAXY:
            return False
        elif self.obj_type == 'cloud' and self.ngas < MINIMUM_GAS_PER_CLOUD:
            return False
        else:
            return True

    def _delete_attribute(self,a):
        """Helper method to delete an attribute if present."""
        if hasattr(self,a):
            delattr(self,a)

    def _delete_key(self,d,k):
        """Helper method to delete a dict key."""
        if k in d:
            del d[k]
            
    def _remove_dm_references(self):
        """Galaxies/clouds do not have DM, so remove references."""
        if self.obj_type != 'galaxy' or not self._valid:
            return
        self._delete_attribute('ndm')        
        self._delete_key(self.radii,'dm_half_mass')
        self._delete_key(self.radii,'dm')
        self._delete_key(self.masses,'dm')
        self._delete_key(self.velocity_dispersions,'dm')
            
    def _cleanup(self):
        """ cleanup function to delete attributes no longer needed """
        self._delete_attribute('global_indexes')
        self._delete_attribute('__glist')
        self._remove_dm_references()
        """ cleanup function to delete attributes no longer needed """
        self._delete_attribute('periodic_r')
        self._delete_attribute('__slist')
        self._delete_attribute('__dmlist')


    def _process_group(self):
        """Process each group after creation.  This entails 
        calculating the total mass, iteratively unbinding (if enabled),
        then calculating more masses, radial quants, virial quants, 
        velocity dispersions, angular quants, and final gas quants.
        """
        self._assign_local_data()

        if self._valid:
            self._calculate_total_mass()
            self._calculate_center_of_mass_quantities()
            self._unbind()  # iterative procedure
            
            if self._valid:
                self._calculate_masses()
                self._calculate_radial_quantities()
                self._calculate_virial_quantities()
                self._calculate_velocity_dispersions()
                self._calculate_angular_quantities()
                self._calculate_gas_quantities()
                self._calculate_star_quantities()
                if self.obj.data_manager.blackholes:
                    self._calculate_bh_quantities()

        self._cleanup()

        
    def _assign_local_data(self):
        """Assign glist/slist/dmlist/bhlist/dlist for this group.  
		Also sets the ngas/nstar/ndm/nbh/ndust attributes."""
        ptypes  = self.obj.data_manager.ptype[self.global_indexes]
        indexes = self.obj.data_manager.index[self.global_indexes]

        # lists for the concatinated global list
        self.__glist = np.where(ptypes == ptype_ints['gas'])[0]
        self.__slist = np.where(ptypes == ptype_ints['star'])[0]
        self.__dmlist = np.where(ptypes == ptype_ints['dm'])[0]

        # individual global lists
        self.glist  = indexes[np.where(ptypes == ptype_ints['gas'])[0]]
        self.slist  = indexes[np.where(ptypes == ptype_ints['star'])[0]]
        self.dmlist = indexes[np.where(ptypes == ptype_ints['dm'])[0]]
        self.bhlist = indexes[np.where(ptypes == ptype_ints['bh'])[0]]
        self.dlist  = indexes[np.where(ptypes == ptype_ints['dust'])[0]]
        
        self.ngas  = len(self.glist)
        self.nstar = len(self.slist)
        self.ndm   = len(self.dmlist)
        self.ndust = len(self.dlist)

        if self.obj.data_manager.blackholes:
            self.bhlist = indexes[np.where(ptypes == ptype_ints['bh'])[0]]
            self.nbh    = len(self.bhlist)

    def _calculate_total_mass(self):
        """Calculate the total mass of the object."""
        self.masses['total'] = self.obj.yt_dataset.quan(np.sum(self.obj.data_manager.mass[self.global_indexes]), self.obj.units['mass'])
        
    def _calculate_masses(self):
        """Calculate various total masses."""
        mass_dm     = np.sum(self.obj.data_manager.mass[self.obj.data_manager.dmlist[self.dmlist]])
        mass_gas    = np.sum(self.obj.data_manager.mass[self.obj.data_manager.glist[self.glist]])
        mass_star   = np.sum(self.obj.data_manager.mass[self.obj.data_manager.slist[self.slist]])
        mass_baryon = mass_gas + mass_star

        self.masses['dm']      = self.obj.yt_dataset.quan(mass_dm, self.obj.units['mass'])
        self.masses['gas']     = self.obj.yt_dataset.quan(mass_gas, self.obj.units['mass'])
        self.masses['stellar'] = self.obj.yt_dataset.quan(mass_star, self.obj.units['mass'])
        self.masses['baryon']  = self.obj.yt_dataset.quan(mass_baryon, self.obj.units['mass'])
        self.masses['H']       = self.masses['gas'] * self.obj.simulation.XH


        if self.obj.simulation.nbh > 0:
            #mass_bh = np.sum(self.obj.data_manager.mass[self.obj.data_manager.bhlist][self.bhlist])
            if self.obj.data_manager.use_bhmass:
                mass_bh = self.obj.data_manager.bhmass[self.bhlist].d
            else:
                mass_bh = self.obj.data_manager.mass[self.obj.data_manager.bhlist][self.bhlist]

            if len(mass_bh):
                self.masses['bh']  = self.obj.yt_dataset.quan(np.max(mass_bh), self.obj.units['mass'])
                mass_baryon       += np.sum(mass_bh)
            else: self.masses['bh'] = self.obj.yt_dataset.quan(0.0, self.obj.units['mass'])


        if self.obj.simulation.ndust > 0:
            mass_dust = np.sum(self.obj.data_manager.mass[self.obj.data_manager.dlist][self.dlist])
            self.masses['dust']  = self.obj.yt_dataset.quan(mass_dust, self.obj.units['mass'])
        
        if self.obj.simulation.ndust <= 0:
            try:
                mass_dust    = np.sum(self.obj.data_manager.dustmass[self.obj.data_manager.glist[self.glist]])
                self.masses['dust'] = mass_dust
            except AttributeError:
                self.masses['dust'] = 0.0


        self.gas_fraction = 0.0
        if self.masses['baryon'] > 0:
            self.gas_fraction = self.masses['gas'].d / self.masses['baryon'].d

        self._calculate_total_mass()

    def _calculate_center_of_mass_quantities(self):
        """Calculate center-of-mass position and velocity.  From caesar_mika """
        def get_center_of_mass_quantity(quantity):  ## REFACTOR ME TO BE MORE GENERIC WITH SHAPE
            val  = np.zeros(3)
            for i in range(0,3):
                quantity_arr = getattr(self.obj.data_manager, quantity)[self.global_indexes, i]
                weights      = self.obj.data_manager.mass[self.global_indexes]
                if (quantity=='pos'):# We need to be consistent with periodic boundaries
                    if (quantity_arr.max() - quantity_arr.min())>0.5*self.obj.simulation.boxsize.d:
                        theta_i          = 6.283185307179586*quantity_arr/self.obj.simulation.boxsize.d #(2pi)
                        Zeta_i           = np.cos(theta_i)
                        Xhi_i            = np.sin(theta_i)
                        Theta            = np.arctan2(-np.average(Xhi_i, weights=weights), -np.average(Zeta_i, weights=weights))+3.141592653589793
                        val[i]           = self.obj.simulation.boxsize.d*Theta/6.283185307179586
                    else: val[i]  = np.average(quantity_arr, weights=weights)
                else: val[i]  = np.average(quantity_arr, weights=weights)
            return val

        self.pos = self.obj.yt_dataset.arr(get_center_of_mass_quantity('pos'), self.obj.units['length'])
        self.vel = self.obj.yt_dataset.arr(get_center_of_mass_quantity('vel'), self.obj.units['velocity'])
        cmpos = (self.pos.to('kpc')).d
        ppos  = self.obj.yt_dataset.arr(self.obj.data_manager.pos[self.global_indexes], self.obj.units['length'])
        ppos  = (ppos.to('kpc')).d

        #Minimum potential position
        pot = self.obj.data_manager.pot[self.global_indexes]
        pos = self.obj.data_manager.pos[self.global_indexes]
        self.minpotpos = self.obj.yt_dataset.arr(pos[np.argmin(pot)], self.obj.units['length'])

        #Compute distances from the center of mass or minimum potential?
        self.periodic_r = np.empty(len(ppos), dtype=np.float64)
        #get_periodic_r(self.obj.simulation.boxsize.to('kpc').d, cmpos, ppos, self.periodic_r) # COM
        get_periodic_r(self.obj.simulation.boxsize.to('kpc').d, self.minpotpos.to('kpc').d, ppos, self.periodic_r) # minimum potential
        #Put the periodic_r for further use and not to compute it everytime
        self.periodic_r = self.obj.yt_dataset.arr(self.periodic_r, 'kpc')

    """Calculate center-of-mass position and velocity.  Desika's version
    def _calculate_center_of_mass_quantities(self):
        def get_center_of_mass_quantity(quantity):  ## REFACTOR ME TO BE MORE GENERIC WITH SHAPE
            val  = np.zeros(3)
            for i in range(0,3):
                val[i] = np.sum(self.obj.data_manager.mass[self.global_indexes] * getattr(self.obj.data_manager, quantity)[self.global_indexes,i])
            val /= self.masses['total'].d
            return val

        self.pos = self.obj.yt_dataset.arr(get_center_of_mass_quantity('pos'), self.obj.units['length'])
        self.vel = self.obj.yt_dataset.arr(get_center_of_mass_quantity('vel'), self.obj.units['velocity'])
    """

    def _unbind(self):
        """Iterative procedure to unbind objects."""
        if not getattr(self.obj.simulation, 'unbind_%s' %
                       group_types[self.obj_type]):
            return

        if not hasattr(self, 'unbound_indexes'):
            self.unbound_indexes = {
                ptype_ints['gas']:[],
                ptype_ints['star']:[],
                ptype_ints['dm']:[],
                ptype_ints['bh']:[],
                ptype_ints['dust']:[],
            }
        if not hasattr(self, 'unbind_iterations'):
            self.unbind_iterations = 0        
        self.unbind_iterations += 1
        
        cmpos = (self.pos.to('kpc')).d
        ppos  = self.obj.yt_dataset.arr(self.obj.data_manager.pos[self.global_indexes], self.obj.units['length'])
        ppos  = (ppos.to('kpc')).d
        cmvel = (self.vel.to('kpc/s')).d
        pvels = self.obj.yt_dataset.arr(self.obj.data_manager.vel[self.global_indexes], self.obj.units['velocity'])
        pvels = (pvels.to('kpc/s')).d
        mass  = self.obj.yt_dataset.arr(self.obj.data_manager.mass[self.global_indexes], self.obj.units['mass'])
        mass  = (mass.to('Msun')).d

        init_mass = (self.masses['total'].to('Msun')).d

        r = np.empty(len(ppos), dtype=np.float64)
        get_periodic_r(self.obj.simulation.boxsize.d, cmpos, ppos, r)

        v2 = ( (pvels[:,0] - cmvel[0])**2 +
               (pvels[:,1] - cmvel[1])**2 +
               (pvels[:,2] - cmvel[2])**2 )
        
        energy = -(mass * self.obj.simulation.G.d * (init_mass - mass) / r) + (0.5 * mass * v2)

        positive = np.where(energy > 0)[0]
        if len(positive) > 0:
            positive = positive[::-1]
            for i in positive:
                global_index = self.global_indexes[i]
                self.unbound_indexes[self.obj.data_manager.ptype[global_index]].append(self.obj.data_manager.index[global_index])
                del self.global_indexes[i]

            self._assign_local_data()                        
            if not self._valid: return            
            self._calculate_total_mass()
            self._calculate_center_of_mass_quantities()
            self._unbind()

    def _calculate_gas_quantities(self):
        """Calculate gas quantities: SFR/Metallicity/Temperature."""
        self.sfr = self.obj.yt_dataset.quan(0.0, '%s/%s' % (self.obj.units['mass'],self.obj.units['time']))
        self.metallicities = dict(
            mass_weighted = self.obj.yt_dataset.quan(0.0, ''),
            sfr_weighted  = self.obj.yt_dataset.quan(0.0, '')            
        )
        self.temperatures['mass_weighted'] = self.obj.yt_dataset.quan(0.0, self.obj.units['temperature'])
        self.temperatures['sfr_weighted']  = self.obj.yt_dataset.quan(0.0, self.obj.units['temperature'])
        if self.ngas == 0:
            return

        gas_mass = self.obj.data_manager.mass[self.__glist]        
        gas_sfr  = self.obj.data_manager.gsfr[self.glist].d
        gas_Z    = self.obj.data_manager.gZ[self.glist].d
        gas_T    = self.obj.data_manager.gT[self.glist].d
        
        gas_mass_sum = np.sum(gas_mass)    
        gas_sfr_sum  = np.sum(gas_sfr)

        #moved this here to avoid many galaxies set with sfr = 1 if they really have sfr = 0
        self.sfr = self.obj.yt_dataset.quan(gas_sfr_sum, '%s/%s' % (self.obj.units['mass'], self.obj.units['time']))

        if gas_sfr_sum == 0:
            gas_sfr_sum = 1.0
        
        self.metallicities = dict(
            mass_weighted = self.obj.yt_dataset.quan(np.sum(gas_Z * gas_mass) / gas_mass_sum, ''),
            sfr_weighted  = self.obj.yt_dataset.quan(np.sum(gas_Z * gas_sfr ) / gas_sfr_sum,  '')
        )
        
        self.temperatures['mass_weighted'] = self.obj.yt_dataset.quan(np.sum(gas_T * gas_mass) / gas_mass_sum, self.obj.units['temperature'])
        self.temperatures['sfr_weighted']  = self.obj.yt_dataset.quan(np.sum(gas_T * gas_sfr ) / gas_sfr_sum,  self.obj.units['temperature'])

    def _calculate_star_quantities(self):
        """Calculate star quantities: Metallicity, ..."""
        if hasattr(self.obj.data_manager, 'sZ'):
            if len(self.slist)==0:
                stellar = 0.
            else:
                star_Z = self.obj.data_manager.sZ[self.slist].d
                star_mass = self.obj.data_manager.mass[self.__slist]
                star_mass_sum = np.sum(star_mass)
                stellar = np.sum(star_Z*star_mass)/star_mass_sum

            self.metallicities['stellar'] = self.obj.yt_dataset.quan(stellar, '')


    def _calculate_bh_quantities(self):
        if hasattr(self.obj.data_manager, 'bhmdot'):
            if self.nbh == 0:
                self.bhmdot = self.obj.yt_dataset.quan(0.0, '%s/%s' % (self.obj.units['mass'], self.obj.units['time']))
            else:
                if self.obj.data_manager.use_bhmass:
                    mass_bh = self.obj.data_manager.bhmass[self.bhlist].d
                else:
                    mass_bh = self.obj.data_manager.mass[self.obj.data_manager.bhlist][self.bhlist]
                bh_mdot     = self.obj.data_manager.bhmdot[self.bhlist].d
                self.bhmdot = self.obj.yt_dataset.quan(bh_mdot[np.argmax(mass_bh)], '%s/%s' % (self.obj.units['mass'], self.obj.units['time']))
        else:
            from yt.funcs import mylog
            mylog.info('No blackholes quantities to compute for groups')


    def _calculate_virial_quantities(self):
        """Calculates virial quantities such as r200, circular velocity, 
        and virial temperature."""
        sim      = self.obj.simulation        
        critical_density = sim.critical_density   # in Msun/kpc^3 PHYSICAL
        mass     = self.masses['total'].to('Msun')
      
        # Mika Rafieferantsoa's virial quantity computation
        pmass = self.obj.yt_dataset.arr(self.obj.data_manager.mass[self.global_indexes], self.obj.units['mass'])
        pmass = pmass.to('Msun')

        r_sort = np.argsort(self.periodic_r.d)
        pmass = np.cumsum(pmass[r_sort])  # cumulative mass from the center of the halo
        periodic_r = self.periodic_r.to('kpc').d[r_sort]  # sorted radii in ascending order

        #def get_r_vir(deltaC):
        #    """ returns r_vir in PHYSICAL kpc; deltaC is in units of critical density """
        #    return (3.0 * mass / (4.0 * np.pi * critical_density * deltaC))**(1./3.)

        collectRadii = np.zeros(len(sim.Densities), dtype = np.float64) # empty array of desired densities in Msun/kpc**3
        collectMasses = np.zeros(len(sim.Densities), dtype = np.float64) # empty array of masses at desired radii
        get_virial_mr(sim.Densities.d, pmass[::-1], periodic_r[::-1], collectRadii, collectMasses)
        self.radii['virial'] = self.obj.yt_dataset.quan(collectRadii[0], 'kpc')
        self.radii['r200c'] = self.obj.yt_dataset.quan(collectRadii[1], 'kpc')
        self.radii['r500c'] = self.obj.yt_dataset.quan(collectRadii[2], 'kpc')
        self.radii['r2500c'] = self.obj.yt_dataset.quan(collectRadii[3], 'kpc')
        
        PiFac = 4./3. * np.pi
        self.masses['virial'] = self.obj.yt_dataset.quan(collectMasses[0], 'Msun')
        self.masses['m200c'] = self.obj.yt_dataset.quan(collectMasses[1], 'Msun')
        self.masses['m500c'] = self.obj.yt_dataset.quan(collectMasses[2], 'Msun')
        self.masses['m2500c'] = self.obj.yt_dataset.quan(collectMasses[3], 'Msun')
        #self.masses['virial'] = 100.*critical_density * PiFac*self.radii['virial']**3
        #self.masses['m200c'] = 200.*critical_density * PiFac*self.radii['r200c']**3
        #self.masses['m500c'] = 500.*critical_density * PiFac*self.radii['r500c']**3
        #self.masses['m2500c'] = 2500.*critical_density * PiFac*self.radii['r2500c']**3

        #print('radii:',collectMasses,collectRadii,self.radii['r200c'],self.radii['r500c'],self.masses['m200c'],self.masses['m500c'],collectMasses[1]/self.masses['m200c'],collectMasses[2]/self.masses['m500c'])
        # eq 1 of Mo et al 2002
        self.radii['r200'] = (sim.G * mass / (100.0 * sim.Om_z * sim.H_z**2))**(1./3.)
        
        # eq 1 of Mo et al 2002
        vc = (np.sqrt( sim.G * mass / self.radii['r200'] )).to('km/s')

        # eq 4 of Mo et al 2002 (K)
        vT = self.obj.yt_dataset.quan(3.6e5 * (vc.d / 100.0)**2, 'K')

        # convert units
        self.radii['virial'] = self.radii['virial'].to(self.obj.units['length'])
        self.radii['r200c']  = self.radii['r200c'].to(self.obj.units['length'])
        self.radii['r500c']  = self.radii['r500c'].to(self.obj.units['length'])
        self.radii['r2500c']  = self.radii['r2500c'].to(self.obj.units['length'])
        self.radii['r200']   = self.radii['r200'].to(self.obj.units['length'])        
        vc = vc.to(self.obj.units['velocity'])
        vT = vT.to(self.obj.units['temperature'])

        self.temperatures['virial'] = vT

        for k in self.masses:
            if isinstance(self.masses[k], float):
                self.masses[k] = self.obj.yt_dataset.quan(self.masses[k], self.obj.units['mass'])
            else:
                self.masses[k] = self.masses[k].to(self.obj.units['mass'])
        

        self.virial_quantities = dict(
            radius = self.radii['virial'],
            r200c  = self.radii['r200c'],
            r500c  = self.radii['r500c'],
            r2500c  = self.radii['r2500c'],
            r200   = self.radii['r200'],
            circular_velocity = vc,
            temperature = vT
        )


    def _calculate_velocity_dispersions(self):
        """Calculate velocity dispersions for the various components."""
        def get_sigma(filtered_v,filtered_m):
            if len(filtered_v) == 0: return 0.0
            mv = np.array([filtered_m[i]*filtered_v[i] for i in range(len(filtered_v))])
            v_std = np.std(mv,axis=0)/np.mean(filtered_m)
            return np.sqrt(v_std.dot(v_std))

        ptypes = self.obj.data_manager.ptype[self.global_indexes]
        v = self.obj.data_manager.vel[self.global_indexes]
        m = self.obj.data_manager.mass[self.global_indexes]
        
        self.velocity_dispersions = dict() 
        
        self.velocity_dispersions['all']     = get_sigma(v,m)
        self.velocity_dispersions['dm']      = get_sigma(v[ ptypes == ptype_ints['dm']],m[ ptypes == ptype_ints['dm']])
        self.velocity_dispersions['baryon']  = get_sigma(v[(ptypes == ptype_ints['gas']) | (ptypes == ptype_ints['star'])],m[(ptypes == ptype_ints['gas']) | (ptypes == ptype_ints['star'])])
        self.velocity_dispersions['gas']     = get_sigma(v[ ptypes == ptype_ints['gas']],m[ ptypes == ptype_ints['gas']])
        self.velocity_dispersions['stellar'] = get_sigma(v[ ptypes == ptype_ints['star']],m[ ptypes == ptype_ints['star']])
        #if np.log10(self.masses['total'])>12: print 'sigma',np.log10(self.masses['total']),self.velocity_dispersions['all'],self.velocity_dispersions['dm'],self.velocity_dispersions['gas'],self.velocity_dispersions['stellar']
        
        for k,v in six.iteritems(self.velocity_dispersions):
            self.velocity_dispersions[k] = self.obj.yt_dataset.quan(v, self.obj.units['velocity'])
            
    def _calculate_angular_quantities(self):
        """Calculate angular momentum, spin, max_vphi and max_vr."""
        pos  = self.obj.yt_dataset.arr(self.obj.data_manager.pos[self.global_indexes],  self.obj.units['length'])
        vel  = self.obj.yt_dataset.arr(self.obj.data_manager.vel[self.global_indexes],  self.obj.units['velocity'])
        mass = self.obj.yt_dataset.arr(self.obj.data_manager.mass[self.global_indexes], self.obj.units['mass'])

        px = mass * vel[:,0]
        py = mass * vel[:,1]
        pz = mass * vel[:,2]        
        x  = (pos[:,0] - self.pos[0]).to('km')
        y  = (pos[:,1] - self.pos[1]).to('km')
        z  = (pos[:,2] - self.pos[2]).to('km')

        Lx = np.sum( y*pz - z*py )
        Ly = np.sum( z*px - x*pz )
        Lz = np.sum( x*py - y*px )
        L  = np.sqrt(Lx**2 + Ly**2 + Lz**2)
        #self.angular_momentum        = self.obj.yt_dataset.quan(L, Lx.units)
        self.angular_momentum_vector = self.obj.yt_dataset.arr([Lx.d,Ly.d,Lz.d], Lx.units)

        
        # Bullock spin or lambda prime
        #self.spin = self.angular_momentum / (1.4142135623730951 *
        self.spin_param = self.obj.yt_dataset.quan(L, Lx.units) / (1.4142135623730951 *
                                             self.masses['total'] *
                                             self.virial_quantities['circular_velocity'].to('km/s') *
                                             self.virial_quantities['r200c'].to('km'))

        PHI   = np.arctan2(Ly.d,Lx.d)
        THETA = np.arccos(Lz.d/L.d)
        
        ex = np.sin(THETA) * np.cos(PHI)
        ey = np.sin(THETA) * np.sin(PHI)
        ez = np.cos(THETA)

        from caesar.utils import rotator
        ALPHA = np.arctan2(Ly.d, Lz.d)
        p     = rotator(np.array([ex,ey,ez]), ALPHA)
        BETA  = np.arctan2(p[0],p[2])
        self.rotation_angles = dict(ALPHA=ALPHA, BETA=BETA)

        ## need max_vphi and max_vr
        rotated_pos = rotator(pos.d, ALPHA, BETA)
        rotated_vel = rotator(vel.d, ALPHA, BETA)

        r    = np.sqrt(rotated_pos[:,0]**2 + rotated_pos[:,1]**2)
        vphi = (rotated_vel[:,0] * -1. * rotated_pos[:,1] + rotated_vel[:,1] * rotated_pos[:,0]) / r
        vr   = (rotated_vel[:,0] *       rotated_pos[:,0] + rotated_vel[:,1] * rotated_pos[:,1]) / r

        self.max_vphi = self.obj.yt_dataset.quan(np.max(vphi), self.obj.units['velocity'])
        self.max_vr   = self.obj.yt_dataset.quan(np.max(vr)  , self.obj.units['velocity'])
            
    def _calculate_radial_quantities(self):
        """ Calculate various component radii and half radii """
        from caesar.group_funcs import get_half_mass_radius, get_full_mass_radius
        
        r = np.empty(len(self.global_indexes), dtype=np.float64)
        get_periodic_r(self.obj.simulation.boxsize.d, self.pos.d, self.obj.data_manager.pos[self.global_indexes], r)
        
        rsort = np.argsort(r)
        r     = r[rsort]
        mass  = self.obj.data_manager.mass[self.global_indexes][rsort]
        ptype = self.obj.data_manager.ptype[self.global_indexes][rsort]

        radial_categories = dict(
            total   = [ptype_ints['gas'],ptype_ints['star'],ptype_ints['dm'],ptype_ints['bh'],ptype_ints['dust']],
            baryon  = [ptype_ints['gas'],ptype_ints['star']],
            gas     = [ptype_ints['gas']],
            stellar = [ptype_ints['star']],
            dm      = [ptype_ints['dm']],
        )
        
        half_masses = {}
        for k,v in six.iteritems(self.masses):
            half_masses[k] = 0.5 * v            

        for k,v in six.iteritems(radial_categories):
            if k == 'dm' and self.obj_type == 'galaxy': continue
            binary = 0
            for p in v:
                binary += 2**p

            full_r = get_full_mass_radius(r[::-1], ptype[::-1], binary)
            self.radii[k] = self.obj.yt_dataset.quan(full_r, self.obj.units['length'])
            
            half_r = get_half_mass_radius(mass, r, ptype, half_masses[k], binary)
            self.radii['%s_half_mass' % k] = self.obj.yt_dataset.quan(half_r, self.obj.units['length'])

    def write_IC_mask(self, ic_ds, filename, search_factor = 2.5,radius_type='total'):
        """Write MUSIC initial condition mask to disk.  If called on
        a galaxy it will look for the parent halo in the IC.

        Parameters
        ----------
        ic_ds : yt dataset
            The initial condition dataset via ``yt.load()``.
        filename : str
            The filename of which to write the mask to.  If a full 
            path is not supplied then it will be written in the 
            current directory.
        search_factor : float, optional
            How far from the center to select DM particles. Default is
            2.5
        print_extents : bool, optional
            Print MUSIC extents for cuboid after mask creation
    
        Examples
        --------
        >>> import yt
        >>> import caesar
        >>>
        >>> snap = 'my_snapshot.hdf5'
        >>> ic   = 'IC.dat'
        >>>
        >>> ds    = yt.load(snap)
        >>> ic_ds = yt.load(ic)
        >>>
        >>> obj = caesar.load('caesar_my_snapshot.hdf5', ds)
        >>> obj.galaxies[0].write_IC_mask(ic_ds, 'mymask.txt')
        
        """ 
        from caesar.zoom_funcs import write_IC_mask
        write_IC_mask(self, ic_ds, filename, search_factor,radius_type=radius_type)
            
    def vtk_vis(self, rotate=False):
        """Method to render this group's points via VTK.

        Parameters
        ----------
        rotate : boolean
            Align angular momentum vector with the z-axis before 
            rendering?

        Notes
        -----
        Opens up a pyVTK window; you must have VTK installed to use
        this method.  It is easiest to install via 
        ``conda install vtk``.
        
        """
        self.obj.data_manager.load_particle_data()
        from caesar.vtk_funcs import group_vis
        group_vis(self, rotate=rotate)


    def info(self):
        """Method to quickly print out object attributes."""
        pdict = {}
        for k,v in six.iteritems(self.__dict__):
            if k in info_blacklist: continue
            pdict[k] = v
        from pprint import pprint
        pprint(pdict)
        pdict = None

    def contamination_check(self, lowres=[2,3,5], search_factor=2.5,
                            printer=True):
        """Check for low resolution particle contamination.

        This method checks for low-resolution particles within 
        ``search_factor`` of the maximum halo radius.  When this
        method is called on a galaxy, it refers to the parent halo.

        Parameters
        ----------
        lowres : list, optional
            Particle types to be considered low-res.  Defaults to
            [2,3,5]; if your simulation contains blackholes you will
            want to pass in [2,3]; if your simulation contains active
			dust particles you will not include 3.
        search_factor : float, optional
            Factor to expand the maximum halo radius search distance
            by.  Default is 2.5
        printer : boolean, optional
            Print results?

        Notes
        -----
        This method currently ONLY works on GADGET/GIZMO HDF5 files.

        """
        from yt.funcs import mylog
        from caesar.zoom_funcs import construct_lowres_tree

        construct_lowres_tree(self, lowres)

        if self.obj_type == 'halo':
            halo = self
            ID   = 'Halo %d' % self.GroupID
        elif self.obj_type == 'galaxy':
            if self.halo == None:
                raise Exception('Galaxy %d has no halo!' % self.GroupID)                
            halo = self.halo
            ID   = "Galaxy %d's halo (ID %d)" % (self.GroupID, halo.GroupID)
        
        r = halo.radii['virial'].d * search_factor

        result  = self.obj._lowres['TREE'].query_ball_point(halo.pos.d, r)
        ncontam = len(result)
        lrmass  = np.sum(self.obj._lowres['MASS'][result])

        self.contamination = lrmass / halo.masses['total'].d
        
        if not printer:
            return
        
        if ncontam > 0:
            mylog.warning('%s has %0.2f%% mass contamination ' \
                          '(%d LR particles with %0.2e % s)' %
                          (ID, self.contamination * 100.0, ncontam,
                           lrmass, halo.masses['total'].units))
        else:
            mylog.info('%s has NO contamination!' % ID)
            
class Galaxy(Group):
    """Galaxy class which has the central boolean."""
    obj_type = 'galaxy'    
    def __init__(self,obj):
        super(Galaxy, self).__init__(obj)
        self.central = False
        self.halo = None
        
class Halo(Group):
    """Halo class which has the dmlist attribute, and child boolean."""
    obj_type = 'halo'
    dmlist   = GroupList('dmlist')
    def __init__(self,obj):
        super(Halo, self).__init__(obj)
        self.child = False
        self.galaxies = []
        self.central_galaxy = None
        self.satellite_galaxies = []
        self.galaxy_index_list = np.array([])

class Cloud(Group):
    """Cloud class which has the central boolean."""
    obj_type = 'cloud'    
    def __init__(self,obj):
        super(Cloud, self).__init__(obj)
        self.central = False
        self.galaxy = None
        self.halo = None

def create_new_group(obj, group_type):
    """Simple function to create a new instance of a specified 
    :class:`group.Group`.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Main caesar object.
    group_type : {'halo', 'galaxy','cloud'}
        Which type of group?  Options are: `halo` and `galaxy`.

    Returns
    -------
    group : :class:`group.Group`
        Subclass :class:`group.Halo` or :class:`group.Galaxy`.

    """
    if group_type == 'halo':
        return Halo(obj)
    elif group_type == 'galaxy':
        return Galaxy(obj)
    elif group_type == 'cloud':
        return Cloud(obj)
