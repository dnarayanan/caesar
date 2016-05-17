import six
import numpy as np

from caesar.property_getter import ptype_ints
from caesar.group_funcs import get_periodic_r

UNBIND_HALOS    = False
UNBIND_GALAXIES = False
MINIMUM_STARS_PER_GALAXY = 32
MINIMUM_DM_PER_HALO      = 32

class GroupList(object):
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
    glist = GroupList('glist')
    slist = GroupList('slist')    
    
    def __init__(self,obj):
        self.obj = obj

        self.masses = {}
        self.radii = {} 
        self.temperatures = {}
        
    def _append_global_index(self, i):
        if not hasattr(self, 'global_indexes'):
            self.global_indexes = []
        self.global_indexes.append(i)

    @property
    def _valid(self):
        if self.obj_type == 'halo' and self.ndm < MINIMUM_DM_PER_HALO:
            return False
        elif self.obj_type == 'galaxy' and self.nstar < MINIMUM_STARS_PER_GALAXY:
            return False
        else:
            return True

    def _delete_attribute(self,a):
        if hasattr(self,a):
            delattr(self,a)

    def _delete_key(self,d,k):
        if k in d:
            del d[k]
            
    def _remove_dm_references(self):
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


    def _process_group(self):
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
            
        self._cleanup()

        
    def _assign_local_data(self):
        ptypes  = self.obj.data_manager.ptype[self.global_indexes]
        indexes = self.obj.data_manager.index[self.global_indexes]

        # lists for the concatinated global list
        self.__glist = np.where(ptypes == ptype_ints['gas'])[0]

        # individual global lists
        self.glist  = indexes[np.where(ptypes == ptype_ints['gas'])[0]]
        self.slist  = indexes[np.where(ptypes == ptype_ints['star'])[0]]
        self.dmlist = indexes[np.where(ptypes == ptype_ints['dm'])[0]]

        self.ngas  = len(self.glist)
        self.nstar = len(self.slist)
        self.ndm   = len(self.dmlist)

        if self.obj.nbh > 0:
            self.bhlist = indexes[np.where(ptypes == ptype_ints['bh'])[0]]
            self.nbh    = len(self.bhlist)

    def _calculate_total_mass(self):
        """ calculate the total mass of the object """
        self.masses['total'] = self.obj.yt_dataset.quan(np.sum(self.obj.data_manager.mass[self.global_indexes]), self.obj.units['mass'])
        
    def _calculate_masses(self):
        """ calculate various total masses """
        mass_dm     = np.sum(self.obj.data_manager.mass[self.obj.data_manager.dmlist][self.dmlist])
        mass_gas    = np.sum(self.obj.data_manager.mass[self.obj.data_manager.glist][self.glist])
        mass_star   = np.sum(self.obj.data_manager.mass[self.obj.data_manager.slist][self.slist])
        mass_baryon = mass_gas + mass_star

        self.masses['dm']      = self.obj.yt_dataset.quan(mass_dm, self.obj.units['mass'])
        self.masses['gas']     = self.obj.yt_dataset.quan(mass_gas, self.obj.units['mass'])
        self.masses['stellar'] = self.obj.yt_dataset.quan(mass_star, self.obj.units['mass'])
        self.masses['baryon']  = self.obj.yt_dataset.quan(mass_baryon, self.obj.units['mass'])
        self.masses['H']       = self.masses['gas'] * self.obj.simulation.XH
        
        if self.obj.nbh > 0:
            mass_bh = np.sum(self.obj.data_manager.mass[self.obj.data_manager.bhlist][self.bhlist])
            self.masses['bh']  = self.obj.yt_dataset.quan(mass_bh, self.obj.units['mass'])

        self.gas_fraction = 0.0
        if self.masses['baryon'] > 0:
            self.gas_fraction = self.masses['gas'].d / self.masses['baryon'].d

        self._calculate_total_mass()
            
    def _calculate_center_of_mass_quantities(self):
        """ calculate center-of-mass position and velocity """
        def get_center_of_mass_quantity(quantity):  ## REFACTOR ME TO BE MORE GENERIC WITH SHAPE
            val  = np.zeros(3)
            for i in range(0,3):
                val[i] = np.sum(self.obj.data_manager.mass[self.global_indexes] * getattr(self.obj.data_manager, quantity)[self.global_indexes,i])
            val /= self.masses['total'].d
            return val

        self.pos = self.obj.yt_dataset.arr(get_center_of_mass_quantity('pos'), self.obj.units['length'])
        self.vel = self.obj.yt_dataset.arr(get_center_of_mass_quantity('vel'), self.obj.units['velocity'])

    def _unbind(self):
        """ Iterative procedure to unbind objects. """        
        if self.obj_type == 'halo' and not UNBIND_HALOS:
            return
        elif self.obj_type == 'galaxy' and not UNBIND_GALAXIES:
            return        

        if not hasattr(self, 'unbound_indexes'):
            self.unbound_indexes = {
                ptype_ints['gas']:[],
                ptype_ints['star']:[],
                ptype_ints['dm']:[],
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

        if gas_sfr_sum == 0:
            gas_sfr_sum = 1.0
        
        self.metallicities = dict(
            mass_weighted = self.obj.yt_dataset.quan(np.sum(gas_Z * gas_mass) / gas_mass_sum, ''),
            sfr_weighted  = self.obj.yt_dataset.quan(np.sum(gas_Z * gas_sfr ) / gas_sfr_sum,  '')
        )
        
        self.temperatures['mass_weighted'] = self.obj.yt_dataset.quan(np.sum(gas_T * gas_mass) / gas_mass_sum, self.obj.units['temperature'])
        self.temperatures['sfr_weighted']  = self.obj.yt_dataset.quan(np.sum(gas_T * gas_sfr ) / gas_sfr_sum,  self.obj.units['temperature'])
        self.sfr = self.obj.yt_dataset.quan(gas_sfr_sum, '%s/%s' % (self.obj.units['mass'], self.obj.units['time']))
        
    def _calculate_virial_quantities(self):
        """ Calculates virial quantities such as r200, circular velocity, and virial temperature """
        sim      = self.obj.simulation        
        rho_crit = sim.critical_density   # in Msun/kpc^3 PHYSICAL
        mass     = self.masses['total'].to('Msun')
        
        def get_r_vir(deltaC):
            """ returns r_vir in PHYSICAL kpc """
            return (3.0 * mass / (4.0 * np.pi * rho_crit * deltaC))**(1./3.)

        # Bryan & Norman 1998
        self.radii['virial'] = self.obj.yt_dataset.quan(get_r_vir(18.0 * np.pi**2), 'kpc')
        self.radii['r200c']  = self.obj.yt_dataset.quan(get_r_vir(200.0), 'kpc')

        # eq 1 of Mo et al 2002
        self.radii['r200'] = (sim.G * mass / (100.0 * sim.Om_z * sim.H_z**2))**(1./3.)
        
        # eq 1 of Mo et al 2002
        vc = (np.sqrt( sim.G * mass / self.radii['r200'] )).to('km/s')

        # eq 4 of Mo et al 2002 (K)
        vT = self.obj.yt_dataset.quan(3.6e5 * (vc.d / 100.0)**2, 'K')

        # convert units
        self.radii['virial'] = self.radii['virial'].to(self.obj.units['length'])
        self.radii['r200c']  = self.radii['r200c'].to(self.obj.units['length'])
        self.radii['r200']   = self.radii['r200'].to(self.obj.units['length'])        
        vc = vc.to(self.obj.units['velocity'])
        vT = vT.to(self.obj.units['temperature'])

        self.temperatures['virial'] = vT
        
        self.virial_quantities = dict(
            radius = self.radii['virial'],
            r200c  = self.radii['r200c'],
            r200   = self.radii['r200'],
            circular_velocity = vc,
            temperature = vT
        )


    def _calculate_velocity_dispersions(self):
        """ Calculate velocity dispersions for the various components """
        def get_sigma(filtered_v):
            if len(filtered_v) == 0:
                return 0.0            
            v_mean = np.mean(filtered_v)
            v_diff = filtered_v - v_mean
            return np.std(v_diff)

        ptypes = self.obj.data_manager.ptype[self.global_indexes]
        particle_vel = self.obj.data_manager.vel[self.global_indexes]
        v = np.sqrt( particle_vel[:,0]**2 +
                     particle_vel[:,1]**2 +
                     particle_vel[:,2]**2 )
            
        self.velocity_dispersions = dict() 

        self.velocity_dispersions['all']     = get_sigma(v)
        self.velocity_dispersions['dm']      = get_sigma(v[ ptypes == ptype_ints['dm']])
        self.velocity_dispersions['baryon']  = get_sigma(v[(ptypes == ptype_ints['gas']) | (ptypes == ptype_ints['star'])])
        self.velocity_dispersions['gas']     = get_sigma(v[ ptypes == ptype_ints['gas']])
        self.velocity_dispersions['stellar'] = get_sigma(v[ ptypes == ptype_ints['star']])

        for k,v in six.iteritems(self.velocity_dispersions):
            self.velocity_dispersions[k] = self.obj.yt_dataset.quan(v, self.obj.units['velocity'])
            
    def _calculate_angular_quantities(self):
        """ Calculate angular momentum, spin, max_vphi and max_vr """
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
        self.angular_momentum        = self.obj.yt_dataset.quan(L, Lx.units)
        self.angular_momentum_vector = self.obj.yt_dataset.arr([Lx.d,Ly.d,Lz.d], Lx.units)

        
        # Bullock spin or lambda prime
        self.spin = self.angular_momentum / (1.4142135623730951 *
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
            total   = [ptype_ints['gas'],ptype_ints['star'],ptype_ints['dm'],ptype_ints['bh']],
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

            
class Galaxy(Group):
    obj_type = 'galaxy'    
    def __init__(self,obj):
        super(Galaxy, self).__init__(obj)
        self.central = False
        
class Halo(Group):
    obj_type = 'halo'
    dmlist   = GroupList('dmlist')
    def __init__(self,obj):
        super(Halo, self).__init__(obj)
        self.child = False

def create_new_group(obj, group_type):
    if group_type == 'halo':
        return Halo(obj)
    elif group_type == 'galaxy':
        return Galaxy(obj)
