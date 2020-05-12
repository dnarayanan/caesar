import numpy as np
from yt.funcs import mylog

class SimulationAttributes(object):
    """A class to hold simulation attributes."""
    def __init__(self):
        pass

    def create_attributes(self, obj):
        """After loading in a yt dataset, store local attributes here."""
        ds = obj.yt_dataset

        self.cosmological_simulation = ds.cosmological_simulation

        self.XH              = 0.76        
        self.redshift        = ds.current_redshift
        self.scale_factor    = 1.0 / (1.0 + self.redshift)
        self.time            = ds.current_time
        self.omega_matter    = ds.omega_matter
        self.omega_lambda    = ds.omega_lambda
        self.fullpath        = ds.fullpath
        self.basename        = ds.basename
        self.hubble_constant = ds.hubble_constant
        self.parameters      = ds.parameters

        if not self.cosmological_simulation:
            # correct for NON comoving coordinates in non-cosmo sims
            if obj.units['length'].endswith('cm') and obj.units['length'] != 'cm':
                obj.units['length'] = obj.units['length'][:-2]
                
        self.boxsize         = ds.domain_width[0].to(obj.units['length'])
        self.boxsize_units   = str(self.boxsize.units)

        sru = 'kpc'
        if self.cosmological_simulation: sru += 'cm'
        self.search_radius   = ds.arr([300,1000,3000], sru).to(obj.units['length'])  # default values

        H0 = ds.quan(self.hubble_constant * 100.0 * 3.24077929e-20, '1/s')
        if self.cosmological_simulation:
            Om_0 = ds.cosmology.omega_matter
            Ol_0 = ds.cosmology.omega_lambda
            Ok_0 = ds.cosmology.omega_curvature
            self.E_z = np.sqrt(
                Ol_0 +
                Ok_0 * (1.0 + self.redshift)**2 +
                Om_0 * (1.0 + self.redshift)**3
            )
            self.Om_z = Om_0 * (1.0 + self.redshift)**3 / self.E_z**2
            H_z       = H0 * self.E_z
            #Kitayama & Suto 1996 v.469, p.480
            fomega = Om_0*(1.+self.redshift)**3 / self.E_z**2
            if fomega >=1.0: mylog.warning('fomega out of bounds! fomega=%g %g %g %g'%(fomega,self.Om_z,self.redshift,Ol_0))
        else:
            H_z = H0

            if hasattr(ds, 'cosmology') and hasattr(ds.cosmology, 'omega_matter'):
                self.Om_z = ds.cosmology.omega_matter
            else:
                self.Om_z = 0.3
            fomega = self.Om_z

                
        self.H_z = H_z
        self.G   = ds.quan(4.51691362044e-39, 'kpc**3/(Msun * s**2)')  ## kpc^3 / (Msun s^2)

        self.critical_density = ds.quan(
            (3.0 * H_z**2) / (8.0 * np.pi * self.G.d),
            'Msun / kpc**3'
        )
        virial_density = (177.65287921960845*(1. + 0.4093*(1./fomega - 1.)**0.9052) - 1.)*self.Om_z #Kitayama & Suto 1996 v.469, p.480
        #print(self.critical_density, self.critical_density.to('g/cm**3'),self.H_z.to('1/s'),(3.0 * H0.d**2) / (8.0 * np.pi * self.G.d))
        # Romeel: Removed virial density, because the FOF with b=0.2*MIS 
        # only finds contour enclosing ~200, so halos do not have all the particles they
        # need to compute spherical ~100xrhocrit. 
        #self.Densities = np.array([virial_density*self.critical_density.to('Msun/kpc**3').d,
        self.Densities = np.array([ 200.*self.critical_density.to('Msun/kpc**3').d,
                                    500.*self.critical_density.to('Msun/kpc**3').d,
                                    2500.*self.critical_density.to('Msun/kpc**3').d])
        self.Densities = ds.arr(self.Densities, 'Msun/kpc**3')


        from caesar.property_manager import has_ptype
        self.baryons_present = False
        if has_ptype(obj, 'gas') or has_ptype(obj, 'star'):
            self.baryons_present = True

        
    def _serialize(self, obj, hd):
        import  six
        from yt.units.yt_array import YTArray

        hdd  = hd.create_group('simulation_attributes')
        
        units = {}        
        for k,v in six.iteritems(self.__dict__):
            if isinstance(v, YTArray):
                hdd.attrs.create(k, v.d)
                units[k] = v.units
            elif isinstance(v, (int, float, bool, np.number)):
                hdd.attrs.create(k, v)
            elif isinstance(v, str):
                hdd.attrs.create(k, v.encode('utf8'))
                
        uhdd = hdd.create_group('units')               
        for k,v in six.iteritems(units):
            uhdd.attrs.create(k, str(v).encode('utf8'))
            
        phdd = hdd.create_group('parameters')
        for k,v in six.iteritems(self.parameters):
            phdd.attrs.create(k, v)

            
    def _unpack(self, obj, hd):
        import six
        if 'simulation_attributes' not in hd.keys():
            return
        from yt.units.yt_array import YTArray
        
        hdd = hd['simulation_attributes']
        for k,v in six.iteritems(hdd.attrs):
            setattr(self, k, v)

        uhdd = hdd['units']
        for k,v in six.iteritems(uhdd.attrs):
            setattr(self, k, YTArray(getattr(self, k), v, registry=obj.unit_registry))

        phdd = hdd['parameters']
        self.parameters = {}
        for k,v in six.iteritems(phdd.attrs):
            self.parameters[k] = v
