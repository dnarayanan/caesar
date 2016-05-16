import numpy as np

class SimulationAttributes(object):
    def __init__(self):
        pass

    def create_attributes(self, obj):    
        ds = obj.yt_dataset

        self.cosmological_simulation = ds.cosmological_simulation

        self.XH              = 0.76        
        self.redshift        = ds.current_redshift
        self.time            = 1.0 / (1.0 + self.redshift)
        self.omega_matter    = ds.omega_matter
        self.omega_lambda    = ds.omega_lambda
        self.fullpath        = ds.fullpath
        self.basename        = ds.basename
        self.hubble_constant = ds.hubble_constant
        self.parameters      = ds.parameters
        
        self.boxsize         = ds.domain_width[0].to(obj.units['length'])
        self.boxsize_units   = str(self.boxsize.units)

        self.search_radius   = ds.quan(500.0, 'kpc').to(obj.units['length'])

        H0 = self.hubble_constant * 100.0
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
        else:
            H_z = H0

            if hasattr(ds, 'cosmology') and hasattr(ds.cosmology, 'omega_matter'):
                self.Om_z = ds.cosmology.omega_matter
            else:
                self.Om_z = 0.3

            # correct for NON comoving coordinates in non-cosmo sims
            if obj.units['length'].endswith('cm') and obj.units['length'] != 'cm':
                obj.units['length'] = obj.units['length'][:-2]

                
        self.H_z = ds.quan(H_z * 3.24077929e-20, '1/s')
        self.G   = ds.quan(4.51691362044e-39, 'kpc**3/(Msun * s**2)')  ## kpc^3 / (Msun s^2)

        self.critical_density = ds.quan(
            (3.0 * self.H_z.d**2) / (8.0 * np.pi * self.G.d),
            'Msun / kpc**3'
        )        

        
    def _serialize(self, obj, hd):
        from yt.extern import six
        from yt.units.yt_array import YTQuantity

        hdd  = hd.create_group('simulation_attributes')
        
        units = {}        
        for k,v in six.iteritems(self.__dict__):
            if isinstance(v, YTQuantity):
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
        if 'simulation_attributes' not in hd.keys():
            return
        from yt.extern import six
        from yt.units.yt_array import YTQuantity
        
        hdd = hd['simulation_attributes']
        for k,v in six.iteritems(hdd.attrs):
            setattr(self, k, v)

        uhdd = hdd['units']
        for k,v in six.iteritems(uhdd.attrs):
            setattr(self, k, YTQuantity(getattr(self, k), v, registry=obj.unit_registry))

        phdd = hdd['parameters']
        self.parameters = {}
        for k,v in six.iteritems(phdd.attrs):
            self.parameters[k] = v
