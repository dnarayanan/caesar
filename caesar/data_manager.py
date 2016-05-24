import numpy as np

from caesar.property_getter import ptype_ints, get_particles_for_FOF, get_property, has_property

class DataManager(object):
    """Class to handle the initial IO and data storage for the duration of
    a CAESAR run.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Main CAESAR object.
    gas_data : boolean
        If True load additional gas data (sfr/temp/etc)

    """    
    def __init__(self, obj, gas_data):
        self.obj = obj
        self.blackholes = False
        self.determine_ptypes()
        self.load_data()
        if gas_data:
            self.load_gas_data()
            
    def determine_ptypes(self):
        """Determines what particle/field types to collect."""
        self.ptypes = ['gas','star']
        if 'blackholes' in self.obj._kwargs and self.obj._kwargs['blackholes']:
            self.ptypes.append('bh')
            self.blackholes = True
        self.ptypes.append('dm')

        #if self.obj._ds_type.grid:
        #    self.ptypes.remove('gas')
        #print self.ptypes
        
    def load_data(self):
        """Loads positions, velocities, masses, particle types, and indexes.
        Assigns a global glist, slist, dmlist, and bhlist used
        throughout the group analysis.  Finally assigns
        ngas/nstar/ndm/nbh values."""

        pdata      = get_particles_for_FOF(self.obj, self.ptypes)
        self.pos   = pdata['pos']
        self.vel   = pdata['vel']
        self.mass  = pdata['mass']
        self.ptype = pdata['ptype']
        self.index = pdata['indexes']
        pdata      = None
                
        self.glist  = np.where(self.ptype == ptype_ints['gas'])[0]
        self.slist  = np.where(self.ptype == ptype_ints['star'])[0]
        self.dmlist = np.where(self.ptype == ptype_ints['dm'])[0]        
        self.bhlist = np.where(self.ptype == ptype_ints['bh'])[0]

        # move these to obj.simulation?
        self.obj.ngas  = len(self.glist)
        self.obj.nstar = len(self.slist)
        self.obj.ndm   = len(self.dmlist)
        self.obj.nbh   = len(self.bhlist)


    def load_gas_data(self):
        """If gas is present loads gas SFR/Metallicity/Temperatures."""
        if self.obj.ngas == 0:
            return

        sfr_unit = '%s/%s' % (self.obj.units['mass'], self.obj.units['time'])

        sfr = self.obj.yt_dataset.arr(np.zeros(self.obj.ngas), sfr_unit)
        gZ  = self.obj.yt_dataset.arr(np.zeros(self.obj.ngas), '')        
        gT  = self.obj.yt_dataset.arr(np.zeros(self.obj.ngas), self.obj.units['temperature'])
            
        if has_property(self.obj, 'gas', 'sfr'):
            sfr = get_property(self.obj, 'sfr', 'gas').to(sfr_unit)

        if has_property(self.obj, 'gas', 'metallicity'):            
            gZ  = get_property(self.obj, 'metallicity', 'gas')

        if has_property(self.obj, 'gas', 'temperature'):
            gT  = get_property(self.obj, 'temperature', 'gas').to(self.obj.units['temperature'])

        self.gsfr = sfr
        self.gZ   = gZ
        self.gT   = gT
