from caesar.property_getter import DatasetType
from caesar.particle_list import ParticleListContainer
from caesar.simulation_attributes import SimulationAttributes

from yt.funcs import mylog, get_hash

class CAESAR(object):
    """ Master CAESAR class """

    def __init__(self, ds=0, *args, **kwargs):
        self.args   = args
        self.kwargs = kwargs
        self._ds    = 0

        self.units = dict(
            mass='Msun',
            length='kpccm',
            velocity='km/s',
            time='yr',
            temperature='K'
        )

        self.global_particle_lists = ParticleListContainer(self)
        self.simulation = SimulationAttributes()
        self.yt_dataset = ds
        
    @property
    def yt_dataset(self):
        return self._ds
    @yt_dataset.setter
    def yt_dataset(self,value):
        if value == 0: return

        if not hasattr(value, 'dataset_type'):
            raise IOError('not a yt dataset?')

        infile = '%s/%s' % (value.fullpath, value.basename)

        if hasattr(self, 'hash'):
            hash = get_hash(infile)
            if hash != self.hash:
                raise IOError('hash mismatch!')
            else:
                self._ds = value
        else:
            self._ds  = value
            self.hash = get_hash(infile)

        self._ds_type = DatasetType(self._ds)
        self._assign_simulation_attributes()

    @property
    def _has_galaxies(self):
        """ ngalaxies gets assigned during fubar() """
        if hasattr(self,'ngalaxies'):
            return True
        else:
            return False

    def _load_data(self):
        from caesar.data_manager import DataManager
        self.data_manager = DataManager(self)
        
    def _assign_simulation_attributes(self):
        self.simulation.create_attributes(self)

    def _assign_objects(self):
        import caesar.assignment as assign
        assign.assign_galaxies_to_halos(self)
        assign.assign_central_galaxies(self)
        
    def _link_objects(self):
        import caesar.linking as link
        link.link_galaxies_and_halos(self)
        link.create_sublists(self)

    def save(self, filename):
        from caesar.saver import save
        save(self, filename)
    
    def member_search(self, *args, **kwargs):
        self._args   = args
        self._kwargs = kwargs

        self._load_data()
        
        from caesar.fubar import fubar
        fubar(self, 'halo')
        fubar(self, 'galaxy')

        import caesar.assignment as assign
        import caesar.linking as link
        assign.assign_galaxies_to_halos(self)
        link.link_galaxies_and_halos(self)
        assign.assign_central_galaxies(self)
        link.create_sublists(self)
        
        import caesar.hydrogen_mass_calc as mass_calc
        mass_calc.hydrogen_mass_calc(self)
