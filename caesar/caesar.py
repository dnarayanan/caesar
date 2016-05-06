from .property_getter import DatasetType
from .particle_list import ParticleListContainer

from yt.funcs import mylog, get_hash

class CAESAR(object):

    def __init__(self, ds=0, *args, **kwargs):
        self.args   = args
        self.kwargs = kwargs
        self._ds    = 0

        self.yt_dataset = ds
        self.global_particle_lists = ParticleListContainer(self)

        self.units = dict(
            mass='Msun',
            length='kpccm',
            velocity='km/s',
            time='year',
        )

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


    @property
    def has_galaxies(self):
        """ ngalaxies gets assigned during fubar() """
        if hasattr(self,'ngalaxies'):
            return True
        else:
            return False

    def _assign_objects(self):
        import assignment as assign
        assign.assign_galaxies_to_halos(self)
        assign.assign_central_galaxies(self)
        
    def _link_objects(self):
        import linking as link
        link.link_galaxies_and_halos(self)
        link.create_sublists(self)

    def save(self, filename):
        from saver import save
        save(self)
    
    def member_search(self):
        from .fubar import fubar
        fubar(self, 'halo')
        fubar(self, 'galaxy')

        import assignment as assign
        import linking as link
        assign.assign_galaxies_to_halos(self)
        link.link_galaxies_and_halos(self)
        assign.assign_central_galaxies(self)
        link.create_sublists(self)

        self.save('test.hdf5')
        
        import ipdb; ipdb.set_trace()
