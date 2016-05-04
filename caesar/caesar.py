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
        

    def member_search(self):
        from .fubar import fubar
        fubar(self, 'halo')
        fubar(self, 'galaxy')
