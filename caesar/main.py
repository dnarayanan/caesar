from yt.funcs import mylog, get_hash
from .property_getter import DatasetType

class CAESAR(object):

    def __init__(self, ds=0, *args, **kwargs):
        self.args   = args
        self.kwargs = kwargs
        self._ds    = 0

        self.yt_dataset = ds

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
                raise IOError('has mismatch!')
            else:
                self._ds = value
        else:
            self._ds  = value
            self.hash = get_hash(infile)

        self._ds_type = DatasetType(self._ds)
        


