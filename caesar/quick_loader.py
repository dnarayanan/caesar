# TODO:
# Implement vtk_vis?
# Implement write_IC_mask?

import os.path
import functools
from pprint import pprint
from collections import defaultdict

import h5py
import numpy as np
from yt.units.yt_array import YTArray, UnitRegistry
from yt.funcs import mylog, get_hash

from caesar.property_manager import DatasetType
from caesar.utils import info_printer
from caesar.simulation_attributes import SimulationAttributes
from caesar.group import info_blacklist


class LazyProperty:
    '''An @property with name `name` which wraps an HDF5 object at h5_path

    On first access this will open the HDF5 file using the path stored in
    self.data_file and stores its contents in an instance variable.
    Subsequent accesses will return the instance variable.
    '''
    def __init__(self, h5_path, name):
        self.name = name
        self.inner_name = '_' + self.name
        self.h5_path = h5_path

    def __get__(self, instance, owner):
        if not hasattr(instance, self.inner_name):
            with h5py.File(instance.data_file, 'r') as hd:
                if self.h5_path in hd:
                    setattr(instance, self.inner_name, hd[self.h5_path][:])
                else:
                    setattr(instance, self.inner_name, None)
        return getattr(instance, self.inner_name)

    def __set__(self, instance, value):
        pass


class LazyList:
    '''A list-like container which laziliy creates its elements

    Wraps a list which is intially filled with None, this list of None is
    very fast to create at any size because None is a singleton. The
    actual elements of the list are created as-needed by calling the
    passed-in callable.
    '''
    def __init__(self, length, builder):
        self._elements = [None] * length
        self._builder = builder

    def __len__(self):
        return len(self._elements)

    def __getitem__(self, index):
        if self._elements[index] is None:
            self._elements[index] = self._builder(index)
        return self._elements[index]


class CAESAR:
    halo_dmlist = LazyProperty('halo_data/lists/dmlist', 'halo_dmlist')
    halo_slist = LazyProperty('halo_data/lists/slist', 'halo_slist')
    halo_glist = LazyProperty('halo_data/lists/glist', 'halo_glist')
    halo_bhlist = LazyProperty('halo_data/lists/bhlist', 'halo_bhlist')
    halo_dlist = LazyProperty('halo_data/lists/dlist', 'halo_dlist')

    galaxy_dmlist = LazyProperty('galaxy_data/lists/dmlist', 'galaxy_dmlist')
    galaxy_slist = LazyProperty('galaxy_data/lists/slist', 'galaxy_slist')
    galaxy_glist = LazyProperty('galaxy_data/lists/glist', 'galaxy_glist')
    galaxy_bhlist = LazyProperty('galaxy_data/lists/bhlist', 'galaxy_bhlist')
    galaxy_dlist = LazyProperty('galaxy_data/lists/dlist', 'galaxy_dlist')

    def __init__(self, filename):
        self._ds = None
        self.data_file = os.path.abspath(filename)
        with h5py.File(filename, 'r') as hd:
            mylog.info('Reading {}'.format(filename))

            self.hash = hd.attrs['hash']

            self.unit_registry = UnitRegistry.from_json(
                hd.attrs['unit_registry_json'].decode('utf8'))

            # Load the information about the simulation itself
            self.simulation = SimulationAttributes()
            self.simulation._unpack(self, hd)

            self.galaxy_index_list = hd['halo_data/lists/galaxy_index_list'][:]

            mylog.info('Loading halos')
            self.halo_data = {}
            for k, v in hd['halo_data'].items():
                if type(v) is h5py.Dataset:
                    if 'unit' in v.attrs:
                        self.halo_data[k] = YTArray(
                            v[:], v.attrs['unit'], registry=self.unit_registry)
                    else:
                        self.halo_data[k] = v[:]

            self.halo_dicts = defaultdict(dict)
            for k, v in hd['halo_data/dicts'].items():
                dictname, arrname = k.split('.')
                if 'unit' in v.attrs:
                    self.halo_dicts[dictname][arrname] = YTArray(
                        v[:], v.attrs['unit'], registry=self.unit_registry)
                else:
                    self.halo_dicts[dictname][arrname] = v[:]

            self.halos = LazyList(hd.attrs['nhalos'], lambda i: Halo(self, i))
            mylog.info('Loaded {} halos'.format(len(self.halos)))

            mylog.info('Loading galaxies')
            self.galaxy_data = {}
            for k, v in hd['galaxy_data'].items():
                if type(v) is h5py.Dataset:
                    if 'unit' in v.attrs:
                        self.galaxy_data[k] = YTArray(
                            v[:], v.attrs['unit'], registry=self.unit_registry)
                    else:
                        self.galaxy_data[k] = v[:]

            self.galaxy_dicts = defaultdict(dict)
            for k, v in hd['galaxy_data/dicts'].items():
                dictname, arrname = k.split('.')
                if 'unit' in v.attrs:
                    self.galaxy_dicts[dictname][arrname] = YTArray(
                        v[:], v.attrs['unit'], registry=self.unit_registry)
                else:
                    self.galaxy_dicts[dictname][arrname] = v[:]

            self.galaxies = LazyList(hd.attrs['ngalaxies'],
                                     lambda i: Galaxy(self, i))
            mylog.info('Loaded {} galaxies'.format(len(self.galaxies)))

    @property
    def yt_dataset(self):
        """The yt dataset to perform actions on."""
        if self._ds is None:
            raise Exception('No yt_dataset assigned!\nPlease assign '
                            'one via `obj.yt_dataset=<YT DATASET>` '
                            'to load particle/field data from disk.')
        return self._ds

    @yt_dataset.setter
    def yt_dataset(self, value):
        if value is None:
            return

        if not hasattr(value, 'dataset_type'):
            raise IOError('not a yt dataset?')

        infile = '%s/%s' % (value.fullpath, value.basename)

        if isinstance(self.hash, bytes):
            self.hash = self.hash.decode('utf8')

        hash = get_hash(infile)
        if hash != self.hash:
            raise IOError('hash mismatch!')
        else:
            self._ds = value

        self._ds = value
        self._ds_type = DatasetType(self._ds)

    def galinfo(self, top=10):
        info_printer(self, 'galaxy', top)

    def haloinfo(self, top=10):
        info_printer(self, 'halo', top)


class Group:
    def write_IC_mask(self,
                      ic_ds,
                      filename,
                      search_factor=2.5,
                      radius_type='total'):
        from caesar.zoom_funcs import write_IC_mask
        write_IC_mask(self,
                      ic_ds,
                      filename,
                      search_factor,
                      radius_type=radius_type)

    def info(self):
        pdict = {}
        for k in dir(self):
            if k not in info_blacklist:
                pdict[k] = getattr(self, k)
        pprint(pdict)

    def contamination_check(self,
                            lowres=[2, 3, 5],
                            search_factor=2.5,
                            printer=True):
        from caesar.zoom_funcs import construct_lowres_tree

        construct_lowres_tree(self, lowres)

        if self.obj_type == 'halo':
            halo = self
            ID = 'Halo %d' % self.GroupID
        elif self.obj_type == 'galaxy':
            if self.halo is None:
                raise Exception('Galaxy %d has no halo!' % self.GroupID)
            halo = self.halo
            ID = "Galaxy %d's halo (ID %d)" % (self.GroupID, halo.GroupID)

        r = halo.radii['virial'].d * search_factor

        result = self.obj._lowres['TREE'].query_ball_point(halo.pos.d, r)
        ncontam = len(result)
        lrmass = np.sum(self.obj._lowres['MASS'][result])

        self.contamination = lrmass / halo.masses['total'].d

        if not printer:
            return

        if ncontam > 0:
            mylog.warning('%s has %0.2f%% mass contamination '
                          '(%d LR particles with %0.2e % s)' %
                          (ID, self.contamination * 100.0, ncontam,
                           lrmass, halo.masses['total'].units))
        else:
            mylog.info('%s has NO contamination!' % ID)


class Halo(Group):
    def __init__(self, obj, index):
        self.obj_type = 'halo'
        self.obj = obj
        self._index = index
        self._galaxies = None
        self._satellite_galaxies = None
        self._central_galaxy = None
        self.metallicity = self.metallicities['mass_weighted']
        self.mass = self.masses['total']
        self.radius = self.radii['total']
        self.temperature = self.temperatures['mass_weighted']
        self.sigma = self.velocity_dispersions['gas']

    def __dir__(self):
        items = [
            'obj', 'obj_type', 'metallicity', 'mass', 'radius', 'temperature',
            'sigma', 'galaxies', 'central_galaxy', 'satellite_galaxies'
        ]
        items += list(self.obj.halo_data) + list(
            self.obj.halo_dicts) + ['glist', 'slist', 'dmlist']
        if self.obj.halo_bhlist is not None:
            items.append('bhlist')
        if self.obj.halo_dlist is not None:
            items.append('dlist')
        return [i for i in items if i not in info_blacklist]

    @property
    def glist(self):
        return self.obj.halo_glist[self.glist_start:self.glist_end]

    @property
    def slist(self):
        return self.obj.halo_slist[self.slist_start:self.slist_end]

    @property
    def dmlist(self):
        return self.obj.halo_dmlist[self.dmlist_start:self.dmlist_end]

    @property
    def bhlist(self):
        if self.obj.halo_bhlist is not None:
            return self.obj.halo_bhlist[self.bhlist_start:self.bhlist_end]

    @property
    def dlist(self):
        if self.obj.halo_dlist is not None:
            return self.obj.halo_dlist[self.dlist_start:self.dlist_end]

    @property
    def galaxy_index_list(self):
        return self.obj.galaxy_index_list[self.galaxy_index_list_start:self.
                                          galaxy_index_list_end]

    def _init_galaxies(self):
        self._galaxies = []
        self._satellite_galaxies = []
        for galaxy_index in self.galaxy_index_list:
            galaxy = self.obj.galaxies[galaxy_index]
            self._galaxies.append(galaxy)
            if galaxy.central:
                self._central_galaxy = galaxy
            else:
                self._satellite_galaxies.append(galaxy)

    @property
    def galaxies(self):
        if self._galaxies is None:
            self._init_galaxies()
        return self._galaxies

    @property
    def central_galaxy(self):
        if self._central_galaxy is None:
            self._init_galaxies()
        return self._central_galaxy

    @property
    def satellite_galaxies(self):
        if self._satellite_galaxies is None:
            self._init_galaxies()
        return self._satellite_galaxies

    @functools.lru_cache(maxsize=None)
    def __getattr__(self, attr):
        if attr in self.obj.halo_data:
            return self.obj.halo_data[attr][self._index]
        if attr in self.obj.halo_dicts:
            out = {}
            for d in self.obj.halo_dicts[attr]:
                out[d] = self.obj.halo_dicts[attr][d][self._index]
            return out
        raise AttributeError("'{}' object as no attribute '{}'".format(
            self.__class__.__name__, attr))

    def info(self):
        pdict = {}
        for k in dir(self):
            if k not in info_blacklist:
                pdict[k] = getattr(self, k)
        pprint(pdict)


class Galaxy(Group):
    def __init__(self, obj, index):
        self.obj_type = 'galaxy'
        self.obj = obj
        self._index = index
        self.halo = obj.halos[self.parent_halo_index]
        self.metallicity = self.metallicities['mass_weighted']
        self.mass = self.masses['total']
        self.radius = self.radii['total']
        self.temperature = self.temperatures['mass_weighted']
        self.sigma = self.velocity_dispersions['gas']

    def __dir__(self):
        items = [
            'obj',
            'obj_type',
            'halo',
            'metallicity',
            'mass',
            'radius',
            'temperature',
            'sigma',
            'info',
        ]
        items += list(self.obj.galaxy_data) + list(
            self.obj.galaxy_dicts) + ['glist', 'slist']
        if hasattr(self.obj, '_galaxy_bhlist'):
            items.append('bhlist')
        if hasattr(self.obj, '_galaxy_dlist'):
            items.append('dlist')
        return [i for i in items if i not in info_blacklist]

    @property
    def glist(self):
        return self.obj.galaxy_glist[self.glist_start:self.glist_end]

    @property
    def slist(self):
        return self.obj.galaxy_slist[self.slist_start:self.slist_end]

    @property
    def bhlist(self):
        if self.obj.galaxy_bhlist is not None:
            return self.obj.galaxy_bhlist[self.bhlist_start:self.bhlist_end]

    @property
    def dlist(self):
        if self.obj.galaxy_dlist is not None:
            return self.obj.galaxy_dlist[self.dlist_start:self.dlist_end]

    @property
    def satellites(self):
        if self.central:
            return self.halo.satellite_galaxies
        return []

    @functools.lru_cache(maxsize=None)
    def __getattr__(self, attr):
        if attr in self.obj.galaxy_data:
            return self.obj.galaxy_data[attr][self._index]
        if attr in self.obj.galaxy_dicts:
            out = {}
            for d in self.obj.galaxy_dicts[attr]:
                out[d] = self.obj.galaxy_dicts[attr][d][self._index]
            return out
        raise AttributeError("'{}' object as no attribute '{}'".format(
            self.__class__.__name__, attr))


def quick_load(filename):
    return CAESAR(filename)
