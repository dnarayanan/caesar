"""This module is a lazy replacement for caesar.loader

Instead of eagerly constructing every Halo, Galaxy, and Cloud, this module
provides a class which lazily constructs Groups and their attributes only
as they are accessed, and caches them using functools.lru_cache.

The design of this module was motivated by profiling the previous eager loader,
which revealed these things dominated load time, in order of importance:
1) Creating unyt.unty_quantity objects
2) Creating Halo/Galaxy/Cloud objects
3) Reading datasets from the HDF5 file
Therefore, this module avoids creating quantities as much as possible and
caches them. It might be nice to only load part of the backing HDF5 datasets,
but that stage is already quite fast and it looks to me like the HDF5 library
(or at least h5py) has some minimum granularity at which it will pull data off
disk which is ~1M items, which at the time of writing (April 21, 2020) exceeds
the size of most datasets in caesar files, including from the m100n1024 SIMBA
run I've been testing with.
"""

import os.path
import functools
from pprint import pprint
from collections import defaultdict
from collections.abc import Sequence, Mapping

import h5py
import numpy as np
from yt.units.yt_array import YTArray, UnitRegistry
from yt.funcs import mylog, get_hash

from caesar.property_manager import DatasetType
from caesar.utils import info_printer
from caesar.simulation_attributes import SimulationAttributes


class LazyDataset:
    """A lazily-loaded HDF5 dataset"""
    def __init__(self, obj, dataset_path):
        self._obj = obj
        self._dataset_path = dataset_path
        self._data = None

    def __getitem__(self, index):
        if self._data is None:
            with h5py.File(self._obj.data_file, 'r') as hd:
                dataset = hd[self._dataset_path]
                if 'unit' in dataset.attrs:
                    self._data = YTArray(dataset[:],
                                         dataset.attrs['unit'],
                                         registry=self._obj.unit_registry)
                else:
                    self._data = dataset[:]
        return self._data.__getitem__(index)


class LazyList(Sequence):
    """This type should be indistinguishable from the built-in list.
    Any observable difference except the explicit type and performance
    is considered a bug.

    The implementation wraps a list which is intially filled with None,
    which is very fast to create at any size because None is a singleton.
    The initial elements are replaced by calling the passed-in callable
    as they are accessed.
    """
    def __init__(self, length, builder):
        self._inner = [None] * length
        self._builder = builder

    def __contains__(self, value):
        for i in range(len(self)):
            if self[i] == value:
                return True
        return False

    def __getitem__(self, index):
        trial_output = self._inner[index]

        # Handle uninitialized elements for integer indices
        if trial_output is None:
            self._inner[index] = self._builder(index)

        # And for all other kinds of indices
        if isinstance(trial_output, list) and None in trial_output:
            for i in range(len(self))[index]:
                if self._inner[i] is None:
                    self._inner[i] = self._builder(i)

        return self._inner[index]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __reversed__(self):
        LazyList(len(self), lambda i: self._builder(len(self) - i))

    def count(self, value):
        return sum(i == value for i in self)

    def index(self, value, start=0, stop=None):
        if stop is None:
            stop = len(self)
        for i in range(start, stop):
            if self[i] == value:
                return i
        raise ValueError('{} is not in LazyList'.format(value))

    def __len__(self):
        return len(self._inner)


class LazyDict(Mapping):
    """This type should be indistinguishable from the built-in dict.
    Any observable difference except the explicit type and performance
    is considered a bug.

    The implementation wraps a dict which initially maps every key to None,
    and are replaced by calling the passed-in callable as they are accessed.
    """
    def __init__(self, keys, builder):
        self._inner = {k: None for k in keys}
        self._builder = builder

    def _init_all(self):
        """Internal use only, for operations that need all values"""
        for k in self._inner:
            self[k]

    def __contains__(self, key):
        return key in self._inner

    def __eq__(self, other):
        self._init_all()
        return self._inner == other

    def __getitem__(self, key):
        value = self._inner[key]
        if value is None:
            value = self._builder(key)
            self._inner[key] = value
        return value

    def get(self, key, default=None):
        if key in self._inner:
            return self[key]
        return default

    def items(self):
        for k in self._inner:
            yield (k, self[k])

    def keys(self):
        return self._inner.keys()

    def values(self):
        for k in self._inner:
            yield self[k]

    def __len__(self):
        return len(self._inner)

    def __iter__(self):
        return iter(self._inner)

    def __str__(self):
        return str(dict(self))

    def __repr__(self):
        return repr(dict(self))

    def _repr_pretty_(self, p, cycle):
        p.pretty(dict(self))


class CAESAR:
    def __init__(self, filename):
        self._ds = None
        self.data_file = os.path.abspath(filename)

        self._halo_dmlist = LazyDataset(self, 'halo_data/lists/dmlist')
        self._halo_slist = LazyDataset(self, 'halo_data/lists/slist')
        self._halo_glist = LazyDataset(self, 'halo_data/lists/glist')
        self._halo_bhlist = LazyDataset(self, 'halo_data/lists/bhlist')
        self._halo_dlist = LazyDataset(self, 'halo_data/lists/dlist')

        self._galaxy_slist = LazyDataset(self, 'galaxy_data/lists/slist')
        self._galaxy_glist = LazyDataset(self, 'galaxy_data/lists/glist')
        self._galaxy_bhlist = LazyDataset(self, 'galaxy_data/lists/bhlist')
        self._galaxy_dlist = LazyDataset(self, 'galaxy_data/lists/dlist')

        self._cloud_glist = LazyDataset(self, 'cloud_data/lists/glist')
        self._cloud_dlist = LazyDataset(self, 'cloud_data/lists/dlist')

        with h5py.File(filename, 'r') as hd:
            mylog.info('Opening {}'.format(filename))

            self.hash = hd.attrs['hash']
            if isinstance(self.hash, np.bytes_):
                self.hash = self.hash.decode('utf8')

            # This should probably be caesar_version or something
            self.caesar = hd.attrs['caesar']

            self.unit_registry = UnitRegistry.from_json(
                hd.attrs['unit_registry_json'].decode('utf8'))

            # Load the information about the simulation itself
            self.simulation = SimulationAttributes()
            self.simulation._unpack(self, hd)

            # Halo data is loaded unconditionally, AFAICT it's always present
            self._galaxy_index_list = None
            if 'halo_data/lists/galaxy_index_list' in hd:
                self._galaxy_index_list = LazyDataset(
                    self, 'halo_data/lists/galaxy_index_list')

            self._halo_data = {}
            for k, v in hd['halo_data'].items():
                if type(v) is h5py.Dataset:
                    self._halo_data[k] = LazyDataset(self, 'halo_data/' + k)

            self._halo_dicts = defaultdict(dict)
            for k in hd['halo_data/dicts']:
                dictname, arrname = k.split('.')
                self._halo_dicts[dictname][arrname] = LazyDataset(
                    self, 'halo_data/dicts/' + k)

            self.nhalos = hd.attrs['nhalos']
            self.halos = LazyList(self.nhalos, lambda i: Halo(self, i))
            mylog.info('Found {} halos'.format(len(self.halos)))

            # Provide default values for everything, so that if a simulation
            # without galaxies is loaded we get zero galaxies, not AttributeErrors
            self._galaxy_data = {}
            self._galaxy_dicts = defaultdict(dict)
            self.ngalaxies = 0
            self.galaxies = LazyList(self.ngalaxies, lambda i: Galaxy(self, i))
            if 'galaxy_data' in hd:
                self._cloud_index_list = None
                if 'galaxy_data/lists/cloud_index_list' in hd:
                    self._cloud_index_list = LazyDataset(
                        self, 'galaxy_data/lists/cloud_index_list')

                if 'tree_data/progen_galaxy_star' in hd:
                    self._galaxy_data['progen_galaxy_star'] = self._progen_galaxy_star = LazyDataset(
                        self, 'tree_data/progen_galaxy_star')

                for k, v in hd['galaxy_data'].items():
                    if type(v) is h5py.Dataset:
                        self._galaxy_data[k] = LazyDataset(
                            self, 'galaxy_data/' + k)

                for k in hd['galaxy_data/dicts']:
                    dictname, arrname = k.split('.')
                    self._galaxy_dicts[dictname][arrname] = LazyDataset(
                        self, 'galaxy_data/dicts/' + k)

                self.ngalaxies = hd.attrs['ngalaxies']
                self.galaxies = LazyList(self.ngalaxies,
                                         lambda i: Galaxy(self, i))
                mylog.info('Found {} galaxies'.format(len(self.galaxies)))

            self._cloud_data = {}
            self._cloud_dicts = defaultdict(dict)
            self.nclouds = 0
            self.clouds = LazyList(self.nclouds, lambda i: Cloud(self, i))
            if 'cloud_data' in hd:
                for k, v in hd['cloud_data'].items():
                    if type(v) is h5py.Dataset:
                        self._cloud_data[k] = LazyDataset(
                            self, 'cloud_data/' + k)

                for k in hd['cloud_data/dicts']:
                    dictname, arrname = k.split('.')
                    self._cloud_dicts[dictname][arrname] = LazyDataset(
                        self, 'cloud_data/dicts/' + k)

                self.nclouds = hd.attrs['nclouds']
                self.clouds = LazyList(self.nclouds, lambda i: Cloud(self, i))
                mylog.info('Found {} clouds'.format(len(self.clouds)))

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
            raise ValueError('not a yt dataset?')

        hash = get_hash(os.path.join(value.fullpath, value.basename))
        if hash != self.hash:
            raise RuntimeError('hash mismatch!')
        else:
            self._ds = value

        self._ds = value
        self._ds_type = DatasetType(self._ds)

    @property
    def central_galaxies(self):
        return [h.central_galaxy for h in self.halos]

    @property
    def satellite_galaxies(self):
        galaxies = []
        for h in self.halos:
            galaxies.extend(h.satellite_galaxies)

    def galinfo(self, top=10):
        info_printer(self, 'galaxy', top)

    def haloinfo(self, top=10):
        info_printer(self, 'halo', top)

    def cloudinfo(self, top=10):
        info_printer(self, 'cloud', top)


class Group:
    @property
    def mass(self):
        return self.masses['total']

    @property
    def metallicity(self):
        return self.metallicities['mass_weighted']

    @property
    def temperature(self):
        return self.temperatures['mass_weighted']

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
        for k in getattr(self.obj, '_{}_data'.format(self.obj_type)):
            pdict[k] = getattr(self, k)
        for k in getattr(self.obj, '_{}_dicts'.format(self.obj_type)):
            pdict[k] = dict(getattr(self, k))
        pprint(pdict)

    def contamination_check(self,
                            lowres=[2, 3, 5],
                            search_factor=2.5,
                            printer=True):
        from caesar.zoom_funcs import construct_lowres_tree

        construct_lowres_tree(self, lowres)

        if self.obj_type == 'halo':
            halo = self
            ID = 'Halo {}'.format(self.GroupID)
        elif self.obj_type == 'galaxy':
            if self.halo is None:
                raise Exception('Galaxy {} has no halo!'.format(self.GroupID))
            halo = self.halo
            ID = 'Galaxy {}\'s halo (ID {})'.format(self.GroupID, halo.GroupID)

        r = halo.radii['virial'].d * search_factor

        result = self.obj._lowres['TREE'].query_ball_point(halo.pos.d, r)
        ncontam = len(result)
        lrmass = np.sum(self.obj._lowres['MASS'][result])

        self.contamination = lrmass / halo.masses['total'].d

        if not printer:
            return

        if ncontam > 0:
            mylog.warning('{} has {0.2f}% mass contamination '
                          '({} LR particles with {0.2e} {}s)'.format(
                              ID, self.contamination * 100.0, ncontam, lrmass,
                              halo.masses['total'].units))
        else:
            mylog.info('{} has NO contamination!'.format(ID))


class Halo(Group):
    def __init__(self, obj, index):
        self.obj_type = 'halo'
        self.obj = obj
        self._index = index
        self._galaxies = None
        self._satellite_galaxies = None
        self._central_galaxy = None

    def __dir__(self):
        return dir(type(self)) + list(self.__dict__) + list(
            self.obj._halo_data) + list(self.obj._halo_dicts)

    @property
    def glist(self):
        return self.obj._halo_glist[self.glist_start:self.glist_end]

    @property
    def slist(self):
        return self.obj._halo_slist[self.slist_start:self.slist_end]

    @property
    def dmlist(self):
        return self.obj._halo_dmlist[self.dmlist_start:self.dmlist_end]

    @property
    def bhlist(self):
        if self.obj._halo_bhlist is not None:
            return self.obj._halo_bhlist[self.bhlist_start:self.bhlist_end]

    @property
    def dlist(self):
        if self.obj._halo_dlist is not None:
            return self.obj._halo_dlist[self.dlist_start:self.dlist_end]

    @property
    def galaxy_index_list(self):
        return self.obj._galaxy_index_list[self.galaxy_index_list_start:self.
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
        if attr in self.obj._halo_data:
            return self.obj._halo_data[attr][self._index]
        if attr in self.obj._halo_dicts:
            return LazyDict(
                self.obj._halo_dicts[attr].keys(),
                lambda d: self.obj._halo_dicts[attr][d][self._index])
        raise AttributeError("'{}' object as no attribute '{}'".format(
            self.__class__.__name__, attr))


class Galaxy(Group):
    def __init__(self, obj, index):
        self.obj_type = 'galaxy'
        self.obj = obj
        self._index = index
        self.halo = obj.halos[self.parent_halo_index]
        self._clouds = None

    def __dir__(self):
        return dir(type(self)) + list(self.__dict__) + list(
            self.obj._galaxy_data) + list(self.obj._galaxy_dicts)

    @property
    def glist(self):
        return self.obj._galaxy_glist[self.glist_start:self.glist_end]

    @property
    def slist(self):
        return self.obj._galaxy_slist[self.slist_start:self.slist_end]

    @property
    def bhlist(self):
        if self.obj._galaxy_bhlist is not None:
            return self.obj._galaxy_bhlist[self.bhlist_start:self.bhlist_end]

    @property
    def dlist(self):
        if self.obj._galaxy_dlist is not None:
            return self.obj._galaxy_dlist[self.dlist_start:self.dlist_end]

    @property
    def satellites(self):
        if self.central:
            return self.halo.satellite_galaxies
        return []

    @property
    def cloud_index_list(self):
        return self.obj._cloud_index_list[self.cloud_index_list_start:self.
                                          cloud_index_list_end]

    def _init_clouds(self):
        self._clouds = []
        for cloud_index in self.cloud_index_list:
            cloud = self.obj.clouds[cloud_index]
            self._clouds.append(cloud)

    @property
    def clouds(self):
        if self._clouds is None:
            self._init_clouds()
        return self._clouds

    @functools.lru_cache(maxsize=None)
    def __getattr__(self, attr):
        if attr in self.obj._galaxy_data:
            return self.obj._galaxy_data[attr][self._index]
        if attr in self.obj._galaxy_dicts:
            return LazyDict(
                self.obj._galaxy_dicts[attr].keys(),
                lambda d: self.obj._galaxy_dicts[attr][d][self._index])
        raise AttributeError("'{}' object as no attribute '{}'".format(
            self.__class__.__name__, attr))


class Cloud(Group):
    def __init__(self, obj, index):
        self.obj_type = 'cloud'
        self.obj = obj
        self._index = index
        self.galaxy = obj.galaxies[self.parent_galaxy_index]
        self.halo = self.galaxy.halo

    def __dir__(self):
        return dir(type(self)) + list(self.__dict__) + list(
            self.obj._cloud_data) + list(self.obj._cloud_dicts)

    @property
    def glist(self):
        return self.obj._cloud_glist[self.glist_start:self.glist_end]

    @property
    def dlist(self):
        if self.obj._cloud_dlist is not None:
            return self.obj._cloud_dlist[self.dlist_start:self.dlist_end]

    @functools.lru_cache(maxsize=None)
    def __getattr__(self, attr):
        if attr in self.obj._cloud_data:
            return self.obj._cloud_data[attr][self._index]
        if attr in self.obj._cloud_dicts:
            return LazyDict(
                self.obj._cloud_dicts[attr].keys(),
                lambda d: self.obj._cloud_dicts[attr][d][self._index])
        raise AttributeError("'{}' object as no attribute '{}'".format(
            self.__class__.__name__, attr))


def load(filename):
    return CAESAR(filename)
