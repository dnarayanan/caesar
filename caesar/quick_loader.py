"""This module is a lazy replacement for caesar.loader

Instead of eagerly constructing every Halo, Galaxy, and Cloud, this module
provides a class which lazily constructs Groups and their attributes only
as they are accessed, and caches them using functools.lru_cache.
"""

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


class LazyArray:
    """A lazily-loaded HDF5 dataset"""
    def __init__(self, obj, dataset_path):
        self._obj = obj
        self._dataset_path = dataset_path
        self._data = None

    def __getitem__(self, index):
        if self._data is None:
            with h5py.File(self._obj.data_file, 'r') as hd:
                dataset = hd[self._dataset_path]
                if 'units' in dataset.attrs:
                    self._data = YTArray(dataset[:], dataset.attrs['units'], registry=self.obj.unit_registry)
                else:
                    self._data = dataset[:]
        return self._data.__getitem__(index)


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
    def __init__(self, filename):
        self._ds = None
        self.data_file = os.path.abspath(filename)

        self._halo_dmlist = LazyArray(self, 'halo_data/lists/dmlist')
        self._halo_slist = LazyArray(self, 'halo_data/lists/slist')
        self._halo_glist = LazyArray(self, 'halo_data/lists/glist')
        self._halo_bhlist = LazyArray(self, 'halo_data/lists/bhlist')
        self._halo_dlist = LazyArray(self, 'halo_data/lists/dlist')

        self._galaxy_slist = LazyArray(self, 'galaxy_data/lists/slist')
        self._galaxy_glist = LazyArray(self, 'galaxy_data/lists/glist')
        self._galaxy_bhlist = LazyArray(self, 'galaxy_data/lists/bhlist')
        self._galaxy_dlist = LazyArray(self, 'galaxy_data/lists/dlist')

        self._cloud_glist = LazyArray(self, 'cloud_data/lists/glist')
        self._cloud_dlist = LazyArray(self, 'cloud_data/lists/dlist')


        with h5py.File(filename, 'r') as hd:
            mylog.info('Reading {}'.format(filename))

            self.hash = hd.attrs['hash']
            self.caesar = hd.attrs['caesar']

            self.unit_registry = UnitRegistry.from_json(
                hd.attrs['unit_registry_json'].decode('utf8'))

            # Load the information about the simulation itself
            self.simulation = SimulationAttributes()
            self.simulation._unpack(self, hd)

            mylog.info('Loading halos')
            self._galaxy_index_list = None
            if 'halo_data/lists/galaxy_index_list' in hd:
                self._galaxy_index_list = LazyArray(self, 'halo_data/lists/galaxy_index_list')

            self._halo_data = {}
            for k, v in hd['halo_data'].items():
                if type(v) is h5py.Dataset:
                    self._halo_data[k] = LazyArray(self, 'halo_data/'+k)

            self._halo_dicts = defaultdict(dict)
            for k in hd['halo_data/dicts']:
                dictname, arrname = k.split('.')
                self._halo_dicts[dictname][arrname] = LazyArray(self, 'halo_data/dicts/'+k)

            self.nhalos = hd.attrs['nhalos']
            self.halos = LazyList(self.nhalos, lambda i: Halo(self, i))
            mylog.info('Loaded {} halos'.format(len(self.halos)))

            self._galaxy_data = {}
            self._galaxy_dicts = defaultdict(dict)
            self.ngalaxies = 0
            self.galaxies = LazyList(self.ngalaxies, lambda i: Galaxy(self, i))
            if 'galaxy_data' in hd:
                mylog.info('Loading galaxies')
                for k, v in hd['galaxy_data'].items():
                    if type(v) is h5py.Dataset:
                        self._galaxy_data[k] = LazyArray(self, 'galaxy_data/'+k)

                for k in hd['galaxy_data/dicts']:
                    dictname, arrname = k.split('.')
                    self._galaxy_dicts[dictname][arrname] = LazyArray(self, 'galaxy_data/dicts/'+k)

                self.ngalaxies = hd.attrs['ngalaxies']
                self.galaxies = LazyList(self.ngalaxies, lambda i: Galaxy(self, i))
                mylog.info('Loaded {} galaxies'.format(len(self.galaxies)))

            self._cloud_data = {}
            self._cloud_dicts = defaultdict(dict)
            self.nclouds = 0
            self.clouds = LazyList(self.nclouds, lambda i: Cloud(self, i))
            if 'cloud_data' in hd:
                mylog.info('Loading clouds')
                for k, v in hd['cloud_data'].items():
                    if type(v) is h5py.Dataset:
                        self._cloud_data[k] = LazyArray(self, 'cloud_data/'+k)

                for k in hd['cloud_data/dicts']:
                    dictname, arrname = k.split('.')
                    self._cloud_dicts[dictname][arrname] = LazyArray(self, 'cloud_data/dicts/'+k)

                self.nclouds = hd.attrs['nclouds']
                self.clouds = LazyList(self.nclouds, lambda i: Cloud(self, i))
                mylog.info('Loaded {} clouds'.format(len(self.clouds)))

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

        hash = get_hash(infile)
        if hash != self.hash:
            raise IOError('hash mismatch!')
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
                          (ID, self.contamination * 100.0, ncontam, lrmass,
                           halo.masses['total'].units))
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
        if self.obj.halo_bhlist is not None:
            return self.obj._halo_bhlist[self.bhlist_start:self.bhlist_end]

    @property
    def dlist(self):
        if self.obj.halo_dlist is not None:
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
            out = {}
            for d in self.obj._halo_dicts[attr]:
                out[d] = self.obj._halo_dicts[attr][d][self._index]
            return out
        raise AttributeError("'{}' object as no attribute '{}'".format(
            self.__class__.__name__, attr))


class Galaxy(Group):
    def __init__(self, obj, index):
        self.obj_type = 'galaxy'
        self.obj = obj
        self._index = index
        self.halo = obj.halos[self.parent_halo_index]

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

    @functools.lru_cache(maxsize=None)
    def __getattr__(self, attr):
        if attr in self.obj._galaxy_data:
            return self.obj._galaxy_data[attr][self._index]
        if attr in self.obj._galaxy_dicts:
            out = {}
            for d in self.obj._galaxy_dicts[attr]:
                out[d] = self.obj._galaxy_dicts[attr][d][self._index]
            return out
        raise AttributeError("'{}' object as no attribute '{}'".format(
            self.__class__.__name__, attr))


class Cloud(Group):
    def __init__(self, obj, index):
        self.obj_type = 'cloud'
        self.obj = obj
        self._index = index

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
            out = {}
            for d in self.obj._cloud_dicts[attr]:
                out[d] = self.obj._cloud_dicts[attr][d][self._index]
            return out
        raise AttributeError("'{}' object as no attribute '{}'".format(
            self.__class__.__name__, attr))


def quick_load(filename):
    return CAESAR(filename)
