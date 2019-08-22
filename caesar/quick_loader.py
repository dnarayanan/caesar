import os.path
import functools
from pprint import pprint

import h5py
from yt.units.yt_array import YTArray, UnitRegistry
from yt.funcs import mylog

from caesar.utils import info_printer
from caesar.simulation_attributes import SimulationAttributes
from caesar.group import info_blacklist


class CAESAR:
    def __init__(self, filename):
        self.data_file = os.path.abspath(filename)
        with h5py.File(filename, 'r') as hd:
            mylog.info('Reading {}'.format(filename))

            self.unit_registry = UnitRegistry.from_json(
                hd.attrs['unit_registry_json'].decode('utf8'))

            # Load the information about the simulation itself
            self.simulation = SimulationAttributes()
            self.simulation._unpack(self, hd)

            # Load the particle index lists; this is the most expensive stage by a lot
            mylog.info('Loading global lists')
            self.halo_dmlist = hd['halo_data/lists/dmlist'][:]
            self.halo_slist = hd['halo_data/lists/slist'][:]
            self.halo_glist = hd['halo_data/lists/glist'][:]

            if 'bhlist' in hd['halo_data/lists']:
                self.halo_bhlist = hd['halo_data/lists/bhlist'][:]
            else:
                self.halo_bhlist = None

            self.galaxy_slist = hd['galaxy_data/lists/slist'][:]
            self.galaxy_glist = hd['galaxy_data/lists/glist'][:]

            if 'bhlist' in hd['galaxy_data/lists']:
                self.galaxy_bhlist = hd['galaxy_data/lists/bhlist'][:]
            else:
                self.galaxy_bhlist = None

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

            self.halo_dicts = {}
            for k, v in hd['halo_data/dicts'].items():
                if 'unit' in v.attrs:
                    dictname, arrname = k.split('.')
                    if dictname not in self.halo_dicts:
                        self.halo_dicts[dictname] = {}
                    self.halo_dicts[dictname][arrname] = YTArray(
                        v[:], v.attrs['unit'], registry=self.unit_registry)
                else:
                    self.halo_dicts[dictname][arrname] = v[:]

            self.halos = [Halo(self, i) for i in range(hd.attrs['nhalos'])]
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

            self.galaxy_dicts = {}
            for k, v in hd['galaxy_data/dicts'].items():
                if 'unit' in v.attrs:
                    dictname, arrname = k.split('.')
                    if dictname not in self.galaxy_dicts:
                        self.galaxy_dicts[dictname] = {}
                    self.galaxy_dicts[dictname][arrname] = YTArray(
                        v[:], v.attrs['unit'], registry=self.unit_registry)
                else:
                    self.galaxy_dicts[dictname][arrname] = v[:]

            self.galaxies = [
                Galaxy(self, i) for i in range(hd.attrs['ngalaxies'])
            ]
            mylog.info('Loaded {} galaxies'.format(len(self.galaxies)))


    def galinfo(self, top=10):
        info_printer(self, 'galaxy', top)

    def haloinfo(self, top=10):
        info_printer(self, 'halo', top)


class Halo:
    __slots__ = ['obj', '_index', '_galaxies', '_satellite_galaxies', '_central_galaxy']

    def __init__(self, obj, index):
        self.obj = obj
        self._index = index
        self._galaxies = None
        self._satellite_galaxies = None
        self._central_galaxy = None

    def __dir__(self):
        items = list(self.obj.halo_data) + list(
            self.obj.halo_dicts) + ['glist', 'slist', 'dmlist']
        if self.obj.halo_bhlist is not None:
            items.append('bhlist')
        return items

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
        if self.obj.galaxy_bhlist is not None:
            return self.obj.galaxy_bhlist[self.bhlist_start:self.bhlist_end]
        else:
            return None

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


class Galaxy:
    __slots__ = ['obj', '_index', 'halo']

    def __init__(self, obj, index):
        self.obj = obj
        self._index = index
        self.halo = obj.halos[self.parent_halo_index]

    def __dir__(self):
        return list(self.obj.galaxy_data) + list(
            self.obj.galaxy_dicts) + ['glist', 'slist']
        if self.obj.galaxy_bhlist is not None:
            items.append('bhlist')
        return items

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
        else:
            return None

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

    def info(self):
        pdict = {}
        for k in dir(self):
            if k not in info_blacklist:
                pdict[k] = getattr(self, k)
        pprint(pdict)


def quick_load(filename):
    return CAESAR(filename)
