

#=========================================================
# IMPORT STATEMENTS
#=========================================================

import caesar
from readgadget import *
import sys
import pylab as plt 
import os
os.environ["OMP_NUM_THREADS"] = "24"
import numpy as np
import fsps
import extinction
import h5py
from scipy import interpolate
from astropy import constants as const
from astropy.cosmology import FlatLambdaCDM
from caesar.utils import memlog
from caesar.property_manager import MY_DTYPE

#from pygas import *
#from auxloser import t_elapsed,parse_args,progress_bar,hubble_z

#from scipy.ndimage.filters import gaussian_filter

# start overall timer
CLIGHT_AA = const.c.to('AA/s').value
Solar = {'total':0.0134, 'He':0.2485, 'C':2.38e-3, 'N':0.70e-3, 'O':5.79e-3, 'Ne':1.26e-3, 'Mg':7.14e-4, 'Si':6.71e-4, 'S':3.12e-4, 'Ca':0.65e-4, 'Fe':1.31e-3} # Asplund abundances used in Simba


#=========================================================
# ROUTINES TO COMPUTE SPECTRA AND MAGNITUDES
#=========================================================

# photometry class
class photometry:

    def __init__(self, obj, group_list, ds=None, band_names='v', ssp_table_file='SSP_Chab_EL.hdf5', view_dir='x', use_dust=True, use_cosmic_ext=True, kernel_type='cubic', nproc=-1):

        from caesar.property_manager import ptype_ints
        self.obj = obj  # caesar object
        self.groups = group_list  # list of objects to process

        # optional arguments
        self.band_names = band_names
        if hasattr(self.obj,'_kwargs') and 'fsps_bands' in self.obj._kwargs:
            self.band_names = self.obj._kwargs['fsps_bands']
        self.ssp_table_file = os.path.expanduser('~/caesar/%s'%ssp_table_file)
        if hasattr(self.obj,'_kwargs') and 'ssp_table_file' in self.obj._kwargs:
            self.ssp_table_file = self.obj._kwargs['ssp_table_file']
        self.ext_law = 'mix_calz_mw'
        if hasattr(self.obj,'_kwargs') and 'ext_law' in self.obj._kwargs:
            self.ext_law = self.obj._kwargs['ext_law'].lower()
        if hasattr(self.obj,'_kwargs') and 'view_dir' in self.obj._kwargs:
            view_dir = self.obj._kwargs['view_dir'].lower()
        if view_dir is 'x': self.viewdir = 0
        if view_dir is 'y': self.viewdir = 1
        if view_dir is 'z': self.viewdir = 2
        self.use_dust = use_dust  # if False, will use metals plus an assumed dust-to-metal ratio
        if hasattr(self.obj,'_kwargs') and 'use_dust' in self.obj._kwargs:
            use_dust = self.obj._kwargs['use_dust'].lower()
        self.use_cosmic_ext = use_cosmic_ext
        if hasattr(self.obj,'_kwargs') and 'use_cosmic_ext' in self.obj._kwargs:
            use_cosmic_ext = self.obj._kwargs['use_cosmic_ext'].lower()
        self.kernel_type = kernel_type
        self.nkerntab = 2000
        if nproc == -1:
            try:
                self.nproc = obj.nproc  
            except:
                self.nproc = 1
        else:
            self.nproc = nproc

        # useful quantities
        self.boxsize = self.obj.simulation.boxsize
        self.solar_abund = Solar
        self.lumtoflux_abs = const.L_sun.to('erg/s').value/(4* np.pi * 10.**2 * const.pc.to('cm').value**2)
        cosmo = FlatLambdaCDM(H0=100.*self.obj.simulation.hubble_constant, Om0=self.obj.simulation.omega_matter, Tcmb0=2.73)  
        lumdist = cosmo.luminosity_distance(self.obj.simulation.redshift).to('pc').value
        self.lumtoflux = const.L_sun.to('erg/s').value/(4* np.pi * lumdist**2 * const.pc.to('cm').value**2)
        self.lumtoflux *= 1.+self.obj.simulation.redshift  # we compute apparent mags by blueshifting the band, which reduces the flux by (1+z); correct for this here

        # if there is no data_manager, assume we're running interactively
        # this means we have to load in the particle info, and set some other info we need
        if not hasattr(self.obj,'data_manager'):
            from caesar.data_manager import DataManager
            from caesar.property_manager import DatasetType
            self.obj.data_manager = DataManager(self.obj)
            self.obj._ds_type = DatasetType(ds)
            self.obj.yt_dataset = ds
            self.obj.units = dict(
                mass='Msun',
                length='kpccm',
                velocity='km/s',
                time='yr',
                temperature='K'
            )
            self.obj.data_manager._photometry_init()

    def run_pyloser(self):

        from caesar.cyloser import compute_AV, compute_mags

        self.init_pyloser()
        #computes AV for all stars in snapshot
        self.obj.AV_star = compute_AV(self)
        #find the AV for stars belonging to the groups that were asked for
        self.Av_per_group()
        spect_dust, spect_nodust = compute_mags(self)
        
        return spect_dust, spect_nodust


    def init_pyloser(self):
        from caesar.cyloser import init_kerntab
        self.init_ssp_table()
        self.init_extinction()
        self.init_bands()
        init_kerntab(self)
        self.init_stars_to_process()

    def Av_per_group(self):
        #separate AV_all_stars by group
        memlog('Finding LOS A_V values for %d objects'%(len(self.groups)))
        for obj_ in self.groups:
            current_id = obj_.GroupID
            start = np.sum([len(x.slist) for x in self.obj.galaxies[:current_id]])
            end = start + len(obj_.slist)
            print('[pyloser/Av_per_star]: Found object %d Av values'%(current_id))
            print('[pyloser/Av_per_star]: Starting index: %d\n[pyloser/Av_per_star]: Star count: %d'%(start, len(obj_.slist)))
            Av_per_star = self.obj.AV_star[start:end]
            obj_.group_Av = Av_per_star


    # initialize extinction curves. last one is cosmic IGM attenution from Madau
    def init_extinction(self):
        wave = self.ssp_wavelengths.astype(np.float64)
        self.ext_curves = []
        self.ext_curves.append(extinction.calzetti00(wave, 1.0, 4.05))  # atten_laws[0]
        self.ext_curves.append(extinction.fm07(wave, 1.0))  # atten_laws[1]
        self.ext_curves = np.asarray(self.ext_curves)

    # set up star and gas lists in each object
    def init_stars_to_process(self):
        from caesar.group import Group, collate_group_ids
        from caesar.property_manager import ptype_ints
        from caesar.cyloser import smass_at_formation

        #if isinstance(self.groups[0],Group):
        self.ngroup, self.gasids, self.gid_bins = collate_group_ids(self.groups,'gas',self.obj.simulation.ngas)
        self.ngroup, self.starids, self.sid_bins = collate_group_ids(self.groups,'star',self.obj.simulation.nstar)
        #else:
        #    sys.exit('Must provide a list of Caesar groups.')
        self.scount = sum([len(i.slist) for i in self.groups])
        self.gcount = sum([len(i.glist) for i in self.groups])
        self.Nobjs = len(self.groups)

        # get original stellar mass at time of formation
        self.obj.smass_orig = smass_at_formation(self.obj,self.groups,self.ssp_mass,self.ssp_ages,nproc=self.nproc)

        memlog('Loaded %d stars and %d gas in %d objects to process'%(self.scount,self.gcount,self.Nobjs))
        return

    # initialize band transmission data interpolated to FSPS wavelengths
    def init_bands(self):
        import fsps
        if isinstance(self.band_names,str):
            self.band_names = [self.band_names]
        if self.band_names[0] == 'all': 
            self.band_names = fsps.list_filters()
        elif self.band_names[0] == 'uvoir':
            self.band_names = []
            for ib,b in enumerate(fsps.list_filters()):
                band = fsps.filters.get_filter(b)  # look up characteristics of desired band 
                band_wave = band.transmission[0]   # filter wavelengths
                band_trans = band.transmission[1]  # filter response function
                meanwave = np.sum(band.transmission[0]*band.transmission[1])/np.sum(band.transmission[1])
                if meanwave < 50000: self.band_names.append(b)
        else:
            # collect all filters containing the input string(s)
            allfilters = fsps.list_filters()
            mybands = []
            for b in self.band_names:  # check that requested bands are actually available
                for b_all in allfilters:
                    if b in b_all: 
                        if b == b_all: mybands.append(b_all)  # if exact match, add
                        elif len(b)>3: mybands.append(b_all)  # avoid adding matching short band names (e.g. 'u')
            if len(mybands) == 0:
                assert b in allfilters, 'Band %s not found among available FSPS filters! Call fsps.list_filters() to list filters.'%self.band_names
            self.band_names = mybands
        # V band is always computed, so that one has A_V (= V_dust - V_nodust)
        if 'v' not in self.band_names:
            self.band_names.append('v')
  
        # Madau IGM attenuation is applied directly to rest-frame bandpasses only when computing apparent magnitudes; compute this curve here for specific redshift
        redshift = self.obj.simulation.redshift
        if self.use_cosmic_ext: 
            from synphot import etau_madau  # see synphot.readthedocs.io/en/latest/synphot/tutorials.html
            extcurve = etau_madau(self.ssp_wavelengths*(1.+redshift), redshift)
            cosmic_ext = extcurve(self.ssp_wavelengths)
        else: cosmic_ext = np.ones(len(self.ssp_wavelengths))

        # set up band information
        nbands = len(self.band_names)
        self.band_meanwave = np.zeros(nbands,dtype=MY_DTYPE)
        self.band_indexes = np.zeros(nbands+1,dtype=np.int32)
        self.band_ftrans = np.empty(0,dtype=MY_DTYPE)
        self.band_iwave0 = np.zeros(nbands,dtype=np.int32)
        self.band_iwave1 = np.zeros(nbands,dtype=np.int32)
        self.band_indz = np.zeros(nbands+1,dtype=np.int32)
        self.band_ztrans = np.empty(0,dtype=MY_DTYPE)
        self.band_iwz0 = np.zeros(nbands,dtype=np.int32)
        self.band_iwz1 = np.zeros(nbands,dtype=np.int32)
        for ib,b in enumerate(self.band_names):
            band = fsps.filters.get_filter(b)  # look up characteristics of desired band 
            band_wave = band.transmission[0]   # filter wavelengths
            band_trans = band.transmission[1]  # filter response function
            self.band_meanwave[ib] = np.sum(band.transmission[0]*band.transmission[1])/np.sum(band.transmission[1])
            # Set up transmission curve in region probed by rest-frame band
            ind = np.where((self.ssp_wavelengths > band_wave[0]) & (self.ssp_wavelengths < band_wave[-1]))[0] # indices of wavelengths in the band
            self.band_iwave0[ib] = ind[0]
            self.band_iwave1[ib] = ind[-1]+1
            ftrans = np.interp(self.ssp_wavelengths[ind],band_wave,band_trans)  # transmission at those wavelengths
            dnu = CLIGHT_AA/self.ssp_wavelengths[ind[0]:ind[-1]+1] - CLIGHT_AA/self.ssp_wavelengths[ind[0]+1:ind[-1]+2]  # convert to delta-nu
            self.band_ftrans = np.append(self.band_ftrans, ftrans*dnu)
            self.band_indexes[ib+1] = len(self.band_ftrans)
            # Now set up band for apparent mag computation
            # We will blueshift the band, corresponding to redshifting the intrinsic spectrum
            ind = np.where((self.ssp_wavelengths > band_wave[0]*self.obj.simulation.scale_factor) & (self.ssp_wavelengths < band_wave[-1]*self.obj.simulation.scale_factor))[0] # indices of wavelengths for redshifted rest-frame spectrum (i.e. blueshifted band)
            self.band_iwz0[ib] = ind[0]
            self.band_iwz1[ib] = ind[-1]+1
            ftrans = np.interp(self.ssp_wavelengths[ind],band_wave*self.obj.simulation.scale_factor,band_trans)  # transmission at those wavelengths
            dnu = CLIGHT_AA/self.ssp_wavelengths[ind[0]:ind[-1]+1] - CLIGHT_AA/self.ssp_wavelengths[ind[0]+1:ind[-1]+2]  # convert to delta-nu
            self.band_ztrans = np.append(self.band_ztrans, np.array(ftrans*dnu*cosmic_ext[ind]))
            self.band_indz[ib+1] = len(self.band_ztrans)

        memlog('Computing %d bands: %s'%(len(self.band_names),self.band_names))

    # initialize SSP table, by either generating it if it doesn't exist or reading it in
    def init_ssp_table(self):
        import os
        read_flag = True
        if os.path.exists(self.ssp_table_file):
            read_flag = False
            try:
                self.read_ssp_table(self.ssp_table_file)
                memlog('Read SSP table %s'%self.ssp_table_file)
            except:
                memlog('Error reading SSP table %s, will generate...'%self.ssp_table_file)
                read_flag = True
        if read_flag:
            self.generate_ssp_table(self.ssp_table_file)

    def generate_ssp_table(self,ssp_lookup_file,Zsol=Solar['total'],fsps_imf_type=1,fsps_nebular=True,fsps_sfh=0,fsps_zcontinuous=1,oversample=[2,2]):
        '''
        Generates an SPS lookup table, oversampling in [age,metallicity] by oversample
        '''
        import fsps
        memlog('Generating SSP lookup table %s'%(ssp_lookup_file))
        fsps_ssp = fsps.StellarPopulation(sfh=fsps_sfh, zcontinuous=fsps_zcontinuous, dust_type=2, imf_type=fsps_imf_type, add_neb_emission=fsps_nebular)
        fsps_options = np.array([fsps_imf_type,int(fsps_nebular),fsps_sfh,fsps_zcontinuous,oversample[0],oversample[1]],dtype=np.int32)
        wavelengths = fsps_ssp.wavelengths
        ssp_ages = []
        mass_remaining = []
        ssp_ages.append(fsps_ssp.ssp_ages[0])
        mass_remaining.append(fsps_ssp.stellar_mass[0])
        for i in range(len(fsps_ssp.ssp_ages)-1):
            for j in range(i+1,i+oversample[0]):
                ssp_ages.append((fsps_ssp.ssp_ages[j]-fsps_ssp.ssp_ages[j-1])*(j-i)/oversample[0]+fsps_ssp.ssp_ages[j-1])
                mass_remaining.append((fsps_ssp.stellar_mass[j]-fsps_ssp.stellar_mass[j-1])*(j-i)/oversample[0]+fsps_ssp.stellar_mass[j-1])
            ssp_ages.append(fsps_ssp.ssp_ages[j])
            mass_remaining.append(fsps_ssp.stellar_mass[j])
        ssp_logZ = []
        ssp_logZ.append(fsps_ssp.zlegend[0])
        for i in range(len(fsps_ssp.zlegend)-1):
            for j in range(i+1,i+oversample[1]):
                ssp_logZ.append((fsps_ssp.zlegend[j]-fsps_ssp.zlegend[j-1])*(j-i)/oversample[1]+fsps_ssp.zlegend[j-1])
            ssp_logZ.append(fsps_ssp.zlegend[j])
        ssp_logZ = np.log10(ssp_logZ)
        ssp_spectra = []
        for age in ssp_ages:
            for Zmet in ssp_logZ:
                fsps_ssp.params["logzsol"] = Zmet-np.log10(Zsol)
                spectrum = fsps_ssp.get_spectrum(tage=10**(age-9))[1]
                ssp_spectra.append(spectrum)
        with h5py.File(ssp_lookup_file, 'w') as hf:
            hf.create_dataset('fsps_options',data=fsps_options)
            hf.create_dataset('ages',data=ssp_ages)
            hf.create_dataset('logZ',data=ssp_logZ)
            hf.create_dataset('mass_remaining',data=mass_remaining)
            hf.create_dataset('wavelengths',data=wavelengths)
            hf.create_dataset('spectra',data=ssp_spectra)
        memlog('Generated lookup table with %d ages and %d metallicities'%(len(ssp_ages),len(ssp_logZ)))
        self.ssp_ages = np.array(ssp_ages,dtype=MY_DTYPE)
        self.ssp_logZ = np.array(ssp_logZ,dtype=MY_DTYPE)
        self.ssp_mass = np.array(mass_remaining,dtype=MY_DTYPE)
        self.ssp_wavelengths = np.array(wavelengths,dtype=MY_DTYPE)
        self.ssp_spectra = np.array(ssp_spectra,dtype=MY_DTYPE)

    def read_ssp_table(self,ssp_lookup_file):
        hf = h5py.File(ssp_lookup_file,'r')
        for i in hf.keys():
            if i=='fsps_options': fsps_options = list(hf[i])
            if i=='wavelengths': wavelengths = list(hf[i])
            if i=='mass_remaining': mass_remaining = list(hf[i])
            if i=='ages': ssp_ages = list(hf[i])
            if i=='logZ': ssp_logZ = list(hf[i])
            if i=='spectra': ssp_spectra = list(hf[i])
        self.ssp_ages = np.array(ssp_ages,dtype=MY_DTYPE)
        self.ssp_logZ = np.array(ssp_logZ,dtype=MY_DTYPE)
        self.ssp_mass = np.array(mass_remaining,dtype=MY_DTYPE)
        self.ssp_wavelengths = np.array(wavelengths,dtype=MY_DTYPE)
        self.ssp_spectra = np.array(ssp_spectra,dtype=MY_DTYPE)

