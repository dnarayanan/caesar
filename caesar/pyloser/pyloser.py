

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
import h5py
from scipy import interpolate
from astropy import constants as const
from astropy.cosmology import FlatLambdaCDM
from caesar.utils import memlog
from caesar.property_manager import MY_DTYPE
from yt.funcs import mylog

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

    """Parent class for photometry objects.  Instantiate this to use the (pyloser) photometry module.
    Note: The heavy lifting for photometry is done in the cython file cyloser.pyx.

    Parameters
    ----------
    obj : CAESAR object
        CAESAR object 
    group_list : array of Group objects
        Groups for which to find photometry
    ds : yt object
        yt object associated with CAESAR object, only needed if running interactively
    band_names : str or list of str
        List of FSPS bands to compute photometry for
    ssp_model : str
        Currently 'FSPS' or 'BPASS' (only needed if generating a new SSP table)
    ssp_table_file : str
        Filename (including path) of SSP table to generate or read in
    view_dir : str
        Lines of sight are along this axis: 'x', 'y', or 'z'
    use_dust : bool
        If true, tries to use dust mass in snapshot, otherwise uses metal mass converted 
        into dust assuming a dust-t-metal ratio
    use_cosmic_ext : bool
        If true, applies Madau (1995) cosmic extinction to spectrum
    kernel_type : str
        Smoothing kernel type: 'cubic' or 'quintic'
    nproc : int
        Number of OpenMP cores, negative means use all but (nproc+1) cores.
    """

    def __init__(self, obj, group_list, ds=None, band_names='v', ssp_model='FSPS', ssp_table_file='FSPS_Chab_EL.hdf5', view_dir='x', use_dust=True, ext_law='mw', use_cosmic_ext=True, kernel_type='cubic', nproc=-1):

        from caesar.property_manager import ptype_ints
        self.obj = obj  # caesar object
        self.groups = group_list  # list of objects to process

        # optional arguments
        self.band_names = band_names
        if hasattr(self.obj,'_kwargs') and 'fsps_bands' in self.obj._kwargs:
            self.band_names = self.obj._kwargs['fsps_bands']
        self.ssp_model = ssp_model
        if hasattr(self.obj,'_kwargs') and 'ssp_model' in self.obj._kwargs:
            self.ssp_model = self.obj._kwargs['ssp_model']
        if 'caesar/' not in ssp_table_file: 
            self.ssp_table_file = os.path.expanduser('~/caesar/%s'%ssp_table_file)
        else:
            self.ssp_table_file = ssp_table_file
        if hasattr(self.obj,'_kwargs') and 'ssp_table_file' in self.obj._kwargs:
            self.ssp_table_file = self.obj._kwargs['ssp_table_file']
        self.ext_law = ext_law
        if hasattr(self.obj,'_kwargs') and 'ext_law' in self.obj._kwargs:
            self.ext_law = self.obj._kwargs['ext_law'].lower()
        if hasattr(self.obj,'_kwargs') and 'view_dir' in self.obj._kwargs:
            view_dir = self.obj._kwargs['view_dir'].lower()
        if view_dir is 'x' or view_dir is '0': self.viewdir = 0
        if view_dir is 'y' or view_dir is '1': self.viewdir = 1
        if view_dir is 'z' or view_dir is '2': self.viewdir = 2
        self.use_dust = use_dust  # if False, will use metals plus an assumed dust-to-metal ratio
        if hasattr(self.obj,'_kwargs') and 'use_dust' in self.obj._kwargs:
            use_dust = self.obj._kwargs['use_dust'].lower()
        self.use_cosmic_ext = use_cosmic_ext
        if hasattr(self.obj,'_kwargs') and 'use_cosmic_ext' in self.obj._kwargs:
            use_cosmic_ext = self.obj._kwargs['use_cosmic_ext'].lower()
        self.kernel_type = kernel_type
        if hasattr(self.obj,'_kwargs') and 'kernel_type' in self.obj._kwargs:
            use_cosmic_ext = self.obj._kwargs['kernel_type'].lower()
        self.nkerntab = 2000
        if nproc == -1:
            try:
                self.nproc = obj.nproc  
            except:
                self.nproc = 1
        else:
            self.nproc = nproc
        self.obj.skip_hash_check = True  # WARNING: this will skip the check that the supplied snapshot file (via ds.load) is the same as the original one used to generate photometry!  

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

        """Main driver routine for pyloser photometry"""

        from caesar.cyloser import compute_AV, compute_mags

        self.init_pyloser()
        #computes AV for all stars in snapshot
        self.obj.AV_star = compute_AV(self)
        #find the AV for stars belonging to the groups that were asked for
        self.Av_per_group()
        spect_dust, spect_nodust = compute_mags(self)
        
        return spect_dust, spect_nodust


    def init_pyloser(self):
        """Initialization routine for pyloser photometry"""

        from caesar.cyloser import init_kerntab
        self.init_ssp_table()
        self.init_extinction()
        self.init_bands()
        init_kerntab(self)
        self.init_stars_to_process()

    #separate AV_all_stars by group
    def Av_per_group(self):
        memlog('Finding LOS A_V values for %d objects'%(len(self.groups)))
        try:
            import tqdm
            for obj_ in tqdm.tqdm(self.groups):
                Av_per_star = self.obj.AV_star[obj_.slist]
                obj_.group_Av = Av_per_star
        
        except: 
            for obj_ in self.groups:
                Av_per_star = self.obj.AV_star[obj_.slist]
                obj_.group_Av = Av_per_star


    # initialize extinction curves.  order: 0=Calzetti, 1=Chevallard, 2=Conroy, 3=Cardelli(MW), 4=SMC, 5=LMC, 6=Mix Calz/MW, 7=Composite Calz/MW/SMC; see atten_laws.py for details (these return optical depths)
    def init_extinction(self):
        """Initialization attenuation laws"""

        from caesar.pyloser.atten_laws import calzetti,chevallard,conroy,cardelli,smc,lmc
        wave = self.ssp_wavelengths.astype(np.float64)
        self.ext_curves = []
        self.ext_curves.append(calzetti(wave))
        self.ext_curves.append(chevallard(wave))
        self.ext_curves.append(conroy(wave))
        self.ext_curves.append(cardelli(wave))
        self.ext_curves.append(smc(wave))
        self.ext_curves.append(lmc(wave))
        self.ext_curves = np.asarray(self.ext_curves)

        if 'calzetti' in self.ext_law: self.ext_law = 0
        elif 'chevallard' in self.ext_law: self.ext_law = 1
        elif 'conroy' in self.ext_law: self.ext_law = 2
        elif self.ext_law == 'mw' or self.ext_law == 'cardelli' or 'CCM' in self.ext_law: self.ext_law = 3
        elif 'smc' in self.ext_law: self.ext_law = 4
        elif 'lmc' in self.ext_law: self.ext_law = 5
        elif self.ext_law == 'mix_calz_MW': self.ext_law = 6
        elif self.ext_law == 'composite': self.ext_law = 7
        else:
            mylog.warning('Extinction law %s not recognized, assuming composite'%self.ext_law)
            self.ext_law = 7


    # set up star and gas lists in each object
    def init_stars_to_process(self):
        """Initialization star and gas particle lists."""

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
        self.obj.smass_orig = smass_at_formation(self.obj,self.groups,self.ssp_mass,self.ssp_ages,self.ssp_logZ,nproc=self.nproc)

        memlog('Loaded %d stars and %d gas in %d objects to process'%(self.scount,self.gcount,self.Nobjs))
        return

    # initialize band transmission data interpolated to FSPS wavelengths
    def init_bands(self):
        """Initialization bands to compute."""

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
            #dnu = self.ssp_wavelengths[ind[0]+1:ind[-1]+2] - self.ssp_wavelengths[ind[0]:ind[-1]+1]  # delta-lambda
            self.band_ftrans = np.append(self.band_ftrans, ftrans*dnu)
            self.band_indexes[ib+1] = len(self.band_ftrans)
            # Now set up band for apparent mag computation
            # We will blueshift the band, corresponding to redshifting the intrinsic spectrum
            ind = np.where((self.ssp_wavelengths > band_wave[0]*self.obj.simulation.scale_factor) & (self.ssp_wavelengths < band_wave[-1]*self.obj.simulation.scale_factor))[0] # indices of wavelengths for redshifted rest-frame spectrum (i.e. blueshifted band)
            self.band_iwz0[ib] = ind[0]
            self.band_iwz1[ib] = ind[-1]+1
            ftrans = np.interp(self.ssp_wavelengths[ind],band_wave*self.obj.simulation.scale_factor,band_trans)  # transmission at those wavelengths
            dnu = CLIGHT_AA/self.ssp_wavelengths[ind[0]:ind[-1]+1] - CLIGHT_AA/self.ssp_wavelengths[ind[0]+1:ind[-1]+2]  # convert to delta-nu
            #dnu = self.ssp_wavelengths[ind[0]+1:ind[-1]+2] - self.ssp_wavelengths[ind[0]:ind[-1]+1]  # delta-lambda
            self.band_ztrans = np.append(self.band_ztrans, np.array(ftrans*dnu*cosmic_ext[ind]))
            self.band_indz[ib+1] = len(self.band_ztrans)

        memlog('Computing %d bands: %s'%(len(self.band_names),self.band_names))

    # initialize SSP table, by either generating it if it doesn't exist or reading it in
    def init_ssp_table(self):
        """Initialization SSP table, either reading it in or creating (and storing) it."""

        import os
        read_flag = False
        if os.path.exists(self.ssp_table_file):
            try:
                self.read_ssp_table(self.ssp_table_file)
                memlog('Read SSP table %s'%self.ssp_table_file)
                read_flag = True
            except:
                memlog('Error reading SSP table %s, will generate...'%self.ssp_table_file)
        if not read_flag:  # generate table with Caesar default options
            if self.ssp_model == 'FSPS':
                ssp_ages, ssp_logZ, mass_remaining, wavelengths, ssp_spectra = generate_ssp_table_fsps(self.ssp_table_file, return_table=True, imf_type=1,add_neb_emission=True,sfh=0,zcontinuous=1)  # note Caesar default FSPS options; run generate_ssp_table() separately to set desired FSPS options
            elif self.ssp_model == 'BPASS':
                ssp_ages, ssp_logZ, mass_remaining, wavelengths, ssp_spectra = generate_ssp_table_bpass(self.ssp_table_file, return_table=True)
            elif self.ssp_model == 'BC03':
                ssp_ages, ssp_logZ, mass_remaining, wavelengths, ssp_spectra = generate_ssp_table_bc03(self.ssp_table_file, return_table=True)
            else:
                print('ssp_model=%s not recognized in generate_ssp_table()')
                sys.exit(-1)
            self.ssp_ages = np.array(ssp_ages,dtype=MY_DTYPE)
            self.ssp_logZ = np.array(ssp_logZ,dtype=MY_DTYPE)
            self.ssp_mass = np.array(mass_remaining,dtype=MY_DTYPE)
            self.ssp_wavelengths = np.array(wavelengths,dtype=MY_DTYPE)
            self.ssp_spectra = np.array(ssp_spectra,dtype=MY_DTYPE)

    def read_ssp_table(self,ssp_lookup_file):
        hf = h5py.File(ssp_lookup_file,'r')
        for i in hf.keys():
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


def generate_ssp_table_fsps(ssp_lookup_file,Zsol=Solar['total'],oversample=[2,2],return_table=False,**fsps_options):
        '''
        Generates an SPS lookup table, oversampling in [age,metallicity] by oversample
        '''
        import fsps
        mylog.info('Generating FSPS SSP lookup table %s'%(ssp_lookup_file))
        mylog.info('with FSPS options: %s'%(fsps_options))
        fsps_opts = ''
        for key, value in fsps_options.items():
            fsps_opts = fsps_opts + ("{0} = {1}, ".format(key, value))
        fsps_opts = np.string_(fsps_opts)
        fsps_ssp = fsps.StellarPopulation(**fsps_options)
        wavelengths = fsps_ssp.wavelengths
        ssp_ages = []
        ssp_ages.append(fsps_ssp.ssp_ages[0])
        for i in range(len(fsps_ssp.ssp_ages)-1):
            for j in range(i+1,i+oversample[0]):
                ssp_ages.append((fsps_ssp.ssp_ages[j]-fsps_ssp.ssp_ages[j-1])*(j-i)/oversample[0]+fsps_ssp.ssp_ages[j-1])
            ssp_ages.append(fsps_ssp.ssp_ages[j])
        ssp_logZ = []
        ssp_logZ.append(fsps_ssp.zlegend[0])
        for i in range(len(fsps_ssp.zlegend)-1):
            for j in range(i+1,i+oversample[1]):
                ssp_logZ.append((fsps_ssp.zlegend[j]-fsps_ssp.zlegend[j-1])*(j-i)/oversample[1]+fsps_ssp.zlegend[j-1])
            ssp_logZ.append(fsps_ssp.zlegend[j])
        ssp_logZ = np.log10(ssp_logZ)
        ssp_spectra = []
        mass_remaining = []
        for Zmet in ssp_logZ:
            for age in ssp_ages:
                fsps_ssp.params["logzsol"] = Zmet-np.log10(Zsol)
                spectrum = fsps_ssp.get_spectrum(tage=10**(age-9))[1]
                ssp_spectra.append(spectrum)
                mass_remaining.append(fsps_ssp.stellar_mass)
        with h5py.File(ssp_lookup_file, 'w') as hf:
            hf.create_dataset('fsps_options',data=fsps_opts)
            hf.create_dataset('ages',data=ssp_ages)
            hf.create_dataset('logZ',data=ssp_logZ)
            hf.create_dataset('mass_remaining',data=mass_remaining)
            hf.create_dataset('wavelengths',data=wavelengths)
            hf.create_dataset('spectra',data=ssp_spectra)
        memlog('Generated FSPS lookup table with %d ages and %d metallicities'%(len(ssp_ages),len(ssp_logZ)))

        if return_table:
            return ssp_ages, ssp_logZ, mass_remaining, wavelengths, ssp_spectra

def generate_ssp_table_bpass(ssp_lookup_file,Zsol=Solar['total'],return_table=False,model_dir = '/home/rad/caesar/BPASSv2.2.1_bin-imf_chab100'):
        '''
        Generates an SPS lookup table from BPASS.
        '''
        from hoki import load
        from glob import glob

        mylog.info('Generating BPASS SSP lookup table %s'%(ssp_lookup_file))
        mylog.info('Using BPASS files in: %s'%(model_dir))

        specfiles = glob(model_dir+'/spectra*')  # these must be gunzipped
        smfiles = glob(model_dir+'/starmass*')  # these must be gunzipped
        output_temp = load.model_output(specfiles[0])
        #output_temp = output_temp[(output_temp.WL>LAMBDA_LO)&(output_temp.WL<LAMBDA_HI)]  # restrict wavelength range for faster calculations
        #print(specfiles[0],output_temp)

        ages = np.array([float(a) for a in output_temp.columns[1:]])
        age_mask = (10**ages / 1e9) < 18 # Gyr
        ages = ages[age_mask]

        wavelengths = output_temp['WL'].values
        metallicities = np.array([None] * len(specfiles))

        for i,mod in enumerate(specfiles):  # parse metallicities from filenames
            try:
                metallicities[i] = float('0.'+mod[-7:-4])
            except: # ...handle em5=1e-5 and em4=1e-4 cases
                metallicities[i] = 10**-float(mod[-5])

        # sort by increasing metallicity
        Z_idx = np.argsort(metallicities)
        metallicities = metallicities[Z_idx].astype(float)

        ssp_spectra = np.zeros((len(ages)*len(metallicities),len(wavelengths)))
        for iZ,mod in enumerate(np.array(specfiles)[Z_idx]):
            output = load.model_output(mod)
            #output = output[(output.WL>LAMBDA_LO)&(output.WL<LAMBDA_HI)]  # restrict wavelength range for faster calculations
            for iage,a in enumerate(ages):
                j = iZ * len(ages) + iage
                ssp_spectra[j] = output[str(a)].values
                ssp_spectra[j] *= wavelengths**2 / CLIGHT_AA  # convert from per AA to per Hz

        mass_remaining = []
        for i,mod in enumerate(np.array(smfiles)[Z_idx]):
            output = load.model_output(mod)
            mass_remaining.append(output['stellar_mass'].values)
        mass_remaining = np.asarray(mass_remaining).flatten()/1.e6  # to Mo

        # convert units
        ssp_ages = ages # log yr
        ssp_logZ = np.log10(metallicities)  
        ssp_spectra /= 1e6 # to Msol
        #print(np.shape(mass_remaining),mass_remaining)

        with h5py.File(ssp_lookup_file, 'w') as hf:
            hf.create_dataset('fsps_options',data=model_dir)
            hf.create_dataset('ages',data=ssp_ages)
            hf.create_dataset('logZ',data=ssp_logZ)
            hf.create_dataset('mass_remaining',data=mass_remaining)
            hf.create_dataset('wavelengths',data=wavelengths)
            hf.create_dataset('spectra',data=ssp_spectra)
        memlog('Generated BPASS lookup table with %d ages and %d metallicities'%(len(ssp_ages),len(ssp_logZ)))

        if return_table:
            return ssp_ages, ssp_logZ, mass_remaining, wavelengths, ssp_spectra

def generate_ssp_table_bc03(ssp_lookup_file,Zsol=Solar['total'],return_table=False,model_dir='/home/rad/caesar/bc03/models/Padova1994/chabrier'):
        '''
        Generates an SPS lookup table from BC03 data.
        '''
        import pandas as pd

        mylog.info('Generating BC03 SSP lookup table %s'%(ssp_lookup_file))
        mylog.info('Using BC03 files in: %s'%(model_dir))

        if 'Padova1994' in model_dir:
            metallicities = np.array([0.0001,0.0004,0.004,0.008,0.02,0.05])  # for Padova 1994 library
        elif 'Padova2000' in model_dir:
            metallicities = np.array([0.0001,0.0004,0.004,0.008,0.019,0.03])  # for Padova 2000 library (not recommended)
        ssp_logZ = np.log10(metallicities)  

        for iZ in range(len(metallicities)):
            ised_file = '%s/bc2003_hr_m%d2_chab_ssp.ised_ASCII'%(model_dir,iZ+2)  # must be un-gzipped!

            # read in entire file
            f = open(ised_file,'r')
            data = f.read().split()

            # get ages
            nage = int(data[0])
            ssp_ages = np.array([np.log10(max(float(x),1.e5)) for x in data[1:nage+1]])

            # get wavelengths
            count = len(ssp_ages)
            while( data[count] != '1221' and data[count] != '6900' ):  # look for possible BC03 nwave values
                count += 1
            nwave = int(data[count])
            wavelengths = np.array([(float(x)) for x in data[count+1:nwave+count+1]])
            count = nwave+count+1

            # initialize arrays
            if iZ == 0:
                ssp_spectra = np.zeros((nage*len(metallicities),nwave))
                mass_remaining = []

            # get spectra
            for iage in range(nage):
                spec = np.array([(float(x)) for x in data[count+1:nwave+count+1]])
                ssp_spectra[iZ*nage+iage] = spec * wavelengths**2 / CLIGHT_AA
                count += nwave+54  # skip past the unknown 52 extra numbers at the end of each line

            # get mass remaining
            m_file = '%s/bc2003_hr_m%d2_chab_ssp.4color'%(model_dir,iZ+2)  
            logage,msleft = np.loadtxt(m_file,usecols=(0,6),unpack=True)
            msleft = np.append(np.array([1.0]),msleft)
            mass_remaining.append(msleft)

        mass_remaining = np.asarray(mass_remaining).flatten()

        with h5py.File(ssp_lookup_file, 'w') as hf:
            hf.create_dataset('fsps_options',data=model_dir)
            hf.create_dataset('ages',data=ssp_ages)
            hf.create_dataset('logZ',data=ssp_logZ)
            hf.create_dataset('mass_remaining',data=mass_remaining)
            hf.create_dataset('wavelengths',data=wavelengths)
            hf.create_dataset('spectra',data=ssp_spectra)
        memlog('Generated BC03 lookup table with %d ages and %d metallicities'%(len(ssp_ages),len(ssp_logZ)))

        if return_table:
            return ssp_ages, ssp_logZ, mass_remaining, wavelengths, ssp_spectra


## For testing: Routine to return a single SSP spectrum at nearest age, Z

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def get_ssp_spectrum(ssp_lookup_file,age,met):
    # age in log(yr), met in log(Z/Zsun)
    hf = h5py.File(ssp_lookup_file,'r')
    for i in hf.keys():
        if i=='wavelengths': wavelengths = np.array(list(hf[i]))
        if i=='mass_remaining': mass_remaining = np.array(list(hf[i]))
        if i=='ages': ssp_ages = np.array(list(hf[i]))
        if i=='logZ': ssp_logZ = np.array(list(hf[i]))
        if i=='spectra': ssp_spectra = np.array(list(hf[i]))
    iage = find_nearest(ssp_ages, age)
    imet = find_nearest(ssp_logZ, met)
    nage = len(ssp_ages)
    ispec = imet*nage + iage
    spec = ssp_spectra[ispec]
    if 'skjdfhl' in ssp_lookup_file:
        print(ssp_lookup_file,ssp_ages,ssp_logZ)
        print(mass_remaining)
        print(ssp_lookup_file,iage,imet,ssp_ages[iage],ssp_logZ[imet],spec)
    return wavelengths,spec


