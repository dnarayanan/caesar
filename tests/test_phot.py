

#=======================================================================
# Test stand-alone photometry code
#=======================================================================

import yt
import caesar
import pylab as plt
import numpy as np
from scipy.special import erf
import h5py
import sys

MODEL = sys.argv[1]
SNAP = int(sys.argv[2])
WIND = sys.argv[3]

band = 'sdss_r'

if __name__ == '__main__':

    snapfile = '/home/rad/data/%s/%s/snap_%s_%03d.hdf5' % (MODEL,WIND,MODEL,SNAP)
    ds = yt.load(snapfile)
    caesarfile = '/home/rad/data/%s/%s/Groups/%s_%03d.hdf5' % (MODEL,WIND,MODEL,SNAP)
    sim = caesar.load(caesarfile) # load caesar file
    redshift = np.round(sim.simulation.redshift,decimals=2)

# get caesar info for galaxies
    myobjs = sim.galaxies
    mag = np.asarray([m.absmag[band] for m in myobjs]) # load desired mags
    print('%s mag original: %g'%(band,mag[0]))
    ms = np.log10(np.asarray([i.masses['stellar'] for i in myobjs]))
    mlim = np.log10(32*sim.simulation.critical_density.value*sim.simulation.boxsize.value**3*sim.simulation.omega_baryon/sim.simulation.effective_resolution**3) # galaxy mass resolution limit: 32 gas particle masses
    sfr = np.asarray([i.sfr for i in myobjs])
    ssfr = np.log10(1.e9*sfr/ms+10**(-2.5+0.3*sim.simulation.redshift))        # with a floor to prevent NaN's

    # now do photometry in post-processing
    #from caesar.pyloser.pyloser import photometry
    #ds  = sim.yt_dataset
    from caesar.pyloser.pyloser import photometry
    galphot = photometry(sim,sim.galaxies,ds=ds,band_names='sdss',ext_law='composite',nproc=16)
    #galphot.run_pyloser(ssp_model='BPASS')
    spect_dust, spec_nodust = galphot.run_pyloser()
    print('Default (should be same):',galphot.groups[0].absmag['sdss_r'])

    galphot = photometry(sim,sim.galaxies,ds=ds,band_names='sdss',ssp_model='BPASS',ssp_table_file='/home/rad/caesar/BPASS_Chab100.hdf5',ext_law='composite',nproc=16)
    spect_dust, spec_nodust = galphot.run_pyloser()
    print('BPASS (should be different):',galphot.groups[0].absmag['sdss_r'])

    galphot = photometry(sim,sim.galaxies,ds=ds,band_names='sdss',ext_law='calzetti',ssp_model='BC03',ssp_table_file='/home/rad/caesar/BC03_Chab_Padova94.hdf5',nproc=8)
    spect_dust, spec_nodust = galphot.run_pyloser()
    print('BC03 (should be different):',galphot.groups[0].absmag['sdss_r'])

#######################################


