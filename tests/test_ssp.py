
# Generate two SSP tables, plot spectra at selected age, met to compare 

import numpy as np
import pylab as plt
from caesar.pyloser.pyloser import get_ssp_spectrum,generate_ssp_table_bpass,generate_ssp_table_bc03

colors = ['b','c','g','r']
#generate_ssp_table_bpass('../BPASS_Chab100.hdf5')
generate_ssp_table_bc03('../BC03_Chab_Padova94.hdf5')
wave = [2000,10000]  # wavelength range to plot
icolor = 0
for age in [9.0,10.0]:  # log(age/yr)
    for met in [-1.3]:  # log(Z_mass_fraction)
        wave_fsps,spec_fsps = get_ssp_spectrum('../FSPS_Chab_EL.hdf5',age,met)
        wave_bpass,spec_bpass = get_ssp_spectrum('../BC03_Chab_Padova94.hdf5',age,met)
        sel_fsps = (wave_fsps>wave[0]) & (wave_fsps<wave[1])
        sel_bpass = (wave_bpass>wave[0]) & (wave_bpass<wave[1])
        plt.plot(wave_fsps[sel_fsps],np.log10(spec_fsps[sel_fsps]),'-',c=colors[icolor],label='FSPS %g %g'%(age,met))
        icolor += 1
        plt.plot(wave_bpass[sel_bpass],np.log10(spec_bpass[sel_bpass]),':',c=colors[icolor],label='BPASS %g %g'%(age,met))
        icolor += 1

plt.xlabel('Wavelength (AA)')
plt.ylabel('Flux (Lsun/Hz)')
plt.legend()
plt.show()

