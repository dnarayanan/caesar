import six
import numpy as np
cimport numpy as np
import sys
cimport cython
#import matplotlib.pyplot as plt
from cython.parallel import prange, threadid
from caesar.utils import memlog,rotator
from yt.funcs import mylog
from caesar.property_manager import MY_DTYPE
from astropy import constants as const
CLIGHT_AA = const.c.to('AA/s').value

""" ===================================== """
""" IMPORT C STUFF NEEDED FOR COMPUTATION """
""" ===================================== """

from libc.stdlib cimport malloc, free
from libc.stdio cimport printf, fflush, stderr, stdout
from libc.math cimport sqrt as c_sqrt, fabs as c_fabs, log10 as c_log10, exp as c_exp, copysign, fabs


""" ======================================================= """
""" THIS IS THE MAIN ROUTINE TO CALCULATE GROUP PHOTOMETRY  """
""" ======================================================= """

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def compute_mags(phot):

    """Computes magnitudes for photometry object phot """

    from caesar.property_manager import ptype_ints

    memlog('Computing magnitudes in %d bands for %d objects (nproc=%d)'%(len(phot.band_names),phot.ngroup,phot.nproc))
    ptype = phot.obj.data_manager.ptype
    # need stars' LOS hubble velocity (from systemic redshift) for generating spectra
    H0 = phot.obj.simulation.H_z.to('1/s').d
    length_to_km = phot.obj.yt_dataset.quan(1.0,phot.obj.units['length']).to('km').d  # to physical km
    vlos = phot.obj.data_manager.vel[ptype==ptype_ints['star']][phot.starids,phot.viewdir] + phot.obj.data_manager.pos[ptype==ptype_ints['star']][phot.starids,phot.viewdir] * length_to_km * H0
    vbox = phot.obj.simulation.boxsize.to('km').d * H0  # velocity across box
    vlos = np.where(vlos<0, vlos+vbox, vlos)  # periodic wrapping 
    vlos = np.where(vlos>vbox, vlos-vbox, vlos)

    # perhaps need ssfr and Z for choosing extinction law
    from caesar.pyloser.pyloser import Solar
    logssfr_gal = np.array([np.log10((g.sfr*1.e9/(g.masses['stellar']+1.e-6)).d+1.e-20) for g in phot.groups],dtype=MY_DTYPE)  # in Gyr^-1
    logZ_gal = np.array([np.log10(g.metallicities['sfr_weighted']/Solar['total']+1.e-20) for g in phot.groups],dtype=MY_DTYPE)

    cdef:
        ## input quantities
        long int[:] starid_bins = phot.sid_bins   # starting indexes of star particle IDs in each group
        float[:] sm = phot.obj.smass_orig[phot.starids]
        float[:] svz = MY_DTYPE(vlos)
        float[:] sage = phot.obj.data_manager.age[phot.starids]
        float[:] sZ = phot.obj.data_manager.sZ[phot.starids] #/phot.solar_abund['total']
        float[:] ssp_wavelengths = phot.ssp_wavelengths
        float[:] ssp_ages = phot.ssp_ages
        float[:] ssp_logZ = phot.ssp_logZ
        float[:,:] ssp_spectra = phot.ssp_spectra
        float[:] AV_star = phot.obj.AV_star  # A_V to each star
        float[:] ftrans = phot.band_ftrans.astype(MY_DTYPE)  # band transmissions
        int[:] itrans = phot.band_indexes  # starting index for band transmission
        int[:] iwave0 = phot.band_iwave0  # starting index covering band in ssp_wavelengths
        int[:] iwave1 = phot.band_iwave1  # ending index 
        float[:] ztrans = phot.band_ztrans.astype(MY_DTYPE)  # blueshifted band transmissions
        int[:] jtrans = phot.band_indz  # starting index for blueshifted band transmission
        int[:] iwz0 = phot.band_iwz0  # starting index covering blueshifted band in ssp_wavelengths
        int[:] iwz1 = phot.band_iwz1  # ending index 
        float[:,:] extinctions = phot.ext_curves.astype(MY_DTYPE) # extinction curves
        float[:] ssfr = logssfr_gal   # specific SFR, used to choose extinction law
        float[:] Zgal = logZ_gal   # SFR-weighted metallicity scaled to solar, used to choose extinction law
        # general variables
        int ng = phot.ngroup
        int npart = len(sm)
        int nlam = len(phot.ssp_wavelengths)
        int nage = len(phot.ssp_ages)
        int nZ = len(phot.ssp_logZ)
        int nbands = len(phot.band_names)
        int nextinct = phot.ext_law
        int my_nproc = phot.nproc
        float redshift = phot.obj.simulation.redshift
        float lumtoflux = phot.lumtoflux
        float lumtoflux_abs = phot.lumtoflux_abs
        float[:] dnu = MY_DTYPE(CLIGHT_AA/phot.ssp_wavelengths[:nlam-1] - CLIGHT_AA/phot.ssp_wavelengths[1:])
        float splitage = phot.splitting_age
        float msum
        int ib,ig,ip,idim,ikern,istart,iend
        # things to compute
        float[:,:] spect_dust = np.zeros((ng,nlam),dtype=MY_DTYPE)   # galaxy spectra with extinction
        float[:,:] spect_nodust = np.zeros((ng,nlam),dtype=MY_DTYPE)   # galaxy spectra without extinction
        float[:,:] absmags = np.zeros((ng,nbands),dtype=MY_DTYPE)    # absolute mags
        float[:,:] appmags = np.zeros((ng,nbands),dtype=MY_DTYPE)    # apparent mags
        float[:,:] absmags_nd = np.zeros((ng,nbands),dtype=MY_DTYPE) # absolute mags, no dust
        float[:,:] appmags_nd = np.zeros((ng,nbands),dtype=MY_DTYPE) # apparent mags, no dust
        float[:]   L_FIR = np.zeros(ng,dtype=MY_DTYPE)               # dust-reprocessed luminosity

    for ig in prange(ng,nogil=True,schedule='dynamic',num_threads=my_nproc):
        istart = starid_bins[ig]
        iend = starid_bins[ig+1]
        # compute spectrum of galaxy with and without dust
        get_galaxy_spectrum(istart,iend,sm,sage,sZ,svz,AV_star,extinctions,nextinct,ssfr[ig],Zgal[ig],splitage,nlam,nage,nZ,ssp_wavelengths,ssp_ages,ssp_logZ,ssp_spectra,spect_dust[ig],spect_nodust[ig])
        get_magnitudes(nbands,nlam,itrans,ftrans,jtrans,ztrans,iwave0,iwave1,iwz0,iwz1,lumtoflux,lumtoflux_abs,spect_dust[ig],spect_nodust[ig],absmags[ig],absmags_nd[ig],appmags[ig],appmags_nd[ig])
        # extincted light goes into bolometric far-IR luminosity 
        for ip in range(nlam-1):
            L_FIR[ig] += (spect_nodust[ig][ip]-spect_dust[ig][ip])*dnu[ip]

    # wavelength indices over which to compute UV spectral slope beta
    bwave0 = (np.abs(phot.ssp_wavelengths - phot.beta_wave[0])).argmin() 
    bwave1 = (np.abs(phot.ssp_wavelengths - phot.beta_wave[1])).argmin()
    # load magnitudes into group objects
    for ig in range(ng):
        phot.groups[ig].absmag = {}
        phot.groups[ig].absmag_nodust = {}
        phot.groups[ig].appmag = {}
        phot.groups[ig].appmag_nodust = {}
        phot.groups[ig].photometry = {}
        for ib,b in enumerate(phot.band_names):
            phot.groups[ig].absmag[b] = absmags[ig,ib]
            phot.groups[ig].absmag_nodust[b] = absmags_nd[ig,ib]
            phot.groups[ig].appmag[b] = appmags[ig,ib]
            phot.groups[ig].appmag_nodust[b] = appmags_nd[ig,ib]
        phot.groups[ig].photometry['L_FIR'] = phot.obj.yt_dataset.quan(L_FIR[ig],'Lsun')
        # Compute beta
        wavefit = np.array(phot.ssp_wavelengths)[bwave0:bwave1]
        specfit = np.array(spect_dust[ig])[bwave0:bwave1]
        phot.groups[ig].photometry['beta'] = get_beta(wavefit,specfit)
        specfit = np.array(spect_nodust[ig])[bwave0:bwave1]
        phot.groups[ig].photometry['beta_nodust'] = get_beta(wavefit,specfit)

    return np.array(spect_dust), np.array(spect_nodust)


""" =============================================== """
""" SUBROUTINES TO CALCULATE PHOTOMETRIC PROPERTIES """
""" =============================================== """

def get_beta(wavefit,specfit):
    wavefit = np.array(wavefit)
    # beta defined for f_lambda, but spectrum is f_nu, so do conversion from 1/Hz -> 1/A
    specfit = np.array(specfit) * const.c.to('AA/s').value / wavefit**2
    wavefit = np.log10(wavefit[specfit>0])
    specfit = np.log10(specfit[specfit>0])
    if (len(specfit)>0):
        slope, intcpt = np.polyfit(wavefit,specfit,1)
    else:
        slope = -100
    return slope

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef void get_magnitudes(int nbands, int nlam, int[:] itrans, float[:] ftrans, int[:] jtrans, float[:] ztrans, int[:] iwave0, int[:] iwave1, int[:] iwz0, int[:] iwz1, float lumtoflux, float lumtoflux_abs, float[:] spect_dust, float[:] spect_nodust, float[:] absmag, float[:] absmag_nd, float[:] appmag, float[:] appmag_nd) noexcept nogil:

    cdef int ib

    for ib in range(nbands):
        # compute absolute magnitudes
        absmag[ib] = apply_bands(ib,spect_dust,ftrans,iwave0[ib],iwave1[ib],itrans[ib],lumtoflux_abs)
        absmag_nd[ib] = apply_bands(ib,spect_nodust,ftrans,iwave0[ib],iwave1[ib],itrans[ib],lumtoflux_abs)
        # Compute apparent magnitudes
        appmag[ib] = apply_bands(ib,spect_dust,ztrans,iwz0[ib],iwz1[ib],jtrans[ib],lumtoflux)
        appmag_nd[ib] = apply_bands(ib,spect_nodust,ztrans,iwz0[ib],iwz1[ib],jtrans[ib],lumtoflux)

    return

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef float apply_bands(int iband, float[:] spectrum, float[:] ftrans, int iwave0, int iwave1, int itrans, float lumtoflux) nogil:

    cdef int i
    cdef double lum=0., res=0.
    cdef float bandmag

    for i in range(iwave0,iwave1):
        lum += spectrum[i]*ftrans[i-iwave0+itrans]
        res += ftrans[i-iwave0+itrans]
    if res > 0 and lum > 0:
        bandmag = <float>(-48.6-2.5*c_log10(lum*lumtoflux/res))
    else:
        bandmag = 100.0   # set mag=100 for no flux
    return bandmag

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef void extinction(float sAV, float[:,:] extinct, int nextinct, float ssfr, float Zgal, int nlam, float *dust_ext) noexcept nogil:

    cdef int i=0
    cdef double sfact,zfact

    for i in range(nlam):
        dust_ext[i] = 1.0
    if sAV < 1.e-4: return 

    if nextinct <= 5:
        for i in range(nlam):
            dust_ext[i] = sAV * extinct[nextinct][i]  # use single attenuation/extinction law
    elif nextinct == 6 or nextinct == 7:  # composite extinction curves
        # Calzetti for log sSFR>0 Gyr^-1, MW for log sSFR<-1, linear mix in between
        sfact = ssfr + 1
        if sfact > 1.: sfact = 1.
        if sfact < 0.: sfact = 0.
        for i in range(nlam):
            dust_ext[i] = sAV * (extinct[0][i]*sfact+ extinct[3][i]*(1.-sfact)) 
        if nextinct == 7: # mix in SMC at low metallicities; ramp up SMC frac from logZ=-1 to -2.
            zfact = Zgal + 2
            if zfact > 1.: zfact = 1.
            if zfact < 0.: zfact = 0.
            for i in range(nlam):
                dust_ext[i] *= (dust_ext[i]*zfact+ extinct[4][i]*(1.-zfact)) 

    for i in range(nlam):
        dust_ext[i] = c_exp(-dust_ext[i])  # turn optical depths into attenuation factor

    return

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef void get_galaxy_spectrum(int istart, int iend, float[:] sm, float[:] sage, float[:] sZ, float[:] svz, float[:] sAV, float[:,:] extinct, int nextinct, float ssfr, float Zgal, float splitage, int nlam, int nage, int nZ, float[:] ssp_wavelengths, float[:] ssp_ages, float[:] ssp_logZ, float[:,:] ssp_spectra, float[:] spec, float[:] spec_nd) noexcept nogil:

    cdef int i,j,k,zsign,nbin
    cdef float *dust_ext
    cdef float *spec_star
    cdef float zfact=0.
    cdef float age,mass,dage

    dust_ext = <float *> malloc(nlam*sizeof(float))
    spec_star = <float *> malloc(nlam*sizeof(float))
    #for i in range(istart,iend):
    #    zfact += sm[i]
    for i in range(nlam):
        spec[i] = 0.
        spec_nd[i] = 0.
    for ip in range(istart,iend):
        # split star into 2*nbin+1 time bins, if younger than splitage
        nbin = int(splitage / sage[ip])
        if nbin > 10: nbin = 10  # reasonable upper limit 
        dage = sage[ip] / (nbin+1)
        mass = sm[ip] / (2*nbin+1)
        for k in range(-nbin,nbin+1):
            # get age and mass for this portion of the (split) star
            age = sage[ip] + k*dage
            # get spectrum of 1 Mo star interpolated to its age and metallicity
            interp_tab(age,sZ[ip],nlam,nage,nZ,ssp_wavelengths,ssp_ages,ssp_logZ,ssp_spectra,spec_star)
            extinction(sAV[ip],extinct,nextinct,ssfr,Zgal,nlam,dust_ext)
            zfact = svz[ip]/3.e10
            if zfact >= 0: zsign = 1
            else: zsign = -1
            for i in range(nlam):
                # determine shift in wavelength index owing to star's peculiar velocity
                j = i
                while fabs(ssp_wavelengths[j]-ssp_wavelengths[i]) < fabs(ssp_wavelengths[i]*zfact):
                    j += zsign
                if fabs(ssp_wavelengths[j]-ssp_wavelengths[i]*(1.+zfact)) > fabs(ssp_wavelengths[j-zsign]-ssp_wavelengths[i]*(1.+zfact)):
                    j -= zsign  # check if it's closer to wavelength j or the previous wavelength
                if j<0 or j>=nlam: continue
                # add star's spectrum to shifted wavelength bin
                spec[j] += mass * dust_ext[i] * spec_star[i] 
                spec_nd[j] += mass * spec_star[i] 

    free(spec_star)
    free(dust_ext)
    return


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef int index_search(float x, float[:] myvec, int nvec) nogil:
    ''' Determines index of value in myvec that is closest but just below x, via bisection search.
    Assumes myvec is sorted in increasing order.'''
    cdef int ilo=0, ihi, imid

    if x < myvec[0]: return 0
    if x > myvec[nvec-2]: return nvec-2
    ihi = nvec
    while ihi - ilo > 1:
        imid = int(0.5*ihi+0.5*ilo)
        if myvec[imid] < x: ilo = imid
        elif myvec[imid] > x: ihi = imid
        else: 
            ilo = imid
            break
    if ilo >= nvec-1: ilo = nvec-2
    return ilo


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef void interp_tab(float age,float met, int nlam, int nage, int nZ, float[:] ssp_wavelengths, float[:] ssp_ages, float[:] ssp_logZ, float[:,:] ssp_spectra, float *spec_star) noexcept nogil:

    cdef int i,iage,iZ,i00,i01,i10,i11
    cdef float logage, logZ, fage, fZ

    logage = c_log10(age+1.e-20)+9
    logZ = c_log10(met+1.e-20)
    iage = index_search(logage,ssp_ages,nage)
    iZ = index_search(logZ,ssp_logZ,nZ)
    fage = (logage-ssp_ages[iage])/(ssp_ages[iage+1]-ssp_ages[iage])
    fZ = (logZ-ssp_logZ[iZ])/(ssp_logZ[iZ+1]-ssp_logZ[iZ])
    i00 = iZ*nage+ iage
    i01 = iZ*nage + iage + 1
    i10 = (iZ+1)*nage + iage
    i11 = (iZ+1)*nage + iage + 1
    for i in range(nlam):
        spec_star[i] = fage*fZ*ssp_spectra[i11,i] + (1-fage)*fZ*ssp_spectra[i10,i] + fage*(1-fZ)*ssp_spectra[i01,i] + (1-fage)*(1-fZ)*ssp_spectra[i00,i]

    return 

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def compute_AV(phot,alpha=0.,beta=0.):

    from caesar.property_manager import ptype_ints

    ptype = phot.obj.data_manager.ptype
    if phot.use_dust:
        gmass = phot.obj.data_manager.mass[ptype==ptype_ints['gas']][phot.gasids]
        gmet = phot.obj.data_manager.dustmass[phot.gasids]
        gmet /= gmass
    else:
        gmass = phot.obj.data_manager.mass[ptype==ptype_ints['gas']][phot.gasids]
        gmet = phot.obj.data_manager.gZ[phot.gasids]

    memlog('Computing A_V for %d objects (nproc=%d)'%(phot.ngroup,phot.nproc))

    cdef:
        ## gas quantities
        long int[:] gasid_bins = phot.gid_bins   # starting indexes of gas particle IDs in each group
        long int[:] starid_bins = phot.sid_bins   # starting indexes of star particle IDs in each group
        double[:,:] spos = phot.obj.data_manager.pos[ptype==ptype_ints['star']][phot.starids].astype('float64')
        double[:,:] gpos = phot.obj.data_manager.pos[ptype==ptype_ints['gas']][phot.gasids].astype('float64')
        float[:]   ghsm = phot.obj.data_manager.hsml[phot.gasids]
        float[:]   gm = gmass
        float[:]   gZ = gmet
        float[:]   kerntab = phot.kerntab
        # general variables
        int ng = phot.ngroup
        int npart = len(spos)  # number of stars
        int idir = phot.viewdir
        int my_nproc = phot.nproc
        int nkerntab = len(phot.kerntab)
        bint usedust = phot.use_dust
        float Lbox = phot.boxsize
        int ig,ip,istart,iend,igstart,igend
        # useful constants
        float redshift = phot.obj.simulation.redshift
        double NHcol_fact = 1.99e33*0.76*(1.+redshift)*(1.+redshift)/(3.086e21**2*1.673e-24)
        double AV_fact = 1./(2.2e21*0.0189) # Watson 2011 arXiv:1107.6031 (note: Watson calibrates to Zsol=0.0189)
        float dtm_MW = 0.4/0.6  # dust-to-(total)metal ratio in MW = 40% (Dwek 1998, Watson 2012)
        # Things to compute
        float[:]   A_V = np.zeros(npart,dtype=MY_DTYPE)  # A_V for stars 
        float[:]   Zcol = np.zeros(npart,dtype=MY_DTYPE)  # metal column density for stars 


    if (alpha != 0.) | (beta != 0.):
        memlog('Rotating particles (alpha=%d,beta=%d)'%(alpha,beta))
        spos = rotator(spos, alpha,beta)
        gpos = rotator(gpos, alpha,beta)

    for ig in prange(ng,nogil=True,schedule='dynamic',num_threads=my_nproc):
        istart = starid_bins[ig]
        iend = starid_bins[ig+1]
        igstart = gasid_bins[ig]
        igend = gasid_bins[ig+1]
        for ip in range(istart,iend):
            A_V[ip] = star_AV(ip, idir, igstart, igend, spos, gpos, gm, gZ, ghsm,  Lbox, nkerntab, kerntab, redshift, dtm_MW,  NHcol_fact, AV_fact, usedust)

    return np.array(A_V,dtype=MY_DTYPE)

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef float star_AV(int ip, int idir, int igstart, int igend, double[:,:] spos, double[:,:] gpos, float[:] gm, float[:] gZ, float[:] ghsm, float Lbox, int nkerntab, float[:] kerntab, float redshift, float dtm_MW, float NHcol_fact, float AV_fact, bint usedust) nogil:

    cdef:
        int i,idim
        float A_V=0.,Zcol=0.
        float dx2, kernint_val
        float dx[3]
        float dtm_slope, dtm_int, dtm

    for i in range(igstart,igend):
        for idim in range(3):
            dx[idim] = spos[ip,idim] - gpos[i,idim] 
            if dx[idim] > 0.5*Lbox: dx[idim] -= Lbox
            if dx[idim] > 0.5*Lbox: dx[idim] += Lbox
        if dx[idir] > 0: continue  # only use gas in front of stars
        if idir == 0: dx2 = dx[1]*dx[1]+dx[2]*dx[2]
        if idir == 1: dx2 = dx[0]*dx[0]+dx[2]*dx[2]
        if idir == 2: dx2 = dx[0]*dx[0]+dx[1]*dx[1]
        if dx2 > ghsm[i]*ghsm[i]: continue  # gas does not intersect LOS to star
        ikern = <int>(nkerntab*c_sqrt(dx2/(ghsm[i]*ghsm[i])))
        kernint_val = kerntab[ikern]
        Zcol += gm[i]*gZ[i] * kernint_val / (ghsm[i]*ghsm[i]) # sum metal column
    A_V = Zcol * NHcol_fact * AV_fact

    if Zcol>0 and not usedust:      #  use average Z to adjust dust-to-metal ratio
        Zcol /= 0.0134
        # z- and Z-dependent fit to Simba-100 dust-to-metal ratio (Li etal 2019)
        dtm_slope = -0.104*redshift + 0.97  # slope at a given redshift
        dtm_int = -0.059*redshift + 0.005   # intercept at a given redshift
        dtm = 10**(dtm_slope*c_log10(Zcol) + dtm_int)  # dtm value at given metallicity
        if dtm<dtm_MW: A_V *= dtm/dtm_MW  # scale dust extinction to MW dtm value

    return A_V

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef double kernel(double q,int ktype) nogil:

    if ktype==0:
        if q < 0.5: return (2.546479089470 + 15.278874536822 * (q - 1) * q * q)
        elif q < 1: return 5.092958178941 * (1.0 - q) * (1.0 - q) * (1.0 - q)
        else: return 0
    elif ktype==1:
        if q<0.333333333: return 27.0*(6.4457752*q*q*q*q*(1.0-q) -1.4323945*q*q +0.17507044)
        elif q<0.666666667: return 27.0*(3.2228876*q*q*q*q*(q-3.0) +10.7429587*q*q*q -5.01338071*q*q +0.5968310366*q +0.1352817016)
        elif q<1: return 27.0*0.64457752*(-q*q*q*q*q +5.0*q*q*q*q -10.0*q*q*q +10.0*q*q -5.0*q +1.0)
        else: return 0

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def init_kerntab(phot):

    if phot.kernel_type == 'cubic': kt = 0
    elif phot.kernel_type == 'quintic': kt = 1
    else:
        memlog('Kernel type %s not recognized, assuming cubic'%phot.kernel_type)
        kt = 0

    cdef:
        int i,j
        int my_nproc = phot.nproc
        int ktype = kt
        int ntab = phot.nkerntab
        double[:] kerntab = np.zeros(phot.nkerntab+1)
        double binsize = 1./phot.nkerntab
        double b,x,xnext,sqrtl,sqrtlnext

    for i in prange(ntab+1,nogil=True,num_threads=my_nproc):
        b = i*binsize
        kerntab[i] = 0.0
        for j in range(0,ntab+1):
            x = j*binsize
            xnext = x+binsize
            sqrtl = c_sqrt(x*x+b*b)
            sqrtlnext = c_sqrt(xnext*xnext+b*b)
            if i>0: kerntab[i] += 0.5*binsize*(x*kernel(sqrtl,ktype)/sqrtl+xnext*kernel(sqrtlnext,ktype)/sqrtlnext)
            else: kerntab[i] += 0.5*binsize*(kernel(sqrtl,ktype)+kernel(sqrtlnext,ktype))
        kerntab[i] *= 2.0       # above calculates integral in half the kernel, so must double

    phot.kerntab = np.array(kerntab, dtype=MY_DTYPE)
    #memlog('Initialized kernel table with %d entries'%phot.nkerntab)

    return 


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def smass_at_formation(obj,group_list,ssp_origmass,ssp_ages,ssp_logZ,nproc=16):

    from caesar.property_manager import ptype_ints

    # get original stellar mass at time of formation
    obj.smass_orig = np.zeros(obj.simulation.nstar,dtype=MY_DTYPE)
    ptype = obj.data_manager.ptype

    cdef:
        int istar,iage,iZ
        float[:] smass = obj.data_manager.mass[ptype==ptype_ints['star']]
        float[:] sage = obj.data_manager.age
        float[:] sZ = obj.data_manager.sZ
        int nstar = len(smass)
        float[:] ssp_morig = ssp_origmass  # fraction of original mass remaining
        float[:] ssp_age = ssp_ages
        float[:] ssp_met = ssp_logZ
        int nage = len(ssp_age)
        int nZ = len(ssp_met)
        int my_nproc = nproc
        float logage,fage,logZ,fZ
        int i00,i01,i10,i11
        float[:] smorig = np.zeros(nstar,dtype=MY_DTYPE)  # quantity to compute

    for istar in prange(nstar,nogil=True,num_threads=my_nproc):
        logage = c_log10(sage[istar]+1.e-20)+9
        logZ = c_log10(sZ[istar]+1.e-20)
        iage = index_search(logage,ssp_age,nage)
        fage = (logage-ssp_age[iage])/(ssp_age[iage+1]-ssp_age[iage])
        iZ = index_search(logZ,ssp_met,nZ)
        fZ = (logZ-ssp_met[iZ])/(ssp_met[iZ+1]-ssp_met[iZ])
        i00 = iZ*nage+ iage
        i01 = iZ*nage + iage + 1
        i10 = (iZ+1)*nage + iage
        i11 = (iZ+1)*nage + iage + 1
        smorig[istar] = fage*fZ*ssp_morig[i11] + (1-fage)*fZ*ssp_morig[i10] + fage*(1-fZ)*ssp_morig[i01] + (1-fage)*(1-fZ)*ssp_morig[i00]  # look up remaining mass fraction
        smorig[istar] = smass[istar] / smorig[istar]  # correct to original mass

    return np.array(smorig)

