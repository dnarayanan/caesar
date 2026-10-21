#!/usr/bin/env python
'''
Compute pyloser galaxy magnitudes within fixed projected apertures for an
existing CAESAR catalogue, and append them to the catalogue.

Steps:
  1. A_V to every galaxy star from the LOS dust of its galaxy's gas, exactly as
     in the standard pyloser run (caesar.cyloser.compute_AV).  Stars outside
     galaxies (e.g. intra-halo stars) get A_V=0.
  2. Band luminosities of every star particle in the parent halos of the galaxies
     (caesar.cyloser.compute_star_lums).  Band convolution is linear, so a
     galaxy's luminosity is just the sum over its stars.
  3. For each galaxy, project the halo stars along --view-dir, sum the luminosities
     of stars within each aperture radius of the galaxy centre, and convert to
     magnitudes.  Apertures are cheap: they are just sums over stars.

With --member galaxy, only the galaxy's own (fof6d) stars are used instead of
all stars in its parent halo.

Results are written to galaxy_data/dicts as four dictionaries per aperture,
    absmag_<x>ckpc, absmag_nodust_<x>ckpc, appmag_<x>ckpc, appmag_nodust_<x>ckpc
(datasets <dict>.<band>), with <x> the aperture radius in comoving kpc, the
CAESAR length unit ('.' replaced by 'p'); with --aperture-units kpc (physical)
the suffix is <x>pkpc.  They are picked up automatically by caesar.load(), e.g.
    obj.galaxies[0].absmag_30ckpc['sdss_r']
Galaxies with no stars inside an aperture get NaN.

--verify checks:
  - the aperture selection: stellar mass within CAESAR's own 3D aperture
    (masses['stellar_<n>kpc'], halo stars, comoving, centred on galaxy.pos),
    computed from the same star lists, centres and periodic wrapping, should
    match the catalogue value;
  - the luminosities: summing over each galaxy's member stars should reproduce
    the catalogue absmag (if the catalogue was run with the same pyloser settings).

Requires caesar to be rebuilt after adding compute_star_lums to cyloser.pyx
(e.g. pip install -e .).

example usage:
    % python aperture_photometry.py snap_m50n512_151.hdf5 Groups/m50n512_151.hdf5 \\
          --apertures 5 10 30 50 --bands sdss_u sdss_g sdss_r --view-dir z --nproc 16 --verify
'''

import os
import re
import sys
import shutil
import argparse
import numpy as np
import h5py
import yt
import caesar
from yt.funcs import mylog
from caesar.utils import memlog
from caesar.property_manager import ptype_ints

MAGTYPES = ['absmag', 'absmag_nodust', 'appmag', 'appmag_nodust']


def parse_args():
    p = argparse.ArgumentParser(description='Fixed-aperture pyloser photometry for a CAESAR catalogue')
    p.add_argument('snapfile', help='simulation snapshot used to build the CAESAR catalogue')
    p.add_argument('caesarfile', help='CAESAR catalogue to add aperture magnitudes to')
    p.add_argument('--apertures', type=float, nargs='+', default=[5., 10., 30., 50.], help='aperture radii (default: 5 10 30 50)')
    p.add_argument('--aperture-units', choices=['kpccm', 'kpc'], default='kpccm', help='comoving (default, as CAESAR) or physical kpc')
    p.add_argument('--view-dir', default='x', choices=['x', 'y', 'z'], help='projection and LOS extinction direction')
    p.add_argument('--member', choices=['halo', 'galaxy'], default='halo', help='use stars of the parent halo (default) or of the galaxy only')
    p.add_argument('--centre', choices=['pos', 'minpotpos'], default='pos', help='galaxy centre to use')
    p.add_argument('--bands', nargs='+', default=None, help='FSPS bands (default: bands already in the catalogue absmag, else v)')
    p.add_argument('--ext-law', default='mix_calz_MW', help='extinction law, see pyloser.init_extinction()')
    p.add_argument('--no-dust', action='store_true', help='use metals x DTM instead of snapshot dust masses')
    p.add_argument('--no-cosmic-ext', action='store_true', help='do not apply Madau IGM attenuation')
    p.add_argument('--ssp-table', default=os.path.join(os.path.dirname(caesar.__file__), 'pyloser', 'SSP_Chab_EL.hdf5'), help='SSP lookup table (generated if missing)')
    p.add_argument('--nproc', type=int, default=16, help='number of OpenMP threads')
    p.add_argument('--outfile', default=None, help='write to a copy of the catalogue instead of updating it in place')
    p.add_argument('--overwrite', action='store_true', help='replace existing aperture magnitude blocks')
    p.add_argument('--verify', action='store_true', help='check aperture selection and luminosities against the catalogue')
    p.add_argument('--dry-run', action='store_true', help='compute (and verify) but do not write to the catalogue')
    return p.parse_args()


def aperture_tag(ap, units):
    return ('%g' % ap).replace('.', 'p') + ('ckpc' if units == 'kpccm' else 'pkpc')


def periodic(dx, L):
    return dx - L * np.round(dx / L)


def lums_to_mags(lum, lumtoflux):
    '''Same conversion as apply_bands in cyloser.pyx; NaN where there is no flux.'''
    mags = np.full(lum.shape, np.nan)
    ok = lum > 0
    mags[ok] = -48.6 - 2.5 * np.log10(lum[ok] * lumtoflux)
    return mags


def main():
    args = parse_args()

    from caesar.pyloser.pyloser import photometry, Solar
    from caesar.cyloser import compute_AV, compute_star_lums, init_kerntab, smass_at_formation

    target = args.caesarfile
    if args.outfile is not None:
        shutil.copy2(args.caesarfile, args.outfile)
        target = args.outfile

    obj = caesar.load(target)
    if obj.ngalaxies == 0:
        sys.exit('No galaxies in %s' % target)
    ds = yt.load(args.snapfile)
    ngal = obj.ngalaxies

    # bands: by default, redo the bands already present in the catalogue
    bands = args.bands
    if bands is None:
        bands = sorted(obj._galaxy_dicts['absmag'].keys()) if 'absmag' in obj._galaxy_dicts else ['v']

    # aperture radii in obj.units['length'] (kpccm), the units of data_manager positions
    apertures = np.asarray(args.apertures, dtype=np.float64)
    ap_cm = apertures * (1. + obj.simulation.redshift) if args.aperture_units == 'kpc' else apertures.copy()
    tags = [aperture_tag(a, args.aperture_units) for a in apertures]

    with h5py.File(target, 'r') as hd5:
        existing = [k for k in hd5['galaxy_data/dicts'] if k.split('.')[0] in ['%s_%s' % (m, t) for m in MAGTYPES for t in tags]]
    if existing and not args.overwrite and not args.dry_run:
        sys.exit('Aperture magnitudes already exist in %s (e.g. %s); use --overwrite to replace them' % (target, existing[0]))

    # ---- set up pyloser; this loads star & gas particle data into obj.data_manager
    phot = photometry(obj, [], ds=ds, band_names=bands, ssp_table_file=args.ssp_table, view_dir=args.view_dir,
                      use_dust=not args.no_dust, ext_law=args.ext_law, use_cosmic_ext=not args.no_cosmic_ext, nproc=args.nproc)
    phot.ssp_table_file = os.path.abspath(args.ssp_table)  # photometry() prepends ~/caesar/ to paths without 'caesar/'
    phot.init_ssp_table()
    phot.init_extinction()
    phot.init_bands()
    init_kerntab(phot)
    obj.smass_orig = smass_at_formation(obj, [], phot.ssp_mass, phot.ssp_ages, phot.ssp_logZ, nproc=phot.nproc)
    for b in phot.band_names:
        assert '.' not in b, 'Band name %s contains a ".", which breaks caesar.load() dict parsing' % b

    dm = obj.data_manager
    nstar = obj.simulation.nstar
    L = obj.simulation.boxsize.to(obj.units['length']).d
    phot.boxsize = L
    spos = dm.pos[dm.ptype == ptype_ints['star']]
    smass = np.asarray(dm.mass[dm.ptype == ptype_ints['star']], dtype=np.float64)

    # ---- galaxy & halo info from the catalogue
    gd, hd = obj._galaxy_data, obj._halo_data
    centres = gd[args.centre][:].to(obj.units['length']).d.astype(np.float64)
    parent = np.asarray(gd['parent_halo_index'][:])
    gal_sbins = np.append(gd['slist_start'][:], gd['slist_end'][-1]).astype(np.int64)
    gal_gbins = np.append(gd['glist_start'][:], gd['glist_end'][-1]).astype(np.int64)
    gal_slist = np.asarray(obj._galaxy_slist[:], dtype=np.int64)
    gal_glist = np.asarray(obj._galaxy_glist[:], dtype=np.int64)
    star_gal = np.repeat(np.arange(ngal), np.diff(gal_sbins))  # galaxy index of each entry in gal_slist

    # galaxy log sSFR and log Z/Zsun used to choose the extinction law, as in compute_mags
    sfr = np.asarray(gd['sfr'][:], dtype=np.float64)
    mstar = np.asarray(obj._galaxy_dicts['masses']['stellar'][:], dtype=np.float64)
    logssfr_gal = np.log10(sfr * 1.e9 / (mstar + 1.e-6) + 1.e-20)
    if 'sfr_weighted' in obj._galaxy_dicts.get('metallicities', {}):
        zsfr = np.asarray(obj._galaxy_dicts['metallicities']['sfr_weighted'][:], dtype=np.float64)
    else:
        mylog.warning('No SFR-weighted metallicities in catalogue; assuming solar for extinction law choice')
        zsfr = np.full(ngal, Solar['total'])
    logZ_gal = np.log10(zsfr / Solar['total'] + 1.e-20)

    # ---- step 1: A_V to each galaxy star from its galaxy's gas (standard pyloser)
    phot.ngroup = ngal
    phot.starids, phot.sid_bins = gal_slist, gal_sbins
    phot.gasids, phot.gid_bins = gal_glist, gal_gbins
    AV_star = np.zeros(nstar)
    AV_star[gal_slist] = compute_AV(phot)
    logssfr_star = np.zeros(nstar)
    logssfr_star[gal_slist] = logssfr_gal[star_gal]
    logZ_star = np.zeros(nstar)
    logZ_star[gal_slist] = logZ_gal[star_gal]

    # ---- group galaxies by the set of stars they draw from (parent halo, or own stars)
    pools = {}
    for ig in range(ngal):
        key = ('halo', parent[ig]) if (args.member == 'halo' and parent[ig] >= 0) else ('galaxy', ig)
        pools.setdefault(key, []).append(ig)
    if args.member == 'halo':
        halo_s0 = np.asarray(hd['slist_start'][:])
        halo_s1 = np.asarray(hd['slist_end'][:])
        halo_slist = np.asarray(obj._halo_slist[:], dtype=np.int64)
    pool_stars = []
    for kind, i in pools:
        if kind == 'halo': pool_stars.append(halo_slist[halo_s0[i]:halo_s1[i]])
        else: pool_stars.append(gal_slist[gal_sbins[i]:gal_sbins[i+1]])
    pool_bins = np.append(0, np.cumsum([len(s) for s in pool_stars]))
    sids = np.concatenate(pool_stars)

    # ---- step 2: band luminosities of every star involved
    lums = compute_star_lums(phot, sids, AV_star[sids], logssfr_star[sids], logZ_star[sids])

    # ---- step 3: projected aperture sums
    perp = [k for k in range(3) if k != phot.viewdir]
    nband = len(phot.band_names)
    Lap = {m: np.zeros((len(ap_cm), ngal, nband)) for m in MAGTYPES}
    if args.verify:
        mass_keys = [k for k in obj._galaxy_dicts.get('masses', {}) if re.fullmatch(r'stellar_\d+kpc', k)]
        mass3d = {k: np.zeros(ngal) for k in mass_keys}
    memlog('Summing luminosities within %s for %d galaxies' % (tags, ngal))
    for ipool, members in enumerate(pools.values()):
        rows = slice(pool_bins[ipool], pool_bins[ipool+1])
        pos = spos[sids[rows]].astype(np.float64)
        for ig in members:
            dx = periodic(pos - centres[ig], L)
            R = np.sqrt(np.sum(dx[:, perp]**2, axis=1))  # projected distance
            near = np.where(R < ap_cm.max())[0]  # stars in the largest aperture
            near_lums = {m: lums[m][rows][near] for m in MAGTYPES}
            for iap, apc in enumerate(ap_cm):
                inap = R[near] < apc
                for m in MAGTYPES:
                    Lap[m][iap, ig] = np.sum(near_lums[m][inap], axis=0, dtype=np.float64)
            if args.verify:
                r3d = np.sqrt(np.sum(dx**2, axis=1))
                for k in mass_keys:
                    mass3d[k][ig] = np.sum(smass[sids[rows]][r3d < float(k[len('stellar_'):-len('kpc')])])

    mags = {m: lums_to_mags(Lap[m], phot.lumtoflux_abs if m.startswith('abs') else phot.lumtoflux) for m in MAGTYPES}

    if args.verify:
        verify(obj, phot, parent, smass, sids, lums, gal_slist, star_gal, mass3d, args)

    if args.dry_run:
        memlog('Dry run: nothing written to %s' % target)
        return

    # ---- append to catalogue
    with h5py.File(target, 'r+') as hd5:
        grp = hd5['galaxy_data/dicts']
        for iap, tag in enumerate(tags):
            for m in MAGTYPES:
                for ib, b in enumerate(phot.band_names):
                    key = '%s_%s.%s' % (m, tag, b)
                    if key in grp:
                        if not args.overwrite:
                            sys.exit('%s already exists in %s; use --overwrite to replace it' % (key, target))
                        del grp[key]
                    grp.create_dataset(key, data=mags[m][iap, :, ib].astype(np.float32), compression=1)
    memlog('Wrote %s for apertures %s, bands %s to %s' % (MAGTYPES, tags, phot.band_names, target))


def verify(obj, phot, parent, smass, sids, lums, gal_slist, star_gal, mass3d, args):
    '''Compare with quantities already in the catalogue.'''

    # 1) aperture selection vs CAESAR's 3D aperture masses (same star lists, centres, periodic wrapping)
    if len(mass3d) == 0:
        mylog.warning('verify: no masses[stellar_<n>kpc] in catalogue, cannot check aperture selection')
    elif args.member != 'halo' or args.centre != 'pos':
        mylog.warning('verify: CAESAR aperture masses use --member halo --centre pos; expect differences')
    mpart = np.median(smass[sids])
    in_halo = parent >= 0
    for k, mine in mass3d.items():
        ref = obj._galaxy_dicts['masses'][k][:]
        ref = ref.to('Msun').d if isinstance(ref, yt.YTArray) else np.asarray(ref, dtype=np.float64)
        diff = np.abs(mine - ref)[in_halo]
        nbad = np.sum(diff > 1.5 * mpart)  # a star right on the boundary may fall either side
        mylog.info('verify %s: %d/%d galaxies match CAESAR within one star particle; max |dM| = %.3g Msun '
                   '(median star particle %.3g Msun)' % (k, len(diff) - nbad, len(diff), diff.max() if len(diff) else 0., mpart))
        if nbad > 0:
            for ig in np.where(in_halo)[0][np.argsort(diff)[::-1][:min(nbad, 10)]]:
                mylog.warning('verify %s: galaxy %d  mine %.4g  CAESAR %.4g Msun' % (k, ig, mine[ig], ref[ig]))

    # 2) luminosities: sum over each galaxy's member stars vs the catalogue's whole-galaxy absmag
    if 'absmag' not in obj._galaxy_dicts:
        mylog.warning('verify: no absmag in catalogue, cannot check luminosities')
        return
    row = np.full(obj.simulation.nstar, -1, dtype=np.int64)
    row[sids] = np.arange(len(sids))
    rows = row[gal_slist]
    ok = rows >= 0
    for ib, b in enumerate(phot.band_names):
        if b not in obj._galaxy_dicts['absmag']: continue
        for m in ['absmag', 'absmag_nodust']:
            Lgal = np.bincount(star_gal[ok], weights=lums[m][rows[ok], ib].astype(np.float64), minlength=obj.ngalaxies)
            ref = np.asarray(obj._galaxy_dicts[m][b][:], dtype=np.float64)
            dmag = lums_to_mags(Lgal, phot.lumtoflux_abs) - ref
            dmag = np.abs(dmag[np.isfinite(dmag) & (ref < 99)])
            if len(dmag) == 0: continue
            mylog.info('verify %s %s (whole galaxy): median |dmag| = %.2e, max |dmag| = %.2e over %d galaxies'
                       % (m, b, np.median(dmag), dmag.max(), len(dmag)))


if __name__ == '__main__':
    main()
