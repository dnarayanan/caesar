import caesar
import yt
import numpy as np

#MODIFIABLE HEADER POINTING TO LOCATION OF TESTING SNAPSHOTS
simba_snap = '/ufrc/narayanan/desika.narayanan/caesar_testing_files/m25n256_full_passive/snapshot_305.hdf5'

small_snap = '/ufrc/narayanan/desika.narayanan/caesar_testing_files/yt_gizmo_64/output/snap_N64L16_135.hdf5'

#----------------------------------------------------------
#1. Make sure that caesar.drive works for a simba snap
#----------------------------------------------------------


ds = yt.load(simba_snap)
obj = caesar.CAESAR(ds)
obj.member_search()
obj.save('caesar_simba.hdf5')

obj2 = caesar.load('caesar_simba.hdf5')

#get the galinfo
print(obj.galinfo())

#get the masses, radii and metallicities
masses = [gal.masses['total'] for gal in obj.galaxies]
metallicity = [gal.metallicity for gal in obj.galaxies]
radii = [gal.radii['total_half_mass'] for gal in obj.galaxies]



#----------------------------------------------------------
#2. Now do it all over again for a generic snapshot
#----------------------------------------------------------

ds = yt.load(small_snap)
obj = caesar.CAESAR(ds)
obj.member_search()
obj.save('caesar_small.hdf5')

obj2 = caesar.load('caesar_small.hdf5')

#get the galinfo
print(obj.galinfo())

#get the masses, radii and metallicities
masses = [gal.masses['total'] for gal in obj.galaxies]
metallicity = [gal.metallicity for gal in obj.galaxies]
radii =    [gal.radii['total_half_mass'] for gal in obj.galaxies]
