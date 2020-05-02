import os
import numpy as np
import yt
import caesar

#MODIFIABLE HEADER POINTING TO LOCATION OF TESTING SNAPSHOTS
snaps = [
    '/ufrc/narayanan/desika.narayanan/caesar_testing_files/m25n256_full_passive/snapshot_305.hdf5',
    '/ufrc/narayanan/desika.narayanan/caesar_testing_files/yt_gizmo_64/output/snap_N64L16_135.hdf5',
]

# Also attempt to caesar everything that we're passed as CLI arguments
snaps.extend(sys.argv[1:])
snaps = [s for s in snaps if os.path.isfile(x)]

for snap in snaps:
    ds = yt.load(simba_snap)
    obj = caesar.CAESAR(ds)
    obj.member_search()
    obj.save('caesar_simba.hdf5')

    obj = caesar.load('caesar_simba.hdf5')

    # This indirectly tests a fair few fields
    print(obj.galinfo())

    # get the masses, radii and metallicities
    masses = [gal.masses['total'] for gal in obj.galaxies]
    metallicity = [gal.metallicity for gal in obj.galaxies]
    radii = [gal.radii['total_half_mass'] for gal in obj.galaxies]

