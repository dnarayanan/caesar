import caesar
import numpy as np
import pprint
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('caesar_file')
args = parser.parse_args()

obj = caesar.load(args.caesar_file)
qobj = caesar.quick_load(args.caesar_file)

assert len(obj.clouds) == len(qobj.clouds)
for cloud, qcloud in zip(obj.clouds, qobj.clouds):
    for k, v in cloud.__dict__.items():
        if k[0] != '_' and k not in ['obj']:
            if isinstance(getattr(cloud, k), np.ndarray):
                if np.any(getattr(cloud, k) != getattr(qcloud, k)):
                    print(k)
                    pprint.pprint(getattr(cloud, k))
                    pprint.pprint(getattr(qcloud, k))
                    print()
            else:
                if getattr(cloud, k) != getattr(qcloud, k):
                    print(k)
                    print(getattr(cloud, k))
                    print(getattr(qcloud, k))
                    print()

assert len(obj.galaxies) == len(qobj.galaxies)
for galaxy, qgalaxy in zip(obj.galaxies, qobj.galaxies):
    for k, v in galaxy.__dict__.items():
        if k[0] != '_' and k not in ['obj', 'halo', 'satellites', 'clouds']:
            if isinstance(getattr(galaxy, k), np.ndarray):
                if np.any(getattr(galaxy, k) != getattr(qgalaxy, k)):
                    print(k)
                    pprint.pprint(getattr(galaxy, k))
                    pprint.pprint(getattr(qgalaxy, k))
                    print()
            else:
                if getattr(galaxy, k) != getattr(qgalaxy, k):
                    print(k)
                    print(getattr(galaxy, k))
                    print(getattr(qgalaxy, k))
                    print()

assert len(obj.halos) == len(qobj.halos)
for halo, qhalo in zip(obj.halos, qobj.halos):
    for k, v in halo.__dict__.items():
        if k[0] != '_' and k not in ['obj', 'galaxies', 'central_galaxy', 'satellite_galaxies', 'masses']:
            if isinstance(getattr(halo, k), np.ndarray):
                if np.any(getattr(halo, k) != getattr(qhalo, k)):
                    print(k)
                    print(getattr(halo, k))
                    print(getattr(qhalo, k))
                    print()
            else:
                if getattr(halo, k) != getattr(qhalo, k):
                    print(k)
                    pprint.pprint(getattr(halo, k))
                    pprint.pprint(getattr(qhalo, k))
                    print()
