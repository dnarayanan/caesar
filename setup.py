from setuptools import setup, find_packages
from setuptools.extension import Extension
import numpy as np
from Cython.Build import cythonize

import sys
sys.path.insert(0,'caesar')
from version import VERSION

cython_extensions = [
    Extension('caesar.group_funcs',
              ['caesar/group_funcs/group_funcs.pyx'],
              include_dirs = [np.get_include()]),
    Extension('caesar.hydrogen_mass_calc',
              ['caesar/hydrogen_mass_calc/hydrogen_mass_calc.pyx'],
              include_dirs = [np.get_include()]),
]

setup(
    name='caesar',
    version=VERSION,
    description='Caesar does cool stuff.',
    url='https://bitbucket.org/rthompson/caesar',
    author='Robert Thompson',
    author_email='rthompsonj@gmail.com',
    license='not sure',

    classifiers=[],
    keywords='',
    entry_points={
        'console_scripts':['caesar = caesar.command_line:run']
    },
    packages=find_packages(),
    setup_requires=['numpy','cython>=0.22'],
    install_requires=['numpy','h5py','cython'],
    ext_modules=cythonize(cython_extensions)
)

