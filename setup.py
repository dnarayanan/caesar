from setuptools import setup, find_packages
from setuptools.extension import Extension
import numpy as np
from Cython.Build import cythonize

VERSION = '0.1'

cython_extensions = [
    Extension('caesar.group_funcs',
              ['caesar/group_funcs/group_funcs.pyx'],
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

