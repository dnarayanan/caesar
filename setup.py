from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.sdist import sdist as _sdist
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.build_ext import build_ext as _build_ext
import os
import sys

sys.path.insert(0, 'caesar')
from __version__ import VERSION


class build_py(_build_py):
    def run(self):
        _build_py.run(self)


class build_ext(_build_ext):
    # subclass setuptools extension builder to avoid importing numpy
    # at top level in setup.py. See http://stackoverflow.com/a/21621689/1382869
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process
        # see http://stackoverflow.com/a/21621493/1382869
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


class sdist(_sdist):
    # subclass setuptools source distribution builder to ensure cython
    # generated C files are included in source distribution.
    # See http://stackoverflow.com/a/18418524/1382869
    def run(self):
        # Make sure the compiled Cython files in the distribution are up-to-date
        from Cython.Build import cythonize
        cythonize(cython_extensions)
        _sdist.run(self)

if sys.platform == 'darwin':
    compile_arg = ""
    link_arg = ""
else:
    compile_arg = "-fopenmp"
    link_arg = '-fopenmp'

        
cython_extensions = [
    Extension('caesar.group_funcs',
              sources=['caesar/group_funcs/group_funcs.pyx'],
              extra_compile_args=[compile_arg],
              extra_link_args=[link_arg]),
    Extension('caesar.hydrogen_mass_calc',
              sources=['caesar/hydrogen_mass_calc/hydrogen_mass_calc.pyx'],
              extra_compile_args=[compile_arg],
              extra_link_args=[link_arg]),
    Extension('caesar.cyloser',
              sources=['caesar/pyloser/cyloser.pyx'],
              extra_compile_args=[compile_arg],
              extra_link_args=[link_arg])
]


setup(
    name='caesar',
    version=VERSION,
    description='CAESAR is a python library for analyzing the outputs from cosmological simulations.',
    url='https://github.com/dnarayanan/caesar',
    author='Robert Thompson, Desika Narayanan, Romeel Davé, Benjamin Kimock',
    author_email='desika.narayanan@gmail.com',
    license='not sure',
    classifiers=[],
    keywords='',
    entry_points={'console_scripts': ['caesar = caesar.command_line:run']},
    packages=find_packages(),
    setup_requires=['numpy', 'cython>=0.22'],
    install_requires=[
        'numpy', 'h5py', 'cython', 'psutil', 'scipy', 'joblib', 'scikit-learn',
        'yt', 'astropy'#,
        #'pygadgetreader @ git+https://github.com/dnarayanan/pygadgetreader'
    ],
    cmdclass={
        'sdist': sdist,
        'build_ext': build_ext,
        'build_py': build_py
    },
    ext_modules=cython_extensions,
)

