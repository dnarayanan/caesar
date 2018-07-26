from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.sdist import sdist as _sdist
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.build_ext import build_ext as _build_ext
import os
import sys

sys.path.insert(0,'caesar')
from __version__ import VERSION

def get_mercurial_changeset_id(target_dir):
    '''
    Returns changeset and branch using hglib
    '''
    try:
        import hglib
    except ImportError:
        return None
    try:
        with hglib.open(target_dir) as repo:
            changeset = repo.identify(
                id=True, branch=True).strip().decode('utf8')
    except hglib.error.ServerError:
        return None
    return changeset

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

        # write out __hg_version__ file
        if not hasattr(self, 'hgid'):
            src_dir    = os.getcwd()
            target_dir = os.path.join(src_dir, 'caesar')
            changeset  = get_mercurial_changeset_id(src_dir)
            with open(os.path.join(target_dir, '__hg_version__.py'), 'w') as f:
                f.write("hg_version = '%s'\n" % changeset)
            self.hgid = changeset
            
class sdist(_sdist):
    # subclass setuptools source distribution builder to ensure cython
    # generated C files are included in source distribution.
    # See http://stackoverflow.com/a/18418524/1382869
    def run(self):
        # Make sure the compiled Cython files in the distribution are up-to-date
        from Cython.Build import cythonize
        cythonize(cython_extensions)
        _sdist.run(self)        
            
cython_extensions = [
    Extension('caesar.group_funcs',
              ['caesar/group_funcs/group_funcs.pyx']),
    Extension('caesar.hydrogen_mass_calc',
              ['caesar/hydrogen_mass_calc/hydrogen_mass_calc.pyx'])
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
    install_requires=['numpy','h5py','cython',],
    cmdclass={'sdist':sdist, 'build_ext':build_ext, 'build_py':build_py},
    ext_modules=cython_extensions,
)

