# setup.py
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext

# A minimal build_ext to inject numpy's include path
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        from six.moves import builtins
        builtins.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

# Platform-specific compile arguments
if sys.platform == 'darwin':
    compile_arg = ""
    link_arg = ""
else:
    compile_arg = "-fopenmp"
    link_arg = '-fopenmp'

cython_extensions = [
    Extension(
        'caesar.group_funcs',
        sources=['caesar/group_funcs/group_funcs.pyx'],
        extra_compile_args=[compile_arg],
        extra_link_args=[link_arg]),
    Extension(
        'caesar.hydrogen_mass_calc',
        sources=['caesar/hydrogen_mass_calc/hydrogen_mass_calc.pyx'],
        extra_compile_args=[compile_arg],
        extra_link_args=[link_arg]),
    Extension(
        'caesar.cyloser',
        sources=['caesar/pyloser/cyloser.pyx'],
        extra_compile_args=[compile_arg],
        extra_link_args=[link_arg])
]

# The main setup call is now very simple
setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=cython_extensions,
)
