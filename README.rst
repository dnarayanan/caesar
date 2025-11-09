.. image:: https://readthedocs.org/projects/caesar/badge/?version=latest
   :target: http://caesar.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

CAESAR
======

CAESAR is a python library for analyzing the outputs from
cosmological simulations.  The general idea is to identify halos and
galaxies within the simulation, then store information about
them in an intuitive, portable, and easy to access manner.  CAESAR
builds upon the `yt <http://yt-project.org/>`_ package enabling it to
read a `multitude of different simulation types
<http://yt-project.org/doc/reference/code_support.html>`_, and attach
`meaningful units
<http://yt-project.org/doc/analyzing/units/index.html>`_ to object
attributes.

Further information about CAESAR (including installation and usage) please visit:

http://caesar.readthedocs.org

VERSION 0.2b0 changelog is detailed here:
https://www.overleaf.com/read/tfpfstktkrjm

For the problem of "ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject", that is due to the different version of numpy has been used between installing and common use.
You need to do:
'''
export CFLAGS="-I$(python -c 'import numpy; print(numpy.get_include())')"
pip install .
'''
