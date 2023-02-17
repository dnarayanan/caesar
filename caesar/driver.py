import os
import sys

import yt
from yt.funcs import mylog

import caesar
from caesar.progen import progen_finder

class Snapshot(object):
    """Class for tracking paths and data for simulation snapshots.

    Parameters
    ----------
    snapdir : str
        Path to snapshot
    snapname : str
        Name of snapshot minus number and extension
    snapnum : int
        Snapshot number
    extension : str, optional
        File extension of your snapshot, 'hdf5' by default.

    Notes
    -----
    This class attempts to concat strings to form a full path to your
    simulation snapshot in the following manner:
    
    >>> '%s/%s%03d.%s' % (snapdir, snapname, snapnum, extension)

    """    
    def __init__(self, snapdir, snapname, snapnum, extension):
        self.snapdir  = snapdir
        self.snapname = snapname
        self.snapnum  = snapnum
        self.snap     = '%s/%s%03d.%s' % (snapdir, snapname, snapnum, extension)

    def set_output_information(self, ds, snapdirformat, prefix='caesar_', suffix='hdf5'):
        """Set the name of the CAESAR output file."""
        if ds.cosmological_simulation == 0:
            time = 't%0.3f' % ds.current_time
        else:
            time = 'z%0.3f' % ds.current_redshift
            

        self.outdir   = '%s/Groups' % ds.fullpath
        if snapdirformat==False:
            self.outfile  = '%s/%s%s%03d.%s' % (self.outdir, prefix, self.snapname.replace('snap_',''), self.snapnum,suffix)
        else:
            #strip off the annoying snapdir so that all the caesar
            #files can be in one place.  if you don't like this,
            #remove this line :) i get it...it's janky code.  but you
            #know.
            snapdir_str = 'snapdir_'+str(self.snapnum).zfill(3)+"/"
            self.outdir=self.outdir.replace(snapdir_str+"Groups","/Groups")
            self.outfile  = '%s/%s%s%03d.%s' % (self.outdir, prefix,self.snapname.replace(snapdir_str,''), self.snapnum,suffix)

    def _make_output_dir(self):
        """If output directory is not present, create it."""
        if not os.path.isdir(self.outdir):
            try:
                os.makedirs(self.outdir)
            except:
                print("\n\n\n\n\n")
                print(self.outdir)
                pass
            
    def member_search(self, skipran, snapdirformat, **kwargs):
        """Perform the member_search() method on this snapshot."""
        if not os.path.isfile(self.snap):
            mylog.warning('%s NOT found, skipping' % self.snap)
            return
        
        ds = yt.load(self.snap)
        self.set_output_information(ds,snapdirformat)

        if os.path.isfile(self.outfile) and skipran:
            mylog.warning('%s FOUND, skipping' % self.outfile)
            return

        self._make_output_dir()

        obj = caesar.CAESAR(ds)
        obj.member_search(snapdirformat,**kwargs)
        obj.save(self.outfile)

        obj = None
        ds  = None

def print_art():
    """Print some ascii art."""
    from caesar.__version__ import VERSION
    copywrite = '    (C) 2016 Robert Thompson'
    version   = '    Version %s' % VERSION

    art = """\
      _____          ______  _____         _____  
     / ____|   /\\   |  ____|/ ____|  /\\   |  __ \\ 
    | |       /  \\  | |__  | (___   /  \\  | |__) |
    | |      / /\\ \\ |  __|  \\___ \\ / /\\ \\ |  _  / 
    | |____ / ____ \\| |____ ____) / ____ \\| | \\ \\ 
     \\_____/_/    \\_\\______|_____/_/    \\_\\_|  \\_\
    """

    print('\n%s\n%s\n%s\n' % (art, copywrite, version))

        
def drive(snapdirs, snapname, snapnums, progen=False, skipran=False,
          member_search=True, extension='hdf5', caesar_prefix='caesar_', snapdirformat=False, **kwargs):
    """Driver function for running ``CAESAR`` on multiple snapshots.

    Can utilize mpi4py to run analysis in parallel given that ``MPI`` 
    and ``mpi4py`` is correctly installed.  To do this you must create
    a script similar to the example below, then execute it via:

    >>> mpirun -np 8 python my_script.py

    Parameters
    ----------
    snapdirs : str or list
        A path to your snapshot directory, or a list of paths to your
        snapshot directories.
    snapname : str
        Formatting of your snapshot name disregarding any integer 
        numbers or file extensions; for example: ``snap_N256L16_``
    snapnums : int or list or array
        A single integer, a list of integers, or an array of integers.
        These are the snapshot numbers you would like to run CAESAR
        on.
    
    snapdirformat : boolean 
        If set, then looks for files in formati snapdir_001/snapshot_001.0.hdf5 
    instead of direct snapshots (such as snapshot_001.hdf5).  default is to False.


    progen : boolean, optional
        Perform most massive progenitor search.  Defaults to False.
    skipran : boolean, optional
        Skip running member_search() if CAESAR outputs are already
        present.  Defaults to False.
    member_search : boolean, optional
        Perform the member_search() method on each snapshot.  Defaults
        to True.  This is useful to set to False if you want to just
        perform progen for instance.
    extension : str, optional
        Specify your snapshot file extension.  Defaults to `hdf5`
    prefix : str, optional
        Specify prefix for caesar filename (replaces 'snap_')
    unbind_halos : boolean, optional
        Unbind halos?  Defaults to False
    unbind_galaxies : boolean, optional
        Unbind galaxies?  Defaults to False
    b_halo : float, optional
        Quantity used in the linking length (LL) for halos.
        LL = mean_interparticle_separation * b_halo.  Defaults to 
        ``b_halo = 0.2``.
    b_galaxy : float, optional
        Quantity used in the linking length (LL) for galaxies.
        LL = mean_interparticle_separation * b_galaxy.  Defaults 
        to ``b_galaxy = b_halo * 0.2``.
    ll_cloud: float, optional
        Linking length in comoving kpc (kpccm_ for clouds.  Defaults
        to same linking length as used for galaxies.
    fofclouds: boolean, optional
        Sets whether or not we run 3D FOF for clouds. Default is that this is not run
        as this isn't the typical use case for Caesar, and slows things down a bit 
    fof6d: boolean, optional 
        Sets whether or not we do 6D FOF for galaxies.  if not set, the default is to do 
        normal 3D FOF for galaxies.
    fof6d_LL_factor: float, optional
        Sets linking length for fof6d
    fof6d_mingrp: float, optional
        Sets minimum group size for fof6d
    fof6d_velLL: float, optional
        Sets linking length for velocity in fof6d
    nproc: int, optional
        Sets number of processors for fof6d 
    blackholes : boolean, optional
        Indicate if blackholes are present in your simulation.  
        This must be toggled on manually as there is no clear 
        cut way to determine if PartType5 is a low-res particle, 
        or a black hole.
    lowres : list, optional
        If you are running ``CAESAR`` on a Gadget/GIZMO zoom
        simulation in HDF5 format, you may want to check each halo for 
        low-resolution contamination.  By passing in a list of 
        particle types (ex. [2,3,5]) we will check ALL objects for 
        contamination and add the ``contamination`` attribute to all 
        objects.  Search distance defaults to 2.5x radii['total'].

    Examples
    --------
    >>> import numpy as np
    >>> snapdir  = '/Users/bob/Research/N256L16/some_sim'
    >>> snapname = 'snap_N256L16_'
    >>> snapnums = np.arange(0,86)
    >>>
    >>> import caesar
    >>> caesar.drive(snapdir, snapname, snapnums, skipran=False, progen=True)

    """
    
    if isinstance(snapdirs, str):
        snapdirs = [snapdirs]
    
    if isinstance(snapnums, int):
        snapnums = [int]
    
    using_mpi = False
    try:
        from mpi4py import MPI
        comm   = MPI.COMM_WORLD
        nprocs = comm.Get_size()
        rank   = comm.Get_rank()
        using_mpi = True
    except:
        nprocs = 1
        rank   = 0

    
    if rank == 0: print_art()

    snaps = []

    if snapdirformat == False:
        for snapdir in snapdirs:
            for snapnum in snapnums:
                snaps.append(Snapshot(snapdir, snapname, snapnum, extension))
    else:
        for snapdir in snapdirs:
            for snapnum in snapnums:
                tempsnapname = 'snapdir_'+str(snapnum).zfill(3)+'/'+snapname
                tempext = '0.'+extension
                snaps.append(Snapshot(snapdir, tempsnapname, snapnum, tempext))

    if member_search:
        rank_snaps = snaps[rank::nprocs]
        for snap in rank_snaps:
            snap.member_search(skipran, snapdirformat, **kwargs)

    if progen:
        caesar.progen.run_progen(snapdirs, snapname, snapnums, prefix=caesar_prefix, suffix=extension, **kwargs)

if __name__ == '__main__':
    print_art()
