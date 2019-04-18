import os

import yt
from yt.funcs import mylog

import caesar
from caesar.progen import progen_finder
from caesar.progen_rad import *
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
        self.snap     = '%s/%s%03d.%s' % (snapdir, snapname,
                                           snapnum, extension)

    def set_output_information(self, ds):
        """Set the name of the CAESAR output file."""
        if ds.cosmological_simulation == 0:
            time = 't%0.3f' % ds.current_time
        else:
            time = 'z%0.3f' % ds.current_redshift
                
        self.outdir   = '%s/Groups' % ds.fullpath
        self.outfile  = '%s/caesar_%04d_%s.hdf5' % (self.outdir,
                                                    self.snapnum,
                                                    time)

    def _make_output_dir(self):
        """If output directory is not present, create it."""
        if not os.path.isdir(self.outdir):
            try:
                os.makedirs(self.outdir)
            except:
                pass
            
    def member_search(self, skipran, **kwargs):
        """Perform the member_search() method on this snapshot."""
        if not os.path.isfile(self.snap):
            mylog.warning('%s NOT found, skipping' % self.snap)
            return
        
        ds = yt.load(self.snap)
        self.set_output_information(ds)

        if os.path.isfile(self.outfile) and skipran:
            mylog.warning('%s FOUND, skipping' % self.outfile)
            return

        self._make_output_dir()

        obj = caesar.CAESAR(ds)
        obj.member_search(**kwargs)
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
     / ____|   /\   |  ____|/ ____|  /\   |  __ \ 
    | |       /  \  | |__  | (___   /  \  | |__) |
    | |      / /\ \ |  __|  \___ \ / /\ \ |  _  / 
    | |____ / ____ \| |____ ____) / ____ \| | \ \ 
     \_____/_/    \_\______|_____/_/    \_\_|  \_\
    """

    print('\n%s\n%s\n%s\n' % (art, copywrite, version))

        
def drive(snapdirs, snapname, snapnums, progen=False, progen_rad = False, skipran=False,
          member_search=True, extension='hdf5', **kwargs):
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
        Sets number of processors for fof6d and progen_rad
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
    for snapdir in snapdirs:
        for snapnum in snapnums:
            snaps.append(Snapshot(snapdir, snapname, snapnum, extension))
        
    if member_search:
        rank_snaps = snaps[rank::nprocs]
        for snap in rank_snaps:
            snap.member_search(skipran, **kwargs)

    if (progen == True) and (progen_rad == True):
        print('You can only set progen or progen_rad as True; exiting')
        sys.exit()

    if progen or progen_rad:
        if using_mpi:
            comm.Barrier()

        verified_snaps = []
        missing_snaps  = []
        for snap in snaps:
            if not hasattr(snap, 'outfile'):
                ds = yt.load(snap.snap)
                snap.set_output_information(ds)
            if os.path.isfile(snap.outfile):
                verified_snaps.append(snap)
            else:
                missing_snaps.append(snap)

        if len(missing_snaps) > 0:
            mylog.warning('Missing the following CAESAR files:')
            for snap in missing_snaps:
                mylog.warning(snap.outfile)

        progen_pairs = []
        for i in reversed(range(1,len(verified_snaps))):
            progen_pairs.append((verified_snaps[i],verified_snaps[i-1]))

        rank_progen_pairs = progen_pairs[rank::nprocs]
        for progen_pair in rank_progen_pairs:
            snap_current = progen_pair[0]
            snap_progens = progen_pair[1]

            ds_current = yt.load(snap_current.snap)
            ds_progens = yt.load(snap_progens.snap)

            snap_current.set_output_information(ds_current)
            snap_progens.set_output_information(ds_progens)

            obj_current = caesar.load(snap_current.outfile)
            obj_progens = caesar.load(snap_progens.outfile)

            if progen:
                progen_finder(obj_current, obj_progens,
                              snap_current, snap_progens)
                
            if progen_rad == True:
                #temporary catch to kill off any sims trying to do progen_Rad with clouds till we fix that
                if 'fofclouds' in kwargs:
                    raise Exception('Cannot call fofclouds and progen_rad - exiting now')
                else:
                    run_progen_rad(obj_current,obj_progens,snap_current,snap_progens)
if __name__ == '__main__':
    print_art()
