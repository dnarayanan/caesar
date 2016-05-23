import os

import yt
from yt.funcs import mylog

import caesar
from caesar.progen import progen_finder

class Snapshot(object):
    def __init__(self, snapdir, snapname, snapnum, extension='hdf5'):
        self.snapdir  = snapdir
        self.snapname = snapname
        self.snapnum  = snapnum
        self.snap     = '%s/%s%03d.%s' % (snapdir, snapname,
                                           snapnum, extension)

    def set_output_information(self, ds):
        if ds.cosmological_simulation == 0:
            time = 't%0.3f' % ds.current_time
        else:
            time = 'z%0.3f' % ds.current_redshift
                
        self.outdir   = '%s/Groups' % ds.fullpath
        self.outfile  = '%s/caesar_%04d_%s.hdf5' % (self.outdir,
                                                    self.snapnum,
                                                    time)

    def _make_output_dir(self):
        if not os.path.isdir(self.outdir):
            try:
                os.makedirs(self.outdir)
            except:
                pass
            
    def member_search(self, skipran, **kwargs):
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

        ds = None

def print_art():
    from caesar.main import VERSION
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

        
def run(snapdirs, snapnames, snapnums,
        progen=False, skipran=False, member_search=True,
        **kwargs):
    
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
            snaps.append(Snapshot(snapdir, snapnames, snapnum))
        
    if member_search:
        rank_snaps = snaps[rank::nprocs]
        for snap in rank_snaps:
            snap.member_search(skipran, **kwargs)

    if progen:
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
        
            progen_finder(obj_current, obj_progens,
                          snap_current, snap_progens)

if __name__ == '__main__':
    run()
