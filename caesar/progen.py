
# Find progenitors or descendants in simulation snapshots, based on the most number of particles in common of a specified type (part_type) within a specified dataset (data_type).
#
# run_progen() finds progenitors/descendants in specified or all snapshot pairs (or redshifts) in a directory. For each pair it runs progen_finder().
# progen_finder() inputs snapshots and Caesar catalog, reads in particle IDs for specified type, runs find_progens() then write_progens().
# find_progens() can also be run stand-alone, if you have a list of particle IDs and caesar objects. Returns progenitor and second-most massive progenitor.
#
# Written by Romeel Dave

import numpy as np
import h5py
import os
import caesar
from yt.funcs import mylog
from caesar.utils import memlog
from caesar.group import group_types
from joblib import Parallel, delayed
from scipy import stats

###################
# DRIVER ROUTINES #
###################

def run_progen(snapdirs, snapname, snapnums, prefix='caesar_', suffix='hdf5', **kwargs):
    """Function to run progenitor/descendant finder in specified snapshots (or redshifts) in a given directory.

    Parameters
    ----------
    snapdirs : str or list of str
        Full path of directory(s) where snapshots are located
    snapname : str
        Formatting of snapshot name excluding any integer numbers or file extensions; e.g. 'snap_N256L16_'
    snapnums : int or list of int
        Snapshot numbers over which to run progen.  Increasing order -> descendants; Decreasing -> progens.
    prefix : str
        Prefix for caesar filename; assumes these are in 'Groups' subdir
    suffix : str
        Filetype suffix for caesar filename
    kwargs : Passed to progen_finder()

    """

    from caesar.driver import Snapshot

    # Find existing snapshots in snapdirs
    if isinstance(snapdirs, str):
        snapdirs = [snapdirs]
    if isinstance(snapnums, int):
        snapnums = [int]

    snaps = []
    for snapdir in snapdirs:
        for snapnum in snapnums:
            snaps.append(Snapshot(snapdir, snapname, snapnum, suffix))
       
    verified_snaps = []
    missing_snaps  = []
    missing = ''
    for isnap,snap in enumerate(snaps):
        fullname = snap.snapdir + '/' + snap.snapname + '%.03d'%snap.snapnum + '.' + suffix
        if not os.path.isfile(fullname):
            missing_snaps.append(snap)
            missing = missing+'%d '%(snapnums[isnap])
            continue
        snap.outfile = caesar_filename(snap,prefix,suffix)
        if not os.path.isfile(snap.outfile):
            missing_snaps.append(snap)
            missing = missing+'%d '%(snapnums[isnap])
            continue
        f = h5py.File(snap.outfile,'r')
        if not '/halo_data' in f:
            missing_snaps.append(snap)
            missing = missing+'%d '%(snapnums[isnap])
            f.close()
            continue
        verified_snaps.append(snap)
        f.close()

    if len(missing_snaps) > 0:
        mylog.warning('Missing snapshot/caesar file, or no halo_data for: %s'%missing)

    # Collect pairs of snapshot names over which to run progen
    progen_pairs = []
    for i in range(0,len(verified_snaps)-1):
        progen_pairs.append((verified_snaps[i],verified_snaps[i+1]))

    # Loop over pairs, find progens
    for progen_pair in progen_pairs:
        snap_current = progen_pair[0]
        snap_progens = progen_pair[1]

        if snap_current.snapnum < snap_progens.snapnum: 
            mylog.info('Progen: Finding descendants of snap %d in snap %d'%(snap_current.snapnum,snap_progens.snapnum))
        else:
            mylog.info('Progen: Finding progenitors of snap %d in snap %d'%(snap_current.snapnum,snap_progens.snapnum))

        obj_current = caesar.load(caesar_filename(snap_current,prefix,suffix))
        obj_progens = caesar.load(caesar_filename(snap_progens,prefix,suffix))

        progen_finder(obj_current, obj_progens, caesar_filename(snap_current,prefix,suffix), snap_dir=snapdirs[0], **kwargs)


def progen_finder(obj_current, obj_target, caesar_file, snap_dir=None, data_type='galaxy', part_type='star', recompute=True, save=True, n_most=None, min_in_common=0.1, nproc=1):
    """Function to find the most massive progenitor of each Caesar object in obj_current
    in the previous snapshot.
    Returns list of progenitors in obj_target associated with objects in obj_current

    Parameters
    ----------
    obj_current : :class:`main.CAESAR`
        Will search for the progenitors of the objects in this object.
    obj_target: :class:`main.CAESAR`
        Looking for progenitors in this object.
    caesar_file : str
        Name (including path) of Caesar file associated with primary snapshot, where 
        progen info will be written 
    snap_dir : str
        Path where snapshot files are located; if None, uses obj.simulation.fullpath
    data_type : str
        'halo', 'galaxy', or 'cloud'
    part_type : str
        Particle type in ptype_ints.  Current options: 'gas', 'dm', 'dm2', 'star', 'bh'
    recompute : bool
        False = see if progen info exists in caesar_file and return, if not then compute
        True = always (re)compute progens
    save : bool
        True/False = write/do not write info to caesar_file
    n_most : int 
        Find n_most most massive progenitors/descendants.  Stored as a list for each galaxy.
        Default: None for all progenitors/descendants
    min_in_common : float 
        Require >this fraction of parts in common between object and progenitor to 
        be a valid progenitor.
    nproc : int
        Number of cores for multiprocessing. 

    """

    if obj_current.simulation.redshift > obj_target.simulation.redshift:
        index_name = 'descend_'+data_type+'_'+part_type
    else:
        index_name = 'progen_'+data_type+'_'+part_type

    if not recompute and check_if_progen_is_present(caesar_file, index_name):
        mylog.warning('%s data already present; returning data (set recompute=True to recompute)!' % (index_name))
        f = h5py.File(caesar_file,'r')
        prog_indexes = f['tree_data/%s'%index_name]
        return np.asarray(prog_indexes)

    ng_current, pid_current, gid_current, pid_hash = collect_group_IDs(obj_current, data_type, part_type, snap_dir)
    ng_target, pid_target, gid_target, _ = collect_group_IDs(obj_target, data_type, part_type, snap_dir)

    'gas', 'dm', 'dm2', 'star', 'bh'
    if part_type == 'gas':
        npart_target = np.array([len(_g.glist) for _g in obj_target.galaxies])
    elif part_type == 'star':
        npart_target = np.array([len(_g.slist) for _g in obj_target.galaxies])
    elif part_type == 'bh':
        npart_target = np.array([len(_g.bhlist) for _g in obj_target.galaxies])
    elif part_type in ['dm','dm2']:
        npart_target = np.array([len(_g.dmlist) for _g in obj_target.galaxies])


    if ng_current == 0 or ng_target == 0:
        mylog.warning('No %s found in current caesar/target file (%d/%d) -- exiting progen_finder'%(data_type,ng_current,ng_target))
        return None

    prog_indexes = find_progens(pid_current, pid_target, gid_current, gid_target, pid_hash, 
                                npart_target, n_most=n_most, min_in_common=min_in_common, nproc=nproc)

    if save:
        if n_most is not None:
            write_progens(obj_current, np.array(prog_indexes).T, caesar_file, index_name, obj_target.simulation.redshift)
        else:
            write_progens(obj_current, prog_indexes, caesar_file, index_name, obj_target.simulation.redshift)

    return prog_indexes


def find_progens(pid_current, pid_target, gid_current, gid_target, pid_hash, npart_target, n_most=None, min_in_common=0.1, nproc=1):
    """Find most massive and second most massive progenitor/descendants.
    
    Parameters
    ----------
    pids_current : np.ndarray
       particle IDs from the current snapshot.
    pids_target : np.ndarray
       particle IDs from the previous/next snapshot.
    gids_current : np.ndarray
       group IDs from the current snapshot.
    gids_target : np.ndarray
       group IDs from the previous/next snapshot.
    pid_hash : np.ndarray
       indexes for the start of each group in pids_current
    n_most : int 
        Find n_most most massive progenitors/descendants, None for all.
    min_in_common : float 
        Require >this fraction of parts in common between object and progenitor to be a valid progenitor.
    nproc : int
        Number of cores for multiprocessing. Note that this doesn't help much since most of the time is spent in sorting.

    """

    # Sort the progenitor IDs and object numbers for faster searching
    isort_target = np.argsort(pid_target)
    pid_target = pid_target[isort_target]  # target particles' IDs
    gid_target = gid_target[isort_target]  # galaxy IDs for the target particles
    ngal_curr = len(pid_hash)-1  # number of galaxies to find progens/descendants for
    memlog('Progen doing %d groups (nproc=%d)'%(ngal_curr,nproc))

    # Loop over current objects to find progens for each
    if nproc>1:
        prog_index_tmp = Parallel(n_jobs=nproc)(delayed(_find_target_group)\
                   (pid_current[pid_hash[ig]:pid_hash[ig+1]],pid_target,
                    gid_target,npart_target,min_in_common,return_N=n_most) \
                  for ig in range(ngal_curr))
        if n_most is not None:
            prog_index = np.array(prog_index_tmp,dtype=np.int32)
        else:
            prog_index = np.array(prog_index_tmp,dtype=object)
    else:
        if n_most is not None:
            prog_index = np.zeros((ngal_curr,n_most),dtype=np.int32)
        else:
            prog_index = np.zeros(ngal_curr,dtype=object)
        for ig in range(ngal_curr):
            prog_index[ig] = _find_target_group(pid_current[pid_hash[ig]:pid_hash[ig+1]],pid_target,
                                                gid_target,npart_target,min_in_common,return_N=n_most)

   
    return prog_index

    # Print some stats and return the indices
    #prog_none = prog_index[prog_index<0]
    #mylog.info('Progen: Most common counterpart %d appeared %d times, %d groups have no counterpart'%(stats.mode(prog_index[prog_index>=0])[0][0],stats.mode(prog_index[prog_index>=0])[1][0],len(prog_none)))
    #except:
    #    mylog.warning('0 had no progenitors')


# Find progenitor/descendant group having the most & second most particle ID's in common with pid_curr
def _find_target_group(pid_curr,pid_targ,gid_targ,npart_targ,min_in_common,return_N=10):
    targ_ind = np.searchsorted(pid_targ,pid_curr) # bisection search to find closest ID in target
    targ_ind = np.where(targ_ind==len(pid_targ),len(pid_targ)-1,targ_ind)
    ig_matched = np.where(pid_targ[targ_ind]==pid_curr,gid_targ[targ_ind],-1)
    ig_matched = ig_matched[ig_matched>=0]
    unique, counts = np.unique(ig_matched,return_counts=True)
    
    _cmask = np.argsort(counts)[::-1]
    match_frac = counts / npart_targ[unique] # ---- fraction of particles from target in current
    out = np.ones(return_N, dtype=int) * -1 # ---- initialise output array (-1 default, no match)  
    _matched = unique[_cmask[match_frac > min_in_common]] # ---- matching targets
    
    # ---- populate output array
    if return_N is not None:
        if len(_matched) > return_N: 
            out = _matched[:return_N]
        else: 
            out[:len(_matched)] = _matched
    else: #return all matched galaxies as a list 
        if len(_matched) > 0:
            out = _matched.tolist()
        else:
            out = []
    return out


def collect_group_IDs(obj, data_type, part_type, snap_dir):
    """Collates list of particle and associated group IDs for all specified objects.
    Returns particle and group ID lists, and a hash list of indexes for particle IDs 
    corresponding to the starting index of each group.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Caesar object for which to collect group IDs
    data_type : str
        'halo', 'galaxy', or 'cloud'
    part_type : str
        Particle type
    snap_dir : str
        Path where snapshot files are located; if None, uses obj.simulation.fullpath

    """
    # read in particle IDs
    from readgadget import readsnap
    if snap_dir is None:
        #snapfile = obj.simulation.fullpath.decode('utf-8')+'/'+obj.simulation.basename.decode('utf-8')
        snapfile = obj.simulation.fullpath.decode('utf-8')+'/'+obj.simulation.basename.decode('utf-8')
    else:
        snapfile = snap_dir+'/'+obj.simulation.basename
    all_pids = np.array(readsnap(snapfile,'pid',part_type,suppress=1),dtype=np.uint64)

    from caesar.fubar_halo import plist_dict
    if data_type == 'halo':
        part_list = 'obj.halos[i].%s'%plist_dict[part_type]
        ngroups = len(obj.halos)
    elif data_type == 'galaxy':
        part_list = 'obj.galaxies[i].%s'%plist_dict[part_type]
        ngroups = len(obj.galaxies)
    elif data_type == 'cloud':
        part_list = 'obj.clouds[i].%s'%plist_dict[part_type]
        ngroups = len(obj.clouds)

    # count number of total particles in groups
    npart = 0
    for i in range(ngroups):
        mylist = eval(part_list)
        npart += len(mylist)

    # fill particle and group ID lists
    pids = np.zeros(npart,dtype=np.int64)
    gids = np.zeros(npart,dtype=np.int32)
    pid_hash = np.zeros(ngroups,dtype=np.int64)
    count = 0
    for i in range(ngroups):
        mylist = eval(part_list)
        pids[count:count+len(mylist)] = all_pids[mylist]
        gids[count:count+len(mylist)] = np.full(len(mylist),i)
        pid_hash[i] = count
        count += len(mylist)
    pid_hash = np.append(pid_hash,npart+1)

    return ngroups, pids, gids, pid_hash



################
# I/O ROUTINES #
################

def write_progens(obj, data, caesar_file, index_name, redshift):
    f = h5py.File(caesar_file,'r+')
    if check_if_progen_is_present(caesar_file, index_name):
        del f['tree_data/%s' % (index_name)]
        mylog.info('Overwriting %s info in tree_data' % (index_name))
    else:
        mylog.info('Writing %s info into tree_data'%(index_name))
    try:
        tree = f.create_group('tree_data')
    except:
        tree = f['tree_data']
        
    if data.dtype == np.int32:
        progens = tree.create_dataset('%s' % (index_name), data=data, compression=1)
    else:
        try:
            progens = f.create_group('tree_data/%s' % (index_name))
        except:
            progens = f['tree_data/%s' % (index_name)]
        for index, element in enumerate(data):
            progens.create_dataset("%d"%index, data=np.array(element, dtype=np.int32), compression=1)
    tree.attrs[('z_'+index_name).encode('utf8')] = redshift
    f.close()
    return    

def check_if_progen_is_present(caesar_file, index_name):
    """Check CAESAR file for progen indexes.

    Parameters
    ----------
    caesar_file : str
        Name (including path) of Caesar file with tree_data
    index_name : str
        Name of progen index to get redshift for (e.g. 'progen_galaxy_star')
    """
    if not os.path.isfile(caesar_file):
        mylog.warning('caesar_file %s not found')
        return False
    f = h5py.File(caesar_file,'r')
    present = False
    if 'tree_data/%s' % (index_name) in f: present = True
    f.close()
    return present


def get_progen_redshift(caesar_file, index_name):
    """Returns redshift of progenitors/descendants currently stored in tree_data.
    Returns -1 (with warning) if no tree_data is found.

    Parameters
    ----------
    caesar_file : str
        Name (including path) of Caesar file with tree_data
    index_name : str
        Name of progen index to get redshift for (e.g. 'progen_galaxy_star')
    """

    f = h5py.File(caesar_file,'r')
    try:
        tree = f['tree_data']
    except:
        return -1
        mylog.warning('Progen data %s does not exist in %s'%(index_name,caesar_file))
    z = tree.attrs['z_'+index_name]
    return z

def wipe_progen_info(caesar_file,index_name=None):
    """Remove all progenitor/descendant info from Caesar file.

    Parameters
    ----------
    caesar_file : str
        Name (including path) of Caesar file with tree_data
    index_name : str (optional)
        Name (or substring) of progen index to remove (e.g. 'progen_galaxy_star').
        If not provided, removes *all* progen/descend info
    """

    f = h5py.File(caesar_file,'r+')
    for dataset in f.keys():
        for name in f[dataset].keys():
            if index_name is None:
                if 'progen' in name or 'descend' in name:
                    mylog.info('Deleting %s from %s in %s'%(name,dataset,caesar_file))
                    del f[dataset][name]
            else:
                if index_name in name:
                    mylog.info('Deleting %s from %s in %s'%(name,dataset,caesar_file))
                    del f[dataset][name]

def caesar_filename(snap,prefix,extension): 
    """return full Caesar filename including filetype extension for given Snapshot object."""
    return snap.snapdir + '/Groups/' + prefix + snap.snapname.replace('snap_','') + '%.03d'%snap.snapnum + '.' + extension

def z_to_snap(redshift, snaplist_file='Simba', mode='closest'):
    """Finds snapshot number and snapshot redshift close to input redshift.

    Parameters
    ----------
    redshift : float
        Redshift you want to find snapshot for
    snaplist_file : str
        Name (including path) of Caesar file with a list of expansion factors (in 
        ascending order) at which snapshots are output.  This is the same file as 
        used when running a Gizmo/Gadget simulation.  
        'Simba' returns the value for the default Simba simulation snapshot list.
    mode : str
        'closest' finds closest one in redshift 
        'higher'/'upper'/'above' finds the closest output >= redshift
        'lower'/'below' finds the closest output <= redshift.
    """

    if snaplist_file.lower() == 'simba':
         aex_output = np.array([0.010000,0.048194,0.050227,0.052301,0.054418,0.056576,0.058776,0.061018,0.063301,0.065627,0.067994,0.070403,0.072854,0.075347,0.077882,0.080459,0.083077,0.085738,0.088440,0.091185,0.093971,0.096800,0.099670,0.102583,0.105538,0.108535,0.111574,0.114656,0.117780,0.120946,0.124155,0.127406,0.130700,0.134036,0.137415,0.140836,0.144301,0.147808,0.151359,0.154952,0.158589,0.162268,0.165992,0.169758,0.173569,0.177423,0.181321,0.185263,0.189249,0.193280,0.197355,0.201475,0.205640,0.209850,0.214105,0.218406,0.222753,0.227145,0.231584,0.236070,0.240602,0.245181,0.249808,0.254483,0.259206,0.263977,0.268797,0.273667,0.278586,0.283556,0.288576,0.293647,0.298769,0.303944,0.309172,0.314452,0.319787,0.325176,0.330620,0.336120,0.341676,0.347290,0.352962,0.358692,0.364482,0.370333,0.376246,0.382220,0.388259,0.394361,0.400530,0.406765,0.413068,0.419440,0.425883,0.432397,0.438985,0.445648,0.452386,0.459203,0.466098,0.473075,0.480135,0.487280,0.494511,0.501832,0.509243,0.516748,0.524347,0.532045,0.539843,0.547744,0.555751,0.563867,0.572094,0.580435,0.588895,0.597476,0.606181,0.615015,0.623981,0.633084,0.642327,0.651715,0.661253,0.670945,0.680796,0.690812,0.700998,0.711360,0.721904,0.732636,0.743563,0.754692,0.766030,0.777585,0.789364,0.801377,0.813632,0.826138,0.838906,0.851945,0.865267,0.878883,0.892806,0.907047,0.921620,0.936540,0.951821,0.967481,0.983534,1.000000])

    else:
        aex_output = np.loadtxt(snaplist_file,usecols=[0])

    #mylog.info('Imported %d aex values from %s'%(len(aex_output),snaplist_file))

    z_output = 1./aex_output - 1.
    idx = (np.abs(z_output-redshift)).argmin() # index of closest redshift

    if mode == 'higher' or mode == 'upper' or mode == 'above':
        if z_output[idx] < redshift: 
            idx = max(idx-1,0)
    elif mode == 'lower' or mode == 'below':
        if z_output[idx] > redshift: 
            idx = min(idx+1,len(z_output)-1)

    return idx,z_output[idx]


