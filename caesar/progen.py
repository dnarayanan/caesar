
# Find progenitors or descendants.
# progen_finder() inputs snapshots and Caesar catalog, reads in particle IDs for specified type, runs find_progens() then write_progens()
# find_progens() can also be run stand-alone, if you have a list of particle ID and caesar objects. Returns progenitor and second-most massive progenitor.
# write_progens() inserts this info into the Caesar file.

import numpy as np
import h5py
from yt.funcs import mylog
from caesar.utils import memlog
from caesar.group import group_types
from joblib import Parallel, delayed
from scipy import stats


def find_progens(pid_current, pid_target, gid_current, gid_target, pid_hash, n_most=1, min_in_common=0.1, nproc=1):
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
        Find n_most most massive progenitors/descendants.  Current options are 1 or 2.
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
    mylog.info('Progen: Sorted %d target IDs, doing %d groups'%(len(pid_target),ngal_curr))

    # Loop over current objects to find progens for each
    if nproc>1:
        prog_index_tmp = Parallel(n_jobs=nproc)(delayed(find_target_group)(pid_current[pid_hash[ig]:pid_hash[ig+1]],pid_target,gid_target,min_in_common) for ig in range(ngal_curr))
        prog_index_tmp = np.array(prog_index_tmp,dtype=int)
        prog_index = np.array(prog_index_tmp.T[0],dtype=int)
        prog_index2 = np.array(prog_index_tmp.T[1],dtype=int)
    else:
        prog_index = np.zeros(ngal_curr,dtype=int)
        prog_index2 = np.zeros(ngal_curr,dtype=int)
        for ig in range(ngal_curr):
            prog_index[ig],prog_index2[ig] = find_target_group(pid_current[pid_hash[ig]:pid_hash[ig+1]],pid_target,gid_target,min_in_common)

    # Print some stats and return the indices
    #prog_none = prog_index[prog_index<0]
    #mylog.info('Progen: Most common counterpart %d appeared %d times, %d groups have no counterpart'%(stats.mode(prog_index[prog_index>=0])[0][0],stats.mode(prog_index[prog_index>=0])[1][0],len(prog_none)))
    #except:
    #    memlog('0 had no progenitors')

    if n_most == 1: return prog_index
    elif n_most == 2: return prog_index,prog_index2
    else:
        myinfo.warning('n_most=%d but must be 1 or 2; using 1'%n_most)
        return prog_index

# Find progenitor/descendant group having the most & second most particle ID's in common with pid_curr
def find_target_group(pid_curr,pid_targ,gid_targ,min_in_common):
    targ_ind = np.searchsorted(pid_targ,pid_curr) # bisection search to find closest ID in target
    targ_ind = np.where(targ_ind==len(pid_targ),len(pid_targ)-1,targ_ind)
    ig_matched = np.where(pid_targ[targ_ind]==pid_curr,gid_targ[targ_ind],-1)
    ig_matched = ig_matched[ig_matched>=0]
    unique, counts = np.unique(ig_matched,return_counts=True)
    #if len(pid_curr)>10000: print(len(ig_matched),len(ig_matched),unique[:10],counts[:10])
    if len(ig_matched)>int(min_in_common*len(pid_curr)):
        modestats = stats.mode(ig_matched) # find target galaxy id with most matches
        prog_index_ig = modestats[0][0]  # prog_index stores target galaxy numbers
        ig_matched = ig_matched[(ig_matched!=prog_index_ig)]  # remove the first-most common galaxy, recompute mode
    else: prog_index_ig = -1
    if len(ig_matched)>0:
        modestats = stats.mode(ig_matched) # find target galaxy id with second-most matches
        prog_index_ig2 = modestats[0][0]  # now we have the second progenitor
    else: prog_index_ig2 = -1
    return prog_index_ig,prog_index_ig2


def write_progens(obj, data, data_type, caesar_file, index_name):
    f = h5py.File(caesar_file,'r+')
    memlog('Writing %s info into %s'%(index_name,caesar_file))
    if check_if_progen_is_present(data_type, caesar_file, index_name):
        del f['%s_data/%s' % (data_type, index_name)]
    f.create_dataset('%s_data/%s' % (data_type, index_name), data=data, compression=1)
    f.close()
    return    

def check_if_progen_is_present(data_type, caesar_file, index_name):
    """Check CAESAR file for progen indexes."""
    f = h5py.File(caesar_file,'r')
    present = False
    if '%s_data/%s' % (data_type,index_name) in f: present = True
    f.close()
    return present

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
        Particle type in ptype_ints.  
    snap_dir : str
        Path where snapshot files are located; if None, uses obj.simulation.fullpath

    """
    # read in particle IDs
    from readgadget import readsnap
    if snap_dir is None:
        snapfile = obj.simulation.fullpath.decode('utf-8')+'/'+obj.simulation.basename.decode('utf-8')
    else:
        snapfile = snap_dir+'/'+obj.simulation.basename.decode('utf-8')
    all_pids = np.array(readsnap(snapfile,'pid',part_type),dtype=np.uint64)

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


def progen_finder(obj_current, obj_target, caesar_file, snap_dir=None, data_type='galaxy', part_type='star', overwrite=False, n_most=1, min_in_common=0.1, nproc=1):
    """Function to find the most massive progenitor of each CAESAR
    object in the previous snapshot.

    Parameters
    ----------
    obj_current : :class:`main.CAESAR`
        Will search for the progenitors of the objects in this object.
    obj_target: :class:`main.CAESAR`
        Looking for progenitors in this object.
    caesar_file : str
        Name (including path) of Caesar file associated with primary snapshot, where progen info will be written
    snap_dir : str
        Path where snapshot files are located; if None, uses obj.simulation.fullpath
    data_type : str
        'halo', 'galaxy', or 'cloud'
    part_type : str
        Particle type in ptype_ints.  Current options: 'gas', 'dm', 'dm2', 'star', 'bh'
    overwrite : bool
        Flag to overwrite existing progen data in Caesar object
    n_most : int 
        Find n_most most massive progenitors/descendants.  Stored as an array for each galaxy
    min_in_common : float 
        Require >this fraction of parts in common between object and progenitor to be a valid progenitor.
    nproc : int
        Number of cores for multiprocessing. 

    """

    if obj_current.simulation.redshift > obj_target.simulation.redshift:
        index_name = 'descend_'+part_type
    else:
        index_name = 'progen_'+part_type

    if check_if_progen_is_present(data_type, caesar_file, index_name) and not overwrite:
        mylog.warning('%s progen data already present as %s; skipping (set overwrite=True to overwrite)!' % (data_type,index_name))
        return

    ng_current, pid_current, gid_current, pid_hash = collect_group_IDs(obj_current, data_type, part_type, snap_dir)
    ng_target, pid_target, gid_target, _ = collect_group_IDs(obj_target, data_type, part_type, snap_dir)

    if ng_current == 0 or ng_target == 0:
        mylog.warning('No %d found in current caesar/target file (%d/%d) -- exiting progen'%(data_type,ng_current,ng_target))
        return

    prog_indexes = find_progens(pid_current, pid_target, gid_current, gid_target, pid_hash, n_most=n_most, min_in_common=min_in_common, nproc=nproc)

    #for i in range(5):
    #    print(i,prog_indexes[i],np.log10(obj_current.galaxies[i].masses['stellar']),np.log10(obj_target.galaxies[prog_indexes[i]].masses['stellar']))

    write_progens(obj_current, np.array(prog_indexes).T, data_type, caesar_file, index_name)

    return prog_indexes
