import numpy as np
import h5py

from yt.funcs import mylog

from caesar.group import group_types

DEBUG = False
progen_index_name = 'progen_index'

def find_progens(pids_current, pids_progens,
                 list_current, list_progens,
                 nobjs_current):
    """Primary most massive progenitor funciton.
    
    Parameters
    ----------
    pids_current : np.ndarray
       Particle IDs from the current snapshot.
    pids_progens : np.ndarray
       Particle IDs from the previous snapshot.
    list_current : list
        Group indexes for the current snapshot
    list_progens : list
        Group indexes fro the previous snapshot
    nobjs_current : int
        Number of objects we are searching for progenitors.

    """
    ## number of particles we are comparing against
    nids_progens = len(pids_progens)

    ## get the sorted indexes for the PIDs
    sorted_pid_indexes_current = np.argsort(pids_current)
    sorted_pid_indexes_progens = np.argsort(pids_progens)

    ## set a list of lists that we will populate with previous indexes
    current_obj_plists = [[] for x in range(nobjs_current)]

    ## cycle through all current particles
    j = 0
    for i in range(0, len(pids_current)):
        ## get sorted values
        index_current = sorted_pid_indexes_current[i]
        index_progens = sorted_pid_indexes_progens[j]

        pid_current = pids_current[index_current]
        pid_progens = pids_progens[index_progens]

        ## compare current IDs to previous IDs until
        ## a match is found; since the lists are sorted
        ## this should go rather quickly
        if pid_current == pid_progens:

            ## determine current and previous indexes
            current_obj_index  = list_current[index_current]
            previous_obj_index = list_progens[index_progens]

            ## if particle was assigned an object in the previous
            ## set of objects, record that value
            if previous_obj_index > -1:
                current_obj_plists[current_obj_index].append(previous_obj_index)
            ## match was found, iterate j
            if j+1 < nids_progens:
                j += 1

    ## set default previous object indexes to -1
    prev_obj_indexes   = np.full(nobjs_current, -1, dtype=np.int32)

    for i in range(0, nobjs_current):
        if len(current_obj_plists[i]) > 0:
            sorted_indexes = np.argsort(np.bincount(np.asarray(current_obj_plists[i])))
            index = sorted_indexes[-1]

            prev_obj_indexes[i] = index

    return prev_obj_indexes

def write_progen_data(obj,data,data_type, outfile,attribute_name=progen_index_name):
    """Write progen indexes to disk."""
    f = h5py.File(outfile)
    f.create_dataset('%s_data/%s' % (data_type, attribute_name), data=data, compression=1)
    f.close()

def rewrite_progen_data(obj,data,data_type, outfile,attribute_name=progen_index_name):
   """Write progen indexes to disk."""
   f = h5py.File(outfile)
   fd = f['%s_data/%s' % (data_type, attribute_name)]
   fd[...] = data
   f.close()


def check_if_progen_is_present(data_type, outfile,attribute_name =progen_index_name):
    """Check CAESAR file for progen indexes."""
    f = h5py.File(outfile)
    present = False
    if '%s_data/%s' % (data_type,attribute_name) in f:
        if DEBUG:
            mylog.warning('removing "%s_data/%s" from dataset (DEBUG=True)' % (data_type,attribute_name))
            del f['%s_data/%s' % (data_type, attribute_name)]
        else:
            present = True
    f.close()
    return present



def progen_finder(obj_current, obj_progens, snap_current, snap_progens):
    """Function to find the most massive progenitor of each CAESAR
    object in the previous snapshot.

    Parameters
    ----------
    obj_current : :class:`main.CAESAR`
        Will search for the progenitors of the objects in this object.
    obj_progens : :class:`main.CAESAR`
        Looking for progenitors in this object.
    snap_current : str
        Name (including path) of the primary snapshot
    snap_progens : yt dataset
        Name (including path) of the secondary snapshot

    """
    try:
        from pygadgetreader import readsnap
    except:
        raise Exception('Please install pyGadgetReader for quick PID reads!\n' \
                        'https://bitbucket.org/rthompson/pygadgetreader')

    caesar_file = snap_current.outfile

    ## halos (use DM)
    data_type  = 'halo'
    if obj_current.nhalos == 0 or obj_progens.nhalos == 0:
        mylog.warning('0 %s found!  skipping progen' % group_types[data_type])
    elif not check_if_progen_is_present(data_type, caesar_file):
        nobjs        = obj_current.nhalos
        PID_current  = readsnap(snap_current.snap, 'pid', 1)
        PID_progens  = readsnap(snap_progens.snap, 'pid', 1)
        list_current = obj_current.global_particle_lists.halo_dmlist
        list_progens = obj_progens.global_particle_lists.halo_dmlist

        prev_indexes = find_progens(PID_current, PID_progens,
                                    list_current, list_progens,
                                    nobjs)
        write_progen_data(obj_current, prev_indexes,
                          data_type, caesar_file)
    else:
        mylog.warning('%s progen data already present, skipping!' % data_type)

        
    if not obj_current.simulation.baryons_present:
        return

    ## galaxies (use stars)
    data_type = 'galaxy'
    if obj_current.ngalaxies == 0 or obj_progens.ngalaxies == 0:
        mylog.warning('0 %s found! skipping progen' % group_types[data_type])
    elif not check_if_progen_is_present(data_type, caesar_file):
        nobjs        = obj_current.ngalaxies
        PID_current  = readsnap(snap_current.snap, 'pid', 4)
        PID_progens  = readsnap(snap_progens.snap, 'pid', 4)
        list_current = obj_current.global_particle_lists.galaxy_slist
        list_progens = obj_progens.global_particle_lists.galaxy_slist

        prev_indexes = find_progens(PID_current, PID_progens,
                                    list_current, list_progens,
                                    nobjs)
        write_progen_data(obj_current, prev_indexes, data_type, caesar_file)
    else:
        mylog.warning('%s progen data already present, skipping!' % data_type)


    ## clouds (use gas)
    data_type = 'cloud'
    if obj_current.nclouds == 0 or obj_progens.nclouds == 0:
        mylog.warning('0 %s found! skipping progen' % group_types[data_type])
    elif not check_if_progen_is_present(data_type, caesar_file):
        nobjs        = obj_current.nclouds
        PID_current  = readsnap(snap_current.snap, 'pid', 0)
        PID_progens  = readsnap(snap_progens.snap, 'pid', 0)
        list_current = obj_current.global_particle_lists.cloud_glist
        list_progens = obj_progens.global_particle_lists.cloud_glist

        prev_indexes = find_progens(PID_current, PID_progens,
                                    list_current, list_progens,
                                    nobjs)
        write_progen_data(obj_current, prev_indexes, data_type, caesar_file)
    else:
        mylog.warning('%s progen data already present, skipping!' % data_type)
