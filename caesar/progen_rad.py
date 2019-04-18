
# Loops over all snapshots (in reverse) from last_snap to first_snap, sets up snapshot pairs [snapfile1,snapfile2] to find progenitors.
# For each galaxy in snapfile1, finds progenitors in snapfile2.  progenitors are defined as having the most star particles in common.
# Writes the progenitor IDs into the "progen_index" field in the galaxies object in the Caesar file associated with snapfile1.
# Writes the second-most common progenitor IDs into the "progen_index2" field in the galaxies object in the Caesar file associated with snapfile1.
# You can either daisychain the snaphots or always use last_snap for snapfile1.
# Works with multi-processor via OpenMP; doesn't help much because most of the time is in the sorting (not yet parallel).  It's fast tho!
# Last modified: Romeel Dave 18 Feb 2019

import caesar
from pygadgetreader import *
import numpy as np
from scipy import stats
import time
import sys
import os
import pdb
from glob2 import glob


#===============================================
#MODIFIABLE HEADER
#===============================================
nproc = 1
#GDIR = 'Groups'
mode = 'daisychain'  # 'daisychain' to link each snapshot to its previous or 'refsnap' to link a single given snapshot to all previous
min_in_common = 0.1  # require at least this fraction of stars in common between galaxy and its progenitor to be called a true progenitor

#===============================================

# Routine to find progenitor for given galaxy ig
def get_galaxy_prog(ig,PID_curr,id_prog,ig_prog,slist):
    ids = [PID_curr[i] for i in slist]
    prog_ind = np.searchsorted(id_prog,ids) # bisection search to find closest ID in prog
    for i in range(len(prog_ind)):  # handle some very occasional weirdness
        if prog_ind[i] >= len(id_prog): prog_ind[i] = len(id_prog)-1
    prog_ind = prog_ind[id_prog[prog_ind]==ids]  # find progenitor IDs that match star IDs in galaxy ig
    ig_matched = ig_prog[prog_ind]  # galaxy numbers of matched progenitor IDs
    if len(ig_matched)>int(min_in_common*len(ids)):
        modestats = stats.mode(ig_matched) # find prog galaxy id with most matches
        prog_index_ig = modestats[0][0]  # prog_index stores prog galaxy numbers
        ig_matched = ig_matched[(ig_matched!=prog_index_ig)]  # remove the first-most common galaxy, recompute mode
    else: prog_index_ig = -1
    if len(ig_matched)>0:
        modestats = stats.mode(ig_matched) # find prog galaxy id with second-most matches
        prog_index_ig2 = modestats[0][0]  # now we have the second progenitor
    else: prog_index_ig2 = -1
    #if ig<10: print ig,prog_index_ig,prog_index_ig2
    return prog_index_ig,prog_index_ig2

# Routine to load in particle IDs from snapshot
def load_IDs(snap1,snap2,t0,parttype='star'):
    PID_curr = np.array(readsnap(snap1,'pid',parttype),dtype=np.uint64) # particle IDs in current snapshot
    PID_prog = np.array(readsnap(snap2,'pid',parttype),dtype=np.uint64) # particle IDs in progenitor snapshot
    if parttype == 'star':
        try:
            IDgen_curr = np.array(readsnap(snap1,'ID_Generations','star')-1,dtype=np.uint64) # ID generation info for duplicate IDs
            IDgen_prog = np.array(readsnap(snap2,'ID_Generations','star')-1,dtype=np.uint64)
        except:  # older snapshots (like Mufasa) do not contain ID_Generations
            IDgen_curr = np.zeros(len(PID_curr))
            IDgen_prog = np.zeros(len(PID_prog))
        #DN surprisingly, we need one more catch here in case this
        #try/except doesn't work because pygr/readsnap returns an
        #empty array sometimes when there's no ID_Generations in the star particles
        if len(IDgen_curr) == 0:
            IDgen_curr = np.zeros(len(PID_curr))
            IDgen_prog = np.zeros(len(PID_prog))


        print('progen : Read ID info for %d current stars and %d progenitor stars [t=%g s]'%(len(PID_curr),len(PID_prog),time.time()-t0))
        maxid = max(max(PID_curr),max(PID_prog))
        #print len(IDgen_curr[IDgen_curr>0])
        PID_curr = np.add(PID_curr, maxid*IDgen_curr)
        PID_prog = np.add(PID_prog, maxid*IDgen_prog) #these ugly lines replace         PID_prog += maxid*IDgen_prog because certain versions of numpy make them unsafe casting and error out
    return PID_curr,PID_prog

# Routine to find progenitors for all galaxies in snapshot
def find_progens(snap1,snap2,obj1,obj2,nproc,t0,objtype='halo'):
    PID_curr,PID_prog = load_IDs(snap1,snap2,t0) # Gather all the IDs and associated galaxy/halo numbers from the progenitor snapshot
    if objtype == 'galaxy':
        objects1=obj1.galaxies
        objects2=obj2.galaxies
    elif objtype == 'halo':
        objects1=obj1.halos
        objects2=obj2.halos
    else: sys.exit('progen : ERROR: objtype %s not valid'%objtype)

    ngal_curr = len(objects1)
    ngal_prog = len(objects2)
    id_prog = np.zeros(len(PID_prog),dtype=np.uint64)  # particle IDs of progenitor stars
    ig_prog = np.zeros(len(PID_prog),dtype=int)        # galaxy IDs of progenitor stars
    count = 0
    for ig in range(ngal_prog):
        # Gather list of progenitor stars and associated galaxy numbers
        for ilist in range(len(objects2[ig].slist)):
            istar = objects2[ig].slist[ilist]
            id_prog[count] = PID_prog[istar]
            ig_prog[count] = ig
            count += 1
        # List comprehension version below is significantly slower
        #ids = [PID_prog[i] for i in obj2.galaxies[ig].slist]
        #igs = [ig for i in obj2.galaxies[ig].slist]
        #id_prog = np.concatenate((id_prog,ids))
        #ig_prog = np.concatenate((ig_prog,igs))
    print('progen : Gathered ID info for %d progenitor stars out of %d total stars [t=%g s]'%(count,len(PID_curr),time.time()-t0))
    id_prog = id_prog[:count]
    ig_prog = ig_prog[:count]

    # Sort the progenitor IDs and galaxy numbers for faster searching
    isort_prog = np.argsort(id_prog,kind='quicksort')
    id_prog = id_prog[isort_prog]  # this stores the progenitor star IDs
    ig_prog = ig_prog[isort_prog]  # this stores the galaxy IDs for the progenitor stars
    print('progen : Sorted progenitor IDs [t=%g s]'%(time.time()-t0))

    # Loop over galaxies in current snapshot
    if nproc>1: 
        prog_index_tmp = Parallel(n_jobs=nproc)(delayed(get_galaxy_prog)(ig,PID_curr,id_prog,ig_prog,objects1[ig].slist) for ig in range(ngal_curr))
        prog_index_tmp = np.array(prog_index_tmp,dtype=int)
        prog_index = np.array(prog_index_tmp.T[0],dtype=int)
        prog_index2 = np.array(prog_index_tmp.T[1],dtype=int)
    else:
        prog_index = np.zeros(ngal_curr,dtype=int)
        prog_index2 = np.zeros(ngal_curr,dtype=int)
        for ig in range(ngal_curr):
            prog_index[ig],prog_index2[ig] = get_galaxy_prog(ig,PID_curr,id_prog,ig_prog,objects1[ig].slist)

    # Print some stats and return the indices
    try:
        print('progen : Out of',ngal_curr,'galaxies, most common prog',stats.mode(prog_index[prog_index>=0])[0][0],'appeared',stats.mode(prog_index[prog_index>=0])[1][0],'times.',stats.mode(prog_index[prog_index<0])[1][0],'had no progenitors.')
    except:
        print('0 had no progenitors.')
    return prog_index,prog_index2

#=========================================================
# MAIN DRIVER ROUTINE
#=========================================================

def run_progen_rad(obj_current,obj_progens,snap_current,snap_progens):

    BASEDIR = snap_current.snapdir
    snapshot = obj_current.simulation.basename.decode()
    GDIR = 'Groups/'

    # Set up pairs;

 
    if not os.path.isfile('%s/%s' % (BASEDIR,snapshot)):
        sys.exit('Reference_snap %s/%s does not exist'%(BASEDIR,snapshot))


    '''
    allsnaps = np.arange(last_snap,first_snap-1,-1)
    # find snapshots with caesar files that exist in the directory, in reverse order
    snapnums = []
    for i in allsnaps:
        print('%s/snapshot_%03d.hdf5' % (BASEDIR,i))
        print('%s/%s/*_%04d*.hdf5' % (BASEDIR,GDIR,i))
        if os.path.isfile('%s/snapshot_%03d.hdf5' % (BASEDIR,i)) and len(glob('%s/%s/*_%04d*.hdf5' % (BASEDIR,GDIR,i))) == 1:
            snapnums.append(i)
    #print('Found these snapshots with caesar files: %s'%snapnums)
    # set up pairs to progen
    pairs = []
    if mode=='refsnap':  prevsnap = snapnums[0]  # find progenitors based on a single reference (final) snapshot
    for i in range(len(snapnums)-1):
        if mode=='daisychain':  prevsnap = snapnums[i] # daisy-chain progenitors in each snapshot to the previous snapshot
        pairs.append([prevsnap,snapnums[i+1]])
    
    print('[progen_rad/run_progen_rad] : Doing snapshot pairs: %s'%pairs)
    '''
# Set up multiprocessing; note: this doesn't help much, because this only parallelizes over galaxy ID searches, not the sorting.
    if nproc == 0:   # use all available cores
        num_cores = multiprocessing.cpu_count()
    if nproc != 1:   # if multi-core, set up Parallel processing
        import multiprocessing
        from joblib import Parallel, delayed
        from functools import partial
        num_cores = multiprocessing.cpu_count()
        if nproc < 0: print('progen : Using %d cores (all but %d)'%(num_cores+nproc+1,-nproc-1) )
        if nproc > 1: print('progen : Using %d of %d cores'%(nproc,num_cores))
        else: print('progen : Using single core')

    '''
# loop over pairs to find progenitors
    t0 = time.time()
    prev_pair = [-1,-1]
    for pair in pairs:

        print('progen : Doing pair %s [t=%g s]'%(pair,np.round(time.time()-t0,3)))
        if pair[0] == prev_pair[1]:  # don't have to reload if we already have this object from the previous iteration
            snapfile1 = snapfile2
            caesarfile1 = caesarfile2
            obj1 = obj2
        else:
    '''
    t0 = time.time()

    snapfile1 = obj_current.simulation.fullpath.decode()+'/'+obj_current.simulation.basename.decode()
    snapfile2 = obj_progens.simulation.fullpath.decode()+'/'+obj_progens.simulation.basename.decode()

    
#    snapfile1 = '%s/snapshot_%03d.hdf5' % (BASEDIR,pair[0])   # current snapshot
#    caesarfile1 = glob('%s/%s/*%04d*.hdf5' % (BASEDIR,GDIR,pair[0]))[0] #taking 0th element since glob returns a list
    obj1 = obj_current#caesar.load(caesarfile1,LoadHalo=0)
#    snapfile2 = '%s/snapshot_%03d.hdf5' % (BASEDIR,pair[1])   # progenitor snapshot
#    caesarfile2 = glob('%s/%s/*%04d*.hdf5' % (BASEDIR,GDIR,pair[1]))[0] #taking 0th element since glob returns a list
    obj2 = obj_progens#caesar.load(caesarfile2,LoadHalo=0)

    for data_type in ['halo','galaxy']:
        prog_index,prog_index2 = find_progens(snapfile1,snapfile2,obj1,obj2,nproc,t0,objtype=data_type)  # find galaxies with most stars in common in progenitor snapshot
        
        # append progenitor info to caesar file
        caesar_file = obj1.data_file  # file to write progen info to
        





        try:
            caesar.progen.write_progen_data(obj1, prog_index, data_type, caesar_file, 'progen_index')
        except:
            caesar.progen.rewrite_progen_data(obj1, prog_index, data_type, caesar_file, 'progen_index')
        try:
            caesar.progen.write_progen_data(obj1, prog_index2, data_type, caesar_file, 'progen_index2')
        except:
            caesar.progen.rewrite_progen_data(obj1, prog_index2, data_type, caesar_file, 'progen_index2')
    
        print('progen : Wrote info to caesar file %s [t=%g s]'%(caesar_file,time.time()-t0))

    #prev_pair = pair


