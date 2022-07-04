
# 6-D FOF.
# Input is a snapshot file containing gas, stars, and BHs.
# Snapshot must contain a HaloID putting each particle in a halo (0=not in halo).
# First groups particles via an approximate FOF using  via quicksorts in successive directions.
# Then does a full 6-D FOF search on the approximate FOF groups
# Outputs SKID-format .grp binary file containing Npart, then galaxy IDs of all particles
#
# Romeel Dave, 25 Feb 2019

import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
import sys
import os
import h5py
from yt.funcs import mylog
from caesar.utils import memlog
from caesar.property_manager import MY_DTYPE, get_property,has_ptype,ptype_ints
from caesar.group import MINIMUM_STARS_PER_GALAXY

class fof6d:

    def __init__(self, obj, group_type):
        self.obj = obj
        self.obj._args   = obj._args
        self.obj._kwargs = obj._kwargs
        self.obj_type = group_type
        self.Lbox = obj.simulation.boxsize.d
        self.counts = {}

        # set up number of processors
        self.nproc = obj.nproc

        # turn off unbinding; no need since fof6d already accounts for kinematics
        from caesar.group import group_types
        for gt in [group_type]:
            unbind_str = 'unbind_%s' % group_types[gt]
            setattr(self.obj.simulation, unbind_str, False)

    def load_haloid(self):
        # Get Halo IDs, either from snapshot or else run fof.
        # This will be a list of numpy arrays for each ptype
        if self.obj.load_haloid:
            from caesar.property_manager import get_haloid
            memlog('Using FOF Halo ID from snapshots')
            self.haloid = get_haloid(self.obj, self.obj.data_manager.ptypes, offset=-1)
        elif 'haloid' in self.obj._kwargs and 'fof' in self.obj._kwargs['haloid']:
            self.run_caesar_fof(self.obj)
        elif 'haloid' in self.obj._kwargs and 'rockstar' in self.obj._kwargs['haloid']:
            sys.exit('Sorry, reading from rockstar files not implemented yet')
        elif 'haloid' in self.obj._kwargs and 'AHF' in self.obj._kwargs['haloid']:
            self.load_ahf_id()
        else:
            from caesar.property_manager import get_haloid
            memlog('No Halo ID source specified -- running FOF.  This is the yt 3D FOF for halos and our homegrown 6D FOF for galaxies ...')
            try:
                self.run_caesar_fof(self.obj)
                self.obj._kwargs['haloid'] = 'fof'
            except:
                sys.exit("No Halo IDs found in snapshot -- please specify a source (haloid='fof' or 'snap')")
    def load_ahf_id(self):
        if 'haloid_file' in self.obj._kwargs and self.obj._kwargs['haloid'] is not None:
            haloid_file = self.obj._kwargs['haloid_file']
            if os.path.isfile(haloid_file):
                memlog('Reading AHF halo IDs from %s'%(haloid_file))
                if '.gz' == haloid_file[-3:]: #compressed txt file
                    import gzip
                    with gzip.open(haloid_file) as f:
                        lines = f.readlines()
                else:        
                    with open(haloid_file) as f:
                        lines = f.readlines()

                nhalos=int(lines[0])
                hpp=1 # halo particle numbers and halo ID position
                hid_info={} #particle ID, particle type, halo_ID as keys fillup with the information from AHF particle file
                for i in range(nhalos):
                    npt,hid=[int(x) for x in lines[hpp].split()]
                    hpp+=1
                    hid_info[str(hid)]=np.loadtxt(lines[hpp:hpp+npt],dtype=int)
                    hpp+=npt
                    # hid_info.extend(tmpd.tolist())
                    
                #exclude subhalo particles in host halo!!
                if '.gz' == haloid_file[-3:]: #compressed txt file
                    haloid_file=haloid_file.replace('particles.gz','halos')
                else:
                    haloid_file=haloid_file.replace('particles','halos')
                halo_info=np.loadtxt(haloid_file,usecols=(0,1),dtype=np.int64) #hid, host hid
                idsh=np.where(halo_info[:,1]>0)[0]
                for i in idsh:
                    if halo_info[i,1] in halo_info[:,0]: # sometimes substructure ID doesn't exist
                        com,x_ind,y_ind = np.intersect1d(hid_info[str(halo_info[i,1])][:,0], hid_info[str(halo_info[i,0])][:,0], return_indices=True)
                        hid_info[str(halo_info[i,1])]=np.delete(hid_info[str(halo_info[i,1])], x_ind, axis=0)

                # now put everything togather
                tmpp=[] #particle ID, particle type, halo_ID
                Nh=0
                for i in hid_info.keys():
                    if hid_info[i].shape[0]>20:
                        tmppd=np.zeros((hid_info[i].shape[0],3),dtype=np.int64)
                        tmppd[:,:2]=hid_info[i]
                        tmppd[:,2]=np.int64(i)
                        tmpp.extend(tmppd.tolist())
                        Nh+=1
                hid_info=np.asarray(tmpp)
                if len(np.unique(hid_info[:,0])) != hid_info.shape[0]:
                    print('!!Warning!! Still dumplicated particle IDs !!', Nh, len(np.unique(hid_info[:,0])), hid_info.shape[0])
                
                # Now load the simulation particle IDs # map back to the position 
                self.haloid = []
                pids = []
                nhid = 0
                for p in self.obj.data_manager.ptypes:
                    if has_ptype(self.obj, p): 
                        data = get_property(self.obj, 'pid', p).d.astype(np.int64)
                        tmpp = np.zeros(len(data),dtype=np.int64)-1
                        tmppd=hid_info[hid_info[:,1]==ptype_ints[p]]
                        com,x_ind, y_ind = np.intersect1d(data,tmppd[:,0],return_indices=True)
                        #tmpp = tmppd[y_ind,2][np.argsort(x_ind)]
                        tmpp[x_ind] = tmppd[y_ind,2]
                        nhid += len(com)
                        pids.extend(tmpp[tmpp>=0])
                        self.haloid.append(tmpp)
                memlog('Total halo particle IDs = %d'%(nhid))
                self.haloid = np.asarray(self.haloid)         # all particles   
                self.obj.data_manager.haloid = np.asarray(pids) # only halo particles
        else:
            mylog.warning("With haloid='AHF' you must also specify a haloid_file containing halo[+subhalo] IDs.")
            sys.exit()
        
    def run_caesar_fof(self,obj):
        from caesar.fubar import get_mean_interparticle_separation,get_b,fof
        LL = get_mean_interparticle_separation(self.obj) * get_b(self.obj, 'halo')
        if 'haloid_file' in self.obj._kwargs and self.obj._kwargs['haloid'] is not None:
            haloid_file = self.obj._kwargs['haloid_file']
            if os.path.isfile(haloid_file):
                memlog('Reading 3D FOF Halo IDs from %s'%haloid_file)
                hf = h5py.File(haloid_file,'r')
                self.obj.data_manager.haloid = np.asarray(hf['all_haloids'])
                self.haloid = []
                for p in self.obj.data_manager.ptypes:  # read haloid arrays for each ptype
                    self.haloid.append(np.asarray(hf['haloids_%s'%p]))
                self.haloid = np.array(self.haloid)
                hf.close()
                return
        else:
            haloid_file = None
        memlog('Running 3D FOF to get Halo IDs, LL=%g'%LL)
        pos = np.empty((0,3),dtype=MY_DTYPE)
        ptype = np.empty(0,dtype=np.int32)
        for ip,p in enumerate(self.obj.data_manager.ptypes):  # get positions
            if not has_ptype(self.obj, p): continue
            data = get_property(self.obj, 'pos', p).to(self.obj.units['length'])
            pos = np.append(pos, data.d, axis=0)
            ptype = np.append(ptype, np.full(len(data), ptype_ints[p], dtype=np.int32), axis=0)
        haloid_all = fof(self.obj, pos, LL, group_type='halo')  # run FOF
        self.haloid = []
        haloid = np.empty(0,dtype=np.int64)
        for p in self.obj.data_manager.ptypes:  # fill haloid arrays for each ptype
            if has_ptype(self.obj, p):
                data = haloid_all[ptype==ptype_ints[p]]
                datasel = data[data>=0]
            else:
                data = np.empty(0,dtype=np.int64)
                datasel = np.empty(0,dtype=np.int64)
            self.haloid.append(data)
            haloid = np.append(haloid,datasel,axis=0)
        self.haloid = np.asarray(self.haloid)
        self.obj.data_manager.haloid = haloid
        if haloid_file is not None:
            memlog('Writing 3D FOF Halo IDs to %s' % haloid_file)
            with h5py.File(haloid_file,'w') as hf:  
                hf.create_dataset('all_haloids',data=haloid, compression=1)
                for ip,p in enumerate(self.obj.data_manager.ptypes):  # write haloid arrays for each ptype
                    haloid_out = self.haloid[ip]
                    hf.create_dataset('haloids_%s'%p,data=haloid_out, compression=1)
                hf.close()

    def plist_init(self,parent=None):
        # set up particle lists 
        from caesar.group import MINIMUM_DM_PER_HALO,MINIMUM_STARS_PER_GALAXY
        if self.obj_type == 'halo' or parent is None:
            grpid = self.obj.data_manager.haloid - 1
            if len(grpid[grpid>=0]) < MINIMUM_DM_PER_HALO:
                mylog.warning('Not enough halo particles for a single valid halo (%d < %d)'%(len(grpid[grpid>=0]),MINIMUM_DM_PER_HALO))
                return False
        else:
            grpid = parent.tags_fof6d
            if len(grpid[grpid>=0]) < MINIMUM_STARS_PER_GALAXY:
                self.nparttot = 0
                return False

        # sort by grpid
        from caesar.property_manager import ptype_ints
        self.nparttot = len(grpid)
        self.nparttype = {}
        for p in self.obj.data_manager.ptypes:
            self.nparttype[p] = len(grpid[self.obj.data_manager.ptype==ptype_ints[p]])
        sort_grpid = np.argsort(grpid)
        self.pid_sorted = np.arange(self.nparttot,dtype=np.int64)[sort_grpid]
        # self.grouplist = np.unique(grpid)  # list of objects (halo/galaxy/cloud) to process
        hid_sorted = grpid[sort_grpid]
        self.hid_bins = find_bins(hid_sorted,self.nparttot)
        self.grouplist = hid_sorted[self.hid_bins[:-1]] # list of objects (halo/galaxy/cloud) as haloid-1 in the same order
        return True

    def run_fof6d(self, target_type, nHlim=0.13, Tlim=1.e5, sfflag=True, minstars=MINIMUM_STARS_PER_GALAXY):

        from joblib import Parallel, delayed
        from caesar.fubar import get_b
        from caesar.group import group_types

        # initialize fof6d parameter
        self.fof_LL = self.MIS * get_b(self.obj, target_type)  
        self.vel_LL = 1.0
        self.kerneltab = kernel_table(self.fof_LL)
        self.nHlim = nHlim  # only include gas above this nH limit (atoms/cm^3)
        self.Tlim = Tlim  # only include gas below this temperature
        self.sfflag = sfflag  # if True, always include particles with nonzero SF regardless of other crit
        self.minstars = minstars
        # set eligible galaxy gas: nH>nHlim, with T<Tlim OR star-forming
        if self.sfflag:
            self.dense_crit = lambda gnh, gtemp, gsfr: (gnh>self.nHlim)&((gtemp<self.Tlim)|(gsfr>0))
        else:
            self.dense_crit = lambda gnh, gtemp, gsfr: (gnh>self.nHlim)&(gtemp<self.Tlim)

        # collect indices for eligible particles
        memlog('Running fof6d on %d halos w/%d proc(s), LL=%g'%(len(self.obj.halo_list),self.nproc,self.fof_LL))
        g_inds = []  # indexes of (gas star BH dust) particle eligible for being in a group
        len_hi = len_gi = 0
        for ih in range(len(self.obj.halo_list)):
            g_inds.append(setup_indexes(self,self.obj.halo_list[ih].global_indexes))
            len_hi += len(self.obj.halo_list[ih].global_indexes)
            len_gi += len(g_inds[ih])
        memlog('%d halo particles, %g%% eligible for galaxies'%(len_hi, np.round(100*len_gi/len_hi,2)))

        # get tags using fof6d
        ngs=0
        if self.nproc == 1:
            grp_tags = [None]*len(self.obj.halo_list)  # particle IDs for fof6d objects
            for ih in range(len(self.obj.halo_list)):
                grp_tags[ih],tmg = fof6d_halo(len(self.obj.halo_list[ih].global_indexes),len(g_inds[ih]),self.obj.data_manager.pos[g_inds[ih]],self.obj.data_manager.vel[g_inds[ih]],self.minstars,self.obj.simulation.boxsize.d,self.fof_LL,self.vel_LL,self.kerneltab)
                ngs+=tmg
        else:
            tmg = Parallel(n_jobs=self.nproc)(delayed(fof6d_halo)(len(self.obj.halo_list[ih].global_indexes),len(g_inds[ih]),self.obj.data_manager.pos[g_inds[ih]],self.obj.data_manager.vel[g_inds[ih]],self.minstars,self.obj.simulation.boxsize.d,self.fof_LL,self.vel_LL,self.kerneltab) for ih in range(len(self.obj.halo_list)))
            grp_tags,ngs = zip(*tmg)
            grp_tags=list(grp_tags)
            ngs = np.sum(ngs)

        # adjust tags to be sequential in galaxy number overall (rather than within each halo)
        ngrp = 0
        memlog('total galaxies %d, total groups %d'%(ngs,len(grp_tags)))
        self.group_parents = np.zeros(ngs,dtype=np.int32)
        for ih in range(len(grp_tags)):
            grp_tags[ih] = np.where(grp_tags[ih]>=0, grp_tags[ih]+ngrp, -1)
            mygals = np.unique(grp_tags[ih])
            mygals = mygals[mygals>=0]
            self.group_parents[mygals] = ih
            ngrp += len(mygals)

        if ngrp != ngs:
            memlog('WARNING!! total galaxies number do not agree: %d, %d'%(ngs,ngrp))
        # collect overall tags 
        self.tags_fof6d = np.zeros(self.nparttot,dtype=np.int64) - 1
        for igrp in range(len(grp_tags)):
            self.tags_fof6d[g_inds[igrp]] = grp_tags[igrp]

        memlog('Done fof6d, found %d %s'%(ngrp,group_types[target_type]))

    def load_lists(self,parent=None):
        # create valid caesar groups, populate index lists
        from caesar.group import create_new_group, group_types
        from caesar.property_manager import ptype_ints, has_ptype
        grp_list = []
        if parent is not None:
            for ihalo in range(len(parent.obj.halo_list)):
                parent.obj.halo_list[ihalo].galaxy_index_list = []
        ngrp = 0
        zero_marker = 0
        for igrp in range(len(self.grouplist)):
            if self.grouplist[igrp] < 0: 
                zero_marker = 1  # if there are particles with tag=-1, these will be in igrp=0 within hid_bins. In this case, group_parents should start their numbering at 1, since igrp=0 is not a valid object.  This should only happen for galaxies/clouds, not halos
                continue
            mygrp = create_new_group(self.obj, self.obj_type)
            my_indexes = self.pid_sorted[self.hid_bins[igrp]:self.hid_bins[igrp+1]]  # indexes for parts in this group
            # load indexes into lists for a given group, globally and for each particle type
            my_ptype = self.obj.data_manager.ptype[my_indexes]
            my_pos = self.obj.data_manager.pos[my_indexes]
            mygrp.global_indexes = my_indexes
            offset = 0
            for ip,p in enumerate(self.obj.data_manager.ptypes):
                if not has_ptype(self.obj, p): continue
                if p == 'gas': 
                    mygrp.glist = my_indexes[my_ptype==ptype_ints[p]]-offset
                    mygrp.ngas = len(mygrp.glist)
                elif p == 'star': 
                    mygrp.slist = my_indexes[my_ptype==ptype_ints[p]]-offset
                    mygrp.nstar = len(mygrp.slist)
                elif p == 'bh': 
                    mygrp.bhlist = my_indexes[my_ptype==ptype_ints[p]]-offset
                    mygrp.nbh = len(mygrp.bhlist)
                elif p == 'dust': 
                    mygrp.dlist = my_indexes[my_ptype==ptype_ints[p]]-offset
                    mygrp.ndust = len(mygrp.dlist)
                elif p == 'dm': 
                    mygrp.dmlist = my_indexes[my_ptype==ptype_ints[p]]-offset
                    mygrp.ndm = len(mygrp.dmlist)
                elif p == 'dm2': 
                    mygrp.dm2list = my_indexes[my_ptype==ptype_ints[p]]-offset
                    mygrp.ndm2 = len(mygrp.dm2list)
                elif p == 'dm3':
                    mygrp.dm3list = my_indexes[my_ptype==ptype_ints[p]]-offset
                    mygrp.ndm3 = len(mygrp.dm3list)
                offset += self.nparttype[p]
            if mygrp._valid:
                mygrp.obj_type = self.obj_type
                if self.obj_type == 'halo':
                    if self.obj._kwargs['haloid'] == 'AHF':
                        mygrp.AHF_haloID = self.grouplist[igrp] + 1 # recover to orginal ID, see line 156
                if parent is not None: 
                    ihalo = parent.group_parents[igrp-zero_marker]
                    mygrp.parent_halo_index = ihalo
                    parent.obj.halo_list[ihalo].galaxy_index_list.append(ngrp)
                    ngrp += 1
                grp_list.append(mygrp)

        if self.obj_type == 'halo': 
            self.obj.halo_list = grp_list
            self.counts[self.obj_type] = len(self.obj.halo_list)
            self.obj.group_types.append(self.obj_type)
        if self.obj_type == 'galaxy': 
            self.obj.galaxy_list = grp_list
            self.counts[self.obj_type] = len(self.obj.galaxy_list)
            self.obj.group_types.append(self.obj_type)
        if self.obj_type == 'cloud': 
            self.obj.cloud_list = grp_list
            self.counts[self.obj_type] = len(self.obj.cloud_list)
            self.obj.group_types.append(self.obj_type)

        memlog('Found %d valid %s, loaded indexes'%(self.counts[self.obj_type],group_types[self.obj_type]))


    def load_fof6dfile(self):
        import os
        from yt.funcs import mylog
        fof6d_file = self.obj._kwargs['fof6d_file']
        if os.path.isfile(fof6d_file):
            mylog.info('Reading galaxy membership from fof6d file %s'%fof6d_file)
        else:
            mylog.info('fof6d file %s not found! Running fof6d' % fof6d_file)
            return True
        hf = h5py.File(fof6d_file,'r')
        self.tags_fof6d = np.asarray(hf['fof6d_tags'])
        self.group_parents = np.asarray(hf['group_parents'])
        hf.close()
        if len(self.tags_fof6d)!=self.nparttot:
            mylog.warning('fof6d_file invalid! len(fof6d_tags) does not match length of particle list in halos (%d != %d) and/or len(group_parents) does not match length of halo list (%d != %d) -- RUNNING FOF6D'%(len(self.tags_fof6d),self.nparttot,len(self.group_parents),self.counts[self.obj_type]))
            return True
        return False

    def save_fof6dfile(self):
        if 'fof6d_file' not in self.obj._kwargs or self.obj._kwargs['fof6d_file'] is None:
            return
        fof6d_file = self.obj._kwargs['fof6d_file']
        memlog('Writing fof6d info to %s' % fof6d_file)
        all_tags = self.tags_fof6d
        group_parents = self.group_parents
        with h5py.File(fof6d_file,'w') as hf:  # overwrites existing fof6d group file
            hf.create_dataset('fof6d_tags',data=all_tags, compression=1)
            hf.create_dataset('group_parents',data=group_parents, compression=1)
            hf.close()

    '''
    def link_to_parent(self,parent=None):
        # Set up cross-matching between parent and children
        if parent == None: return
        for ih in range(len(parent.grouplist)):
            group_nums = np.unique(parent.tags_fof6d[ih])
            if parent.obj_type == 'halo':
                if self.obj_type == 'galaxy':
                    self.obj.halo_list[ih].galaxy_index_list = np.empty(0, dtype=np.int32)
                elif self.obj_type == 'cloud':
                    self.obj.halo_list[ih].cloud_index_list = np.empty(0, dtype=np.int32)
            if parent.obj_type == 'galaxy':
                self.obj.galaxy_list[ih].cloud_index_list = np.empty(0, dtype=np.int32)
            for igrp in group_nums[group_nums>=0]:
                if parent.obj_type == 'halo': 
                    mygrp.parent_halo_index = ih
                    if self.obj_type == 'galaxy':
                        self.obj.halo_list[ih].galaxy_index_list = np.append(self.obj.halo_list[ih].galaxy_index_list,self.counts[target_type])
                    if self.obj_type == 'cloud':
                        self.obj.halo_list[ih].cloud_index_list = np.append(self.obj.halo_list[ih].cloud_index_list,self.counts[target_type])
    '''


#=========================================================
# 6DFOF ROUTINES
#=========================================================

def find_bins(sorted_list,last_value):
    # find particle indexes in sorted halos list (from alimanfoo/find_runs.py)
    loc_run_start = np.empty(len(sorted_list), dtype=bool)
    loc_run_start[0] = True
    np.not_equal(sorted_list[:-1], sorted_list[1:], out=loc_run_start[1:])
    sorted_bins = np.nonzero(loc_run_start)[0]
    sorted_bins = np.append(sorted_bins,last_value)
    return sorted_bins

def setup_indexes(self,halo_indexes):
    ''' Collect indexes of eligible gas/stars/bh/dust from among particles in a given halo
    NOTE: The indexes returned are tagged to the particles within a given halo '''
    from caesar.property_manager import ptype_ints
    # first quickly check if there are enough stars
    my_ptype = self.obj.data_manager.ptype[halo_indexes]
    star_indexes = halo_indexes[my_ptype == ptype_ints['star']]
    if len(star_indexes) < self.minstars:
        return np.zeros(1,dtype=np.int32)
    # collect particles for fof6d: first apply dense gas cut
    gas_indexes = halo_indexes[my_ptype == ptype_ints['gas']]
    gtemp = self.obj.data_manager.gT[gas_indexes]
    gsfr = self.obj.data_manager.gsfr[gas_indexes]
    gnh = self.obj.data_manager.gnh[gas_indexes]
    select_dense_gas = self.dense_crit(gnh, gtemp, gsfr)
    dense_indexes = gas_indexes[select_dense_gas]
    # add in other particle types
    bh_indexes = halo_indexes[my_ptype == ptype_ints['bh']]
    dust_indexes = halo_indexes[my_ptype == ptype_ints['dust']]
    all_indexes = np.concatenate((dense_indexes,star_indexes,bh_indexes,dust_indexes),axis=None).astype(np.int32)
    # concatenate everything in the proper order and return
    return all_indexes

def fof6d_halo(nparthalo,npart,pos,vel,minstars,Lbox,fof_LL,vel_LL,kerneltab):
    ''' Routine to find galaxies within a given halo using fof6d '''

    #initialize fof6d
    fof6d_tags = np.zeros(npart,dtype=np.int64)-1  # default is that no particles are in galaxies
    if npart <= minstars:  # no possible valid galaxies; we're done
        return fof6d_tags, 0
    mypos = np.copy(pos)
    myvel = np.copy(vel)
    mypos = mypos.T # transpose since fof6d routines expect [npart,ndim]
    myvel = myvel.T

    # run fof6d
    groups = [[0,npart]]  # group to process has the entire list of particles in halo
    pindex = np.arange(npart,dtype=np.int32) # index to keep track of particle sorting
    myhaloID = np.zeros(npart,dtype=np.int32)  # fof6d doing one halo at a time; arbitrarily assign to halo 0
    for idir in range(len(mypos)):  # sort in each direction, find groups within sorted list
        if len(groups) > 0: groups = fof_sorting_old(groups,mypos,myvel,myhaloID,pindex,fof_LL,Lbox,idir,mingrp=minstars)
    if len(groups) == 0: 
        return fof6d_tags,0  # found no valid groups after sorting
    fof6d_results = [None]*len(groups)
    for igrp in range(len(groups)):
        if groups[igrp][1]-groups[igrp][0] < minstars: continue
        fof6d_results[igrp] = fof6d_main(igrp,groups,mypos.T[groups[igrp][0]:groups[igrp][1]],myvel.T[groups[igrp][0]:groups[igrp][1]],kerneltab,0.,Lbox,minstars,fof_LL,vel_LL)

    # insert galaxy IDs into particle lists
    nfof = 0
    galindex = np.zeros(npart,dtype=int)-1
    for igrp in range(len(groups)):
        if fof6d_results[igrp] is None: continue  # no valid galaxies
        istart = groups[igrp][0]  # starting particle index for group igrp
        iend = groups[igrp][1]
        galindex[istart:iend] = np.where(fof6d_results[igrp][1]>=0,fof6d_results[igrp][1]+nfof,-1)  # for particles in galaxies, increment galaxy ID with counter (nfof)
        nfof += fof6d_results[igrp][0]

    # reset back into original particle order and collect tags for particles in this halo
    for i in range(npart):
        fof6d_tags[pindex[i]] = galindex[i]
    # returns the group to which each particle belongs (-1 if not in group)
    return fof6d_tags,nfof

def fof6d_main(igrp,groups,poslist,vellist,kerneltab,t0,Lbox,mingrp,fof_LL,vel_LL=None,nfof=0):
    # find neighbors of all particles within fof_LL
    istart = groups[igrp][0]  # starting particle index for group igrp
    iend = groups[igrp][1]
    nactive = iend-istart
    if nactive < mingrp: return [0,[]]
    neigh = NearestNeighbors(radius=fof_LL)  # set up neighbor finder
    neigh.fit(poslist)  # do neighbor finding
    nlist = neigh.radius_neighbors(poslist)  # get neighbor properties (radii, indices)

    # compute velocity criterion for neighbors, based on local velocity dispersion 
    if vel_LL is not None:
        LLinv = 1./fof_LL
        sigma = np.zeros(nactive)
        siglist = []  # list of boolean arrays storing whether crit is satisfied for each neighbor pair
        # compute local velocity dispersion from neighbors
        for i in range(nactive):
            ngblist = nlist[1][i]  # list of indices of neighbors
            rlist = nlist[0][i]  # list of radii of neighbors
            # compute kernel-weighted velocity dispersion
            wt = kernel(rlist*LLinv,kerneltab)
            dv = np.linalg.norm(vellist[ngblist]-vellist[i],axis=1)
            sigma[i] = np.sqrt(np.sum(wt*dv*dv)/np.sum(wt))
            siglist.append((dv <= vel_LL*sigma[i]))
    else:
        # if velocity criterion not used, then all particles satisfy it by default
        siglist = []  # list of boolean arrays storing whether vel disp crit is satisfied
        for i in range(nactive):
            ngbnum = len(nlist[1][i])
            sigs = np.ones(ngbnum,dtype=bool)  # array of True's
            siglist.append(sigs)

    # determine counts within fof_LL, set up ordering of most dense to least
    ncount = np.zeros(nactive,dtype=int)
    for i in range(len(ncount)):
        ncount[i] = len(nlist[1][i])  # count number of neighbors for each particle
    dense_order = np.argsort(-ncount)  # find ordering of most dense to least

    # main loop to do FOF
    galind = np.zeros(nactive,dtype=int)-1
    linked = []
    galcount = 0
    for ipart in range(nactive):
        densest = dense_order[ipart]  # get next densest particle
        galind_ngb = galind[nlist[1][densest]]  # indices of neighbors' galaxies
        galind_ngb = np.where(siglist[densest],galind_ngb,-1)  # apply velocity criterion here
        if len(galind_ngb[galind_ngb>=0]) > 0:  # if it already has neighbors (incl itself) in a galaxy...
            galmin = np.unique(galind_ngb[galind_ngb>=0])  # find sorted, unique indices of neighbors' gals
            galind[nlist[1][densest]] = min(galmin)  # put all neighbors in lowest-# galaxy
            for i in range(1,len(galmin)):  # link all other galaxies to lowest
                if linked[galmin[i]]==-1: 
                    linked[galmin[i]] = min(galmin)  # link all other galaxies to lowest index one
                else:
                    linked[galmin[i]] = min(linked[galmin[i]],min(galmin))  # connect all other galaxies to lowest index
        else:  # it has no neighbors in a galaxy, so create a new one
            galind[nlist[1][densest]] = galcount  # put all neighbors in a new galaxy
            linked.append(galcount) # new galaxy is linked to itself
            galcount += 1

    # handle linked galaxies by resetting indices of their particles to its linked galaxy
    for i in range(galcount-1,-1,-1):
        if linked[i] != i:
            assert linked[i]<i,'Trouble: mis-ordered linking %d > %d'%(i,linked[i])
            for j in range(iend-istart):
                if galind[j] == i: galind[j] = linked[i]

    # assign indices of particles to FOF groups having more than mingrp particles
    pcount,bin_edges = np.histogram(galind,bins=galcount)  # count particles in each galaxy
    for i in range(iend-istart):  # set indices of particles in groups with <mingrp members to -1
        if pcount[galind[i]] < mingrp: galind[i] = -1  
    if len(galind[galind>=0])==0: return 0,galind  # if there are no valid groups left, return
    galind_unique = np.unique(galind[galind>=0])  # find unique groups
    galind_inv = np.zeros(max(galind_unique)+1,dtype=int)
    for i in range(len(galind_unique)):
        galind_inv[galind_unique[i]] = i  # create mapping from original groups to unique set
    for i in range(iend-istart): 
        if galind[i]>=0: galind[i] = galind_inv[galind[i]] # re-assign group indices sequentially
    galcount = max(galind)+1
    #raw_input('press ENTER to continue')

    '''
    # check: are there groups that are too large?
    for i in range(galcount):
        npgal = len(galind[galind==i])
        galpos = np.array([poslist[j] for j in range(len(poslist)) if galind[j]==i])
        galpos = galpos.T
        #print npgal,galpos
        galsize = np.array([max(galpos[0])-min(galpos[0]),max(galpos[1])-min(galpos[1]),max(galpos[2])-min(galpos[2])])
        galsize = np.where(galsize>Lbox/2,Lbox-galsize,galsize)
        toolarge = 300
        if nfof < 10 and nfof > 0: print 'in fof6d:',nfof+i,i,npgal,np.mean(galpos[0]),np.mean(galpos[1]),np.mean(galpos[2]),galsize
        if galsize[0]>toolarge or galsize[1]>toolarge or galsize[2]>toolarge: print 'Too large?',igrp,iend-istart,galsize,galpos.T
    '''

    # Compile result to return: number of new groups found, and a galaxy index for particles from istart:iend
    result = [galcount]
    result.append(galind)
    #progress_bar(1.*groups[igrp][1]/groups[len(groups)-1][1],barLength=50,t=time.time()-t0)

    return result

def fof_sorting(groups,pos,vel,haloID,pindex,fof_LL,Lbox,mingrp,idir):
    oldgroups = groups[:]  # stores the groups found from the previous sorting direction
    npart = len(pindex)
    groups = [[0,npart]]  # (re-)initialize the group as containing all particles
    grpcount = 0
    for igrp in range(len(oldgroups)):  # loop over old groups, sort within each, find new groups
        # sort particles within group in given direction
        istart = oldgroups[igrp][0]  # starting particle index for group igrp
        iend = oldgroups[igrp][1]
        sort_parts(pos,vel,haloID,pindex,istart,iend,idir,'pos')  # sort group igrp in given direction
        # create new groups by looking for breaks of dx[idir]>fof_LL between particle positions
        oldpos = pos[idir][istart]
        groups[grpcount][0] = istart  # set start of new group to be same as for old group
        for i in range(istart,iend):
            if periodic(pos[idir][i],oldpos,Lbox) > fof_LL or i == iend-1:  
                groups[grpcount][1] = i  # set end of old group to be i
                groups.append([i,i])  # add new group, starting at i; second index is a dummy that will be overwritten when the end of the group is found
                grpcount += 1
            oldpos = pos[idir][i]
    assert grpcount>0,'fof6d : Found no groups or unable to separate groups via sorting; exiting. %d %d %d %d'%(idir,len(groups),len(oldgroups),npart)
    # Remove groups that have less than mingrp particles (gas+star)
    oldgroups = groups[:]
    groups = []
    for igrp in range(len(oldgroups)):
        istart = oldgroups[igrp][0]  
        iend = oldgroups[igrp][1]
        if iend-istart >= mingrp: groups.append(oldgroups[igrp])
    return groups

# progress bar, from https://stackoverflow.com/questions/3160699/python-progress-bar
def progress_bar(progress,barLength=10,t=None):
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done!\r\n"
    block = int(round(barLength*progress))
    if t is None: text = "\r[{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), np.round(progress*100,2), status)
    else: text = "\r[{0}] {1}% [t={2} s] {3}".format( "#"*block + "-"*(barLength-block), np.round(progress*100,2), np.round(t,2), status)
    sys.stdout.write(text)
    sys.stdout.flush()

# set up kernel table
def kernel_table(fof_LL,ntab=1000):
    kerneltab = np.zeros(ntab+1)
    hinv = 1./fof_LL
    norm = 0.31832*hinv**3
    for i in range(ntab):
        r = 1.*i/ntab
        q = 2*r*hinv
        if q > 2: kerneltab[i] = 0.0
        elif q > 1: kerneltab[i] = 0.25*norm*(2-q)**3
        else: kerneltab[i] = norm*(1-1.5*q*q*(1-0.5*q))
    return kerneltab

# kernel table lookup
def kernel(r_over_h,kerneltab):
    ntab = len(kerneltab)-1
    rtab = ntab*r_over_h+0.5
    itab = rtab.astype(int)
    return kerneltab[itab]


#=========================================================
# DEFUNCT FOF6D ROUTINES
#=========================================================

# 6-D FOF.
# Input is a snapshot file containing gas, stars, and BHs.
# Snapshot must contain a HaloID putting each particle in a halo (0=not in halo).
# First groups particles via an approximate FOF using  via quicksorts in successive directions.
# Then does a full 6-D FOF search on the approximate FOF groups
# Outputs SKID-format .grp binary file containing Npart, then galaxy IDs of all particles
#
# Romeel Dave, 25 Feb 2019

from astropy import constants as const

#BASEDIR = sys.argv[1]
#snapnum = sys.argv[2]

'''
MODEL = sys.argv[1]
WIND = sys.argv[2]
SNAP = int(sys.argv[3])
if len(sys.argv)==5: nproc = int(sys.argv[4])
else: nproc = 1
'''
nproc = 1

#BASEDIR = '/cosma/home/dc-dave2/data/%s/%s'%(MODEL,WIND)

# FOF options
mingrp = 16
LL_factor = 0.02  # linking length is this times the mean interparticle spacing 
vel_LL = 1.0  # velocity space linking length factor, multiplies local velocity dispersion

#=========================================================
# MISCELLANEOUS ROUTINES
#=========================================================

# Loads gas, star, BH information from snapshot.  Requires HaloID for all particles.
def loadsnap(snap,t0):
    import pygadgetreader as pygr
    redshift = pygr.readheader(snap,'redshift')
    h = pygr.readheader(snap,'h')

    # get gas Halo IDs so we can select only gas particles in halos
    ghalo = np.array(pygr.readsnap(snap,'HaloID','gas'),dtype=int)  # Halo ID of gas; 0=not in halo
    ngastot = len(ghalo)
    gas_select = (ghalo>0)
    ngastot = np.uint64(len(ghalo))
    pindex = np.arange(ngastot,dtype=np.uint64)  # keep an index for the original order of the particles

    # Load in gas info for selecting gas particles
    gnh = pygr.readsnap(snap,'rho','gas',units=1)[gas_select]*h*h*0.76*(1+redshift)**3/const.m_p.to('g').value   # number density in phys H atoms/cm^3
    gsfr = pygr.readsnap(snap,'sfr','gas',units=1)[gas_select]
    gtemp = pygr.readsnap(snap,'u','gas',units=1)[gas_select]  # temperature in K
    # Apply additional selection criteria on gas
    dense_gas_select = ((gnh>0.13)&((gtemp<1.e5)|(gsfr>0))) # dense, cool, or SF gas only 
    gas_select[gas_select>0] = (gas_select[gas_select>0] & dense_gas_select)

    # load in selected gas 
    gpos = pygr.readsnap(snap,'pos','gas',units=1)[gas_select]/h  # positions in ckpc
    gvel = pygr.readsnap(snap,'vel','gas',units=1,suppress=1)[gas_select] # vel in physical km/s
    ghalo = np.array(pygr.readsnap(snap,'HaloID','gas'),dtype=int)[gas_select]  # Halo ID of gas; 0=not in halo

    # load in all stars+BHs; don't bother with halo selection since most will be in halos anyways
    shalo = np.array(pygr.readsnap(snap,'HaloID','star'),dtype=int)  # Halo ID of gas; 0=not in halo
    star_select = (shalo==2)
    spos = pygr.readsnap(snap,'pos','star',units=1)[star_select]/h  # star positions in ckpc
    svel = pygr.readsnap(snap,'vel','star',units=1,suppress=1)[star_select] # star vels in physical km/s
    shalo = np.array(pygr.readsnap(snap,'HaloID','star'),dtype=int)[star_select]  # Halo ID of stars
    nstartot = np.uint64(len(spos))
    try:
        bhhalo = np.array(pygr.readsnap(snap,'HaloID','bndry'),dtype=int)  # Halo ID of gas; 0=not in halo
        bh_select = (bhhalo==2)
        bpos = pygr.readsnap(snap,'pos','bndry',units=1)[bh_select]/h  # BH positions in ckpc
        bvel = pygr.readsnap(snap,'vel','bndry',units=1,suppress=1)[bh_select] # BH vels in physical km/s
        bhalo = np.array(pygr.readsnap(snap,'HaloID','bndry'),dtype=int)[bh_select]  # Halo ID of BHs
        nbhtot = np.uint64(len(bpos))
    except: 
        print('fof6d : Creating one fake BH particle at origin (not in a halo) to avoid crash.')
        bpos = [[0,0,0]]
        bvel = [[0,0,0]]
        bhalo = [0]
        nbhtot = 0
    # set up combined arrays for positions and velocities, along with indexing
    pos = np.vstack((gpos,spos,bpos)).T  # transpose gives pos[0] as list of x positions, pos[1] as y, etc
    vel = np.vstack((gvel,svel,bvel)).T  # same for vel
    haloID = np.concatenate((ghalo,shalo,bhalo))  # compile list of halo IDs
    pindex = np.concatenate((pindex[gas_select],np.arange(ngastot,ngastot+nstartot+nbhtot,dtype=np.uint64))) 
    print('fof6d_old : Loaded %d (of %d) gas + %d stars + %g bh = %d total particles [t=%.2f s]'%(len(gpos),ngastot,len(spos),len(bpos),len(pindex),time.time()-t0))
    return pos,vel,haloID,pindex,ngastot,nstartot,nbhtot,gas_select

# Returns array galindex to original particles order in snapshot, and splits into separate arrays
# for gas, stars, and BHs.  Returns a list of galaxy IDs for all gas, stars, and BHs in
# snapshot, with galaxy ID = -1 for particles not in a fof6d galaxy
def reset_order_old(galindex,pindex,ngas,nstar,nbh,snap):
    import pygadgetreader as pygr
    # create galaxy ID arrays for gas, stars, BHs
    gindex = np.zeros(ngas,dtype=np.int64)-1
    sindex = np.zeros(nstar,dtype=np.int64)-1
    bindex = np.zeros(nbh,dtype=np.int64)-1
    # loop through all searched particles and place galaxy IDs into appropriate array
    for i in range(len(pindex)):
        if pindex[i] < ngas:
            gindex[pindex[i]] = galindex[i]
        elif pindex[i] < ngas+nstar:
            sindex[pindex[i]-ngas] = galindex[i]
        else:
            bindex[pindex[i]-(ngas+nstar)] = galindex[i]

    if False:  # a check for debugging
        ghalo = np.array(pygr.readsnap(snap,'HaloID','gas'),dtype=int)  # Halo ID of gas; 0=not in halo
        shalo = np.array(pygr.readsnap(snap,'HaloID','star'),dtype=int)  # Halo ID of gas; 0=not in halo
        bhalo = np.array(pygr.readsnap(snap,'HaloID','bndry'),dtype=int)  # Halo ID of gas; 0=not in halo
        assert len(ghalo)==len(gindex),'Gas lengths not equal! %d != %d'%(len(ghalo),len(gindex))
        assert len(shalo)==len(sindex),'Star lengths not equal! %d != %d'%(len(shalo),len(sindex))
        assert len(bhalo)==len(bindex),'BH lengths not equal! %d != %d'%(len(bhalo),len(bindex))
        for i in range(len(gindex)):
            if ghalo[i] == 0 and gindex[i] >= 0: sys.exit('Found particle in galaxy but not in halo! i=%d %d %d'%(i,ghalo[i],gindex[i]))
        for i in range(len(sindex)):
            if shalo[i] == 0 and sindex[i] >= 0: sys.exit('Found particle in galaxy but not in halo! i=%d %d %d'%(i,shalo[i],sindex[i]))
        for i in range(len(bindex)):
            if bhalo[i] == 0 and bindex[i] >= 0: sys.exit('Found particle in galaxy but not in halo! i=%d %d %d'%(i,bhalo[i],bindex[i]))

    return gindex,sindex,bindex

def periodic(x1,x2,L):  # periodic distance between scalars x1 and k2
    dx = x1-x2
    if dx>0.5*L: return L-dx
    else: return dx


#=========================================================
# 6DFOF ROUTINES
#=========================================================

def fof6d_old(igrp,groups,poslist,vellist,kerneltab,t0,Lbox,fof_LL,vel_LL=None,nfof=0):
    # find neighbors of all particles within fof_LL
    istart = groups[igrp][0]  # starting particle index for group igrp
    iend = groups[igrp][1]
    nactive = iend-istart
    if nactive < mingrp: return [0,[]]
    neigh = NearestNeighbors(radius=fof_LL)  # set up neighbor finder
    neigh.fit(poslist)  # do neighbor finding
    nlist = neigh.radius_neighbors(poslist)  # get neighbor properties (radii, indices)

    # compute velocity criterion for neighbors, based on local velocity dispersion 
    if vel_LL is not None:
        LLinv = 1./fof_LL
        sigma = np.zeros(nactive)
        siglist = []  # list of boolean arrays storing whether crit is satisfied for each neighbor pair
        # compute local velocity dispersion from neighbors
        for i in range(nactive):
            ngblist = nlist[1][i]  # list of indices of neighbors
            rlist = nlist[0][i]  # list of radii of neighbors
            # compute kernel-weighted velocity dispersion
            wt = kernel(rlist*LLinv,kerneltab)
            dv = np.linalg.norm(vellist[ngblist]-vellist[i],axis=1)
            sigma[i] = np.sqrt(np.sum(wt*dv*dv)/np.sum(wt))
            siglist.append((dv <= vel_LL*sigma[i]))
    else:
        # if velocity criterion not used, then all particles satisfy it by default
        siglist = []  # list of boolean arrays storing whether vel disp crit is satisfied
        for i in range(nactive):
            ngbnum = len(nlist[1][i])
            sigs = np.ones(ngbnum,dtype=bool)  # array of True's
            siglist.append(sigs)

    # determine counts within fof_LL, set up ordering of most dense to least
    ncount = np.zeros(nactive,dtype=int)
    for i in range(len(ncount)):
        ncount[i] = len(nlist[1][i])  # count number of neighbors for each particle
    dense_order = np.argsort(-ncount)  # find ordering of most dense to least

    # main loop to do FOF
    galind = np.zeros(nactive,dtype=int)-1
    linked = []
    galcount = 0
    for ipart in range(nactive):
        densest = dense_order[ipart]  # get next densest particle
        galind_ngb = galind[nlist[1][densest]]  # indices of neighbors' galaxies
        galind_ngb = np.where(siglist[densest],galind_ngb,-1)  # apply velocity criterion here
        if len(galind_ngb[galind_ngb>=0]) > 0:  # if it already has neighbors (incl iteself) in a galaxy...
            galmin = np.unique(galind_ngb[galind_ngb>=0])  # find sorted, unique indices of neighbors' gals
            galind[nlist[1][densest]] = min(galmin)  # put all neighbors in lowest-# galaxy
            for i in range(1,len(galmin)):  # link all other galaxies to lowest
                if linked[galmin[i]]==-1: 
                    linked[galmin[i]] = min(galmin)  # link all other galaxies to lowest index one
                else:
                    linked[galmin[i]] = min(linked[galmin[i]],min(galmin))  # connect all other galaxies to lowest index
        else:  # it has no neighbors in a galaxy, so create a new one
            galind[nlist[1][densest]] = galcount  # put all neighbors in a new galaxy
            linked.append(galcount) # new galaxy is linked to itself
            #print 'particle %d creating galaxy %d with neighbors %s'%(densest,galcount,nlist[1][densest])
            galcount += 1

    '''
    # Do final linking
    nreset = 1
    while nreset > 0:
        nreset = 0
        for ipart in range(len(ncount)):
            for k in range(len(nlist[1][ipart])):
                j = nlist[1][ipart][k]
                if galind[ipart] > galind[j]:
                    linked[galind[ipart]] = galind[j]
                    galind[ipart] = galind[j]
                    #print nlist[0][ipart]
                    #print nlist[1][ipart]
                    #print 'resetting part %d group to %d to match %d'%(ipart,galind[ipart],j)
                    nreset += 1
                    #print 'Trouble: Part %d (%s) in gal %d has neighbor %d (%d/%d) (%s) in gal %d, r=%g %d %d'%(ipart,poslist[ipart],galind[ipart],j,k,len(nlist[1][ipart]),poslist[j],galind[j],nlist[0][ipart][k],linked[galind[ipart]],linked[galind[j]])
                    #raw_input("Press Enter to continue ...")
    '''

    '''
    for ipart in range(nactive):  # check that objects don't have neighbors that should have been linked but aren't
        for k in range(len(nlist[1][ipart])):
            j = nlist[1][ipart][k]
            if galind[ipart] != galind[j]:
                print 'Trouble: Part %d in gal %d has neighbor %d (%d/%d) in gal %d, r=%g %d %d'%(ipart,galind[ipart],j,k,len(nlist[1][ipart]),galind[j],nlist[0][ipart][k],linked[galind[ipart]],linked[galind[j]])
    '''

    # handle linked galaxies by resetting indices of their particles to its linked galaxy
    for i in range(galcount-1,-1,-1):
        if linked[i] != i:
            assert linked[i]<i,'Trouble: mis-ordered linking %d > %d'%(i,linked[i])
            for j in range(iend-istart):
                if galind[j] == i: galind[j] = linked[i]

    # assign indices of particles to FOF groups having more than mingrp particles
    pcount,bin_edges = np.histogram(galind,bins=galcount)  # count particles in each galaxy
    for i in range(iend-istart):  # set indices of particles in groups with <mingrp members to -1
        if pcount[galind[i]] < mingrp: galind[i] = -1  
    if len(galind[galind>=0])==0: return 0,galind  # if there are no valid groups left, return
    galind_unique = np.unique(galind[galind>=0])  # find unique groups
    galind_inv = np.zeros(max(galind_unique)+1,dtype=int)
    for i in range(len(galind_unique)):
        galind_inv[galind_unique[i]] = i  # create mapping from original groups to unique set
    for i in range(iend-istart): 
        if galind[i]>=0: galind[i] = galind_inv[galind[i]] # re-assign group indices sequentially
    galcount = max(galind)+1
    #raw_input('press ENTER to continue')

    '''
    # check: are there groups that are too large?
    for i in range(galcount):
        npgal = len(galind[galind==i])
        galpos = np.array([poslist[j] for j in range(len(poslist)) if galind[j]==i])
        galpos = galpos.T
        #print npgal,galpos
        galsize = np.array([max(galpos[0])-min(galpos[0]),max(galpos[1])-min(galpos[1]),max(galpos[2])-min(galpos[2])])
        galsize = np.where(galsize>Lbox/2,Lbox-galsize,galsize)
        toolarge = 300
        if nfof < 10 and nfof > 0: print 'in fof6d:',nfof+i,i,npgal,np.mean(galpos[0]),np.mean(galpos[1]),np.mean(galpos[2]),galsize
        if galsize[0]>toolarge or galsize[1]>toolarge or galsize[2]>toolarge: print 'Too large?',igrp,iend-istart,galsize,galpos.T
    '''

    # Compile result to return: number of new groups found, and a galaxy index for particles from istart:iend
    result = [galcount]
    result.append(galind)
    progress_bar(1.*groups[igrp][1]/groups[len(groups)-1][1],barLength=50,t=time.time()-t0)

    return result

#=========================================================
# FOFRAD ROUTINES
#=========================================================

# sort particles by position in direction idir0, for particles from istart:iend
def sort_parts(pos,vel,haloID,pindex,istart,iend,idir0,key='pos'):
    if key == 'pos':
        sort_ind = np.argsort(pos[idir0][istart:iend])  # sort in desired direction
    elif key == 'haloID':
        sort_ind = np.argsort(haloID[istart:iend])  # sort by halo ID (idir0 is irrelevant)
    idir1 = (idir0+1)%3  # these are the other two directions
    idir2 = (idir0+2)%3
    pos[idir0][istart:iend] = pos[idir0][istart:iend][sort_ind]  # keep all arrays likewise sorted 
    pos[idir1][istart:iend] = pos[idir1][istart:iend][sort_ind]
    pos[idir2][istart:iend] = pos[idir2][istart:iend][sort_ind]
    vel[idir0][istart:iend] = vel[idir0][istart:iend][sort_ind]
    vel[idir1][istart:iend] = vel[idir1][istart:iend][sort_ind]
    vel[idir2][istart:iend] = vel[idir2][istart:iend][sort_ind]
    haloID[istart:iend] = haloID[istart:iend][sort_ind]  
    pindex[istart:iend] = pindex[istart:iend][sort_ind]  # keep particle indices in the same order

def fof_sorting_old(groups,pos,vel,haloID,pindex,fof_LL,Lbox,idir,mingrp=16):
    oldgroups = groups[:]  # stores the groups found from the previous sorting direction
    npart = len(pindex)
    groups = [[0,npart]]  # (re-)initialize the group as containing all particles
    grpcount = 0
    for igrp in range(len(oldgroups)):  # loop over old groups, sort within each, find new groups
        # sort particles within group in given direction
        istart = oldgroups[igrp][0]  # starting particle index for group igrp
        iend = oldgroups[igrp][1]
        sort_parts(pos,vel,haloID,pindex,istart,iend,idir,'pos')  # sort group igrp in given direction
        # create new groups by looking for breaks of dx[idir]>fof_LL between particle positions
        oldpos = pos[idir][istart]
        groups[grpcount][0] = istart  # set start of new group to be same as for old group
        for i in range(istart,iend):
            if periodic(pos[idir][i],oldpos,Lbox) > fof_LL or i == iend-1:  
                groups[grpcount][1] = i  # set end of old group to be i
                groups.append([i,i])  # add new group, starting at i; second index is a dummy that will be overwritten when the end of the group is found
                grpcount += 1
            oldpos = pos[idir][i]
    #assert grpcount>0,'fof6d : Unable to separate groups via sorting; exiting.'
    # Remove groups that have less than mingrp particles (gas+star)
    oldgroups = groups[:]
    groups = []
    for igrp in range(len(oldgroups)):
        istart = oldgroups[igrp][0]  
        iend = oldgroups[igrp][1]
        if iend-istart >= mingrp: groups.append(oldgroups[igrp])
    return groups

def fofrad_old(snap,nproc,mingrp,LL_factor,vel_LL):
    import pygadgetreader as pygr
    t0 = time.time()
    Lbox = pygr.readheader(snap,'boxsize')
    h = pygr.readheader(snap,'h')
    Lbox = Lbox/h  # to kpc
    #Omega = pygr.readheader(snap,'O0')
    #Lambda = pygr.readheader(snap,'Ol')
    n_side = int(pygr.readheader(snap,'dmcount')**(1./3.)+0.5)
    MIS = Lbox/n_side
    fof_LL = LL_factor*MIS
    print('fof6d : Guessing %d particles per side, fof_LL=%g ckpc' % (n_side,fof_LL))
    kerneltab = kernel_table(fof_LL)

    # Load particles positions and velocities from snapshot
    pos,vel,haloID,pindex,ngastot,nstartot,nbhtot,gas_select = loadsnap(snap,t0)

    # initialize groups as containing all the particles in each halo
    npart = len(pindex)
    sort_parts(pos,vel,haloID,pindex,0,npart,0,'haloID')
    groups = []
    hstart = 0
    for i in range(1,len(haloID)):  # set up groups[]
        if haloID[i] == 0: 
            hstart = i
            continue
        if haloID[i]>haloID[i-1] and haloID[i-1]>0:
            if i-hstart>=mingrp: groups.append([hstart,i])
            hstart = i
    '''
    # use all particles, not just ones in halos
    groups = [[0,npart]]  # initial group has the entire list of particles
    '''

    # within each group (i.e. halo), create sub-groups of particles via directional sorting, ala FOFRAD
    print('fof6d_old : FOF via sorting beginning with %d halos/groups [t=%.2f s]'%(len(groups),time.time()-t0))
    for idir in range(len(pos)):  # sort in each direction, find groups within sorted list
        groups = fof_sorting_old(groups,pos,vel,haloID,pindex,fof_LL,Lbox,idir)
        print('fof6d_old : Axis %d approx FOF via sorting found %d groups [t=%.2f s]'%(idir,len(groups),time.time()-t0))
        if(len(groups)==0): break

    # for each sub-group, do a proper 6D FOF search to find galaxies
    galindex = np.zeros(npart,dtype=int)-1
    results = [None]*len(groups)
    if nproc > 1:
        npart_par = 2000 # optimization: groups with > this # of parts done via Parallel()
        igpar = []
        igser = []
        for igrp in range(len(groups)):  
            if groups[igrp][1]-groups[igrp][0] > npart_par:
                igpar.append(igrp)
            else: 
                igser.append(igrp)
        igpar = np.array(igpar,dtype=int)
        print('fof6d : Doing %d groups with npart<%d on single core [t=%.2f s]'%(len(igser),npart_par,time.time()-t0))
        for igrp in igser: results[igrp] = fof6d_old(igrp,groups,pos.T[groups[igrp][0]:groups[igrp][1]],vel.T[groups[igrp][0]:groups[igrp][1]],kerneltab,t0,Lbox,fof_LL,vel_LL)
        print('\nfof6d : Doing %d groups with npart>%d on %d cores (progressbar approximate) [t=%.2f s]'%(len(igpar),npart_par,nproc,time.time()-t0))
        if len(igpar)>0: 
            from joblib import Parallel, delayed
            results_par = Parallel(n_jobs=nproc)(delayed(fof6d)(igrp,groups,pos.T[groups[igrp][0]:groups[igrp][1]],vel.T[groups[igrp][0]:groups[igrp][1]],kerneltab,t0,Lbox,fof_LL,vel_LL) for igrp in igpar)

        for i in range(len(igpar)): results[igpar[i]] = results_par[i]
    else:
        print('fof6d : Doing %d groups on single core [t=%.2f s]'%(len(groups),time.time()-t0))
        for igrp in range(len(groups)): 
            results[igrp] = fof6d_old(igrp,groups,pos.T[groups[igrp][0]:groups[igrp][1]],vel.T[groups[igrp][0]:groups[igrp][1]],kerneltab,t0,Lbox,fof_LL,vel_LL)

    # insert galaxy IDs into particle lists
    nfof = 0
    for igrp in range(len(groups)):
        result = results[igrp]
        istart = groups[igrp][0]  # starting particle index for group igrp
        iend = groups[igrp][1]
        galindex[istart:iend] = np.where(result[1]>=0,result[1]+nfof,-1)  # for particles in groups, increment group ID with counter (nfof)
        #if nfof<1 and len(result[1][result[1]>=0])>0: 
        #        print 'inserting IDs',igrp,nfof,result[0],groups[igrp][1]-groups[igrp][0]
        #        for igal in range(result[0]):
        #            print 'galaxies found:',igal+nfof,len(result[1][result[1]==igal]),np.mean(posgrp[result[1]==igal].T[0]),np.mean(posgrp[result[1]==igal].T[1]),np.mean(posgrp[result[1]==igal].T[2])
        #            print len(posgrp[result[1]==igal]),posgrp[result[1]==igal]
        nfof += result[0]

    gindex,sindex,bindex = reset_order_old(galindex,pindex,ngastot,nstartot,nbhtot,snap)

    if 0:
        spos = pygr.readsnap(snap,'pos','star',units=1)/h  # star positions in ckpc
        for ifof in range(nfof):
            sgrp = spos[ifof==sindex]
            sgrp = sgrp.T
            if ifof<2: 
                print('final check',ifof,np.mean(sgrp[0]),np.mean(sgrp[1]),np.mean(sgrp[2]),max(sgrp[0])-min(sgrp[0]),max(sgrp[1])-min(sgrp[1]),max(sgrp[2])-min(sgrp[2]))
                print (len(sgrp.T),sgrp.T)

    print('\nfof6d: Found %d galaxies, with %d gas+%d star+%d BH = %d particles [t=%.2f s]'%(nfof,len(gindex[gindex>=0]),len(sindex[sindex>=0]),len(bindex[bindex>=0]),len(gindex[gindex>=0])+len(sindex[sindex>=0])+len(bindex[bindex>=0]),time.time()-t0))

    return gindex,sindex,bindex,t0

#=========================================================
# MAIN DRIVER ROUTINE
#=========================================================

def run_fof_6d(snapfile,mingrp,LL_factor,vel_LL,nproc):

    #pdb.set_trace()
    #snapfile = '%s/snapshot_%03d.hdf5'%(BASEDIR,int(snapnum))
    if not os.path.isfile(snapfile):
        sys.exit('Snapfile %s does not exist'%snapfile)
    else: print('fof6d : Doing snapfile: %s'%snapfile)

# Set up multiprocessing
    import multiprocessing
    if nproc == 0:   # use all available cores
        num_cores = multiprocessing.cpu_count()
    if nproc != 1:   # if multi-core, set up Parallel processing
        num_cores = multiprocessing.cpu_count()
        if nproc < 0: print('fof6d : Using %d cores (all but %d)'%(num_cores+nproc+1,-nproc-1) )
        if nproc > 1: 
            print('progen : Using %d of %d cores'%(nproc,num_cores))
        else: print('progen : Using single core')
        if nproc>8: print('fof6d : FYI you are using nproc=%d. nproc>8 tends to give minimal or negative benefit.'%nproc)

    # find friends of friends groups 
    gas_index,star_index,bh_index,t0 = fofrad_old(snapfile,nproc,mingrp,LL_factor,vel_LL)   # returns galaxy indices for *all* gas, stars, bh
    

    #return statements
    nparts = np.array([len(gas_index),len(star_index),len(bh_index)])
    return nparts,gas_index,star_index,bh_index
    
    '''
    # output csv file to be read into Caesar
    outfile = '%s/Groups/fof6d_%03d.hdf5'%(BASEDIR,int(snapnum))
    with h5py.File(outfile, 'w') as hf:
        nparts = np.array([len(gas_index),len(star_index),len(bh_index)])
        hf.create_dataset('nparts',data=nparts)
        hf.create_dataset('gas_index',data=gas_index)
        hf.create_dataset('star_index',data=star_index)
        hf.create_dataset('bh_index',data=bh_index)
    print('fof6d : Outputted galaxy IDs for gas, stars, and BHs to %s -- FOF6D DONE [t=%.2f s]'%(outfile,time.time()-t0))

   '''
