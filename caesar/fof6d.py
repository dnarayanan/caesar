# 6-D FOF.
# Input is a snapshot file containing gas, stars, and BHs.
# Snapshot must contain a HaloID putting each particle in a halo (0=not in halo).
# First groups particles via an approximate FOF using  via quicksorts in successive directions.
# Then does a full 6-D FOF search on the approximate FOF groups
# Outputs SKID-format .grp binary file containing Npart, then galaxy IDs of all particles
#
# Romeel Dave, 25 Feb 2019

from pygadgetreader import *
import numpy as np
from sklearn.neighbors import NearestNeighbors
from astropy import constants as const
from astropy import units as u
# time and sys are only used for the progressbar. They should be replaced with the one provided by yt or tqdm
import time
import sys

nproc = 1

# FOF options
mingrp = 16
LL_factor = 0.02  # linking length is this times the mean interparticle spacing 
vel_LL = 1.0  # velocity space linking length factor, multiplies local velocity dispersion

#=========================================================
# MISCELLANEOUS ROUTINES
#=========================================================

# Loads gas, star, BH information from snapshot.  Requires HaloID for all particles.
def loadsnap(snap,t0):
    redshift = readheader(snap,'redshift')
    h = readheader(snap,'h')

    # get gas Halo IDs so we can select only gas particles in halos
    ghalo = np.array(readsnap(snap,'HaloID','gas'),dtype=int)  # Halo ID of gas; 0=not in halo
    ngastot = len(ghalo)
    gas_select = (ghalo>0)
    ngastot = np.uint64(len(ghalo))
    pindex = np.arange(ngastot,dtype=np.uint64)  # keep an index for the original order of the particles
    # Load in gas info for selecting gas particles
    gnh = readsnap(snap,'rho','gas',units=1)[gas_select]*h*h*0.76*(1+redshift)**3/const.m_p.to('g').value   # number density in phys H atoms/cm^3
    gsfr = readsnap(snap,'sfr','gas',units=1)[gas_select]
    gtemp = readsnap(snap,'u','gas',units=1)[gas_select]  # temperature in K
    # Apply additional selection criteria on gas
    dense_gas_select = ((gnh>0.13)&((gtemp<1.e5)|(gsfr>0))) # dense, cool, or SF gas only 
    gas_select[gas_select>0] = (gas_select[gas_select>0] & dense_gas_select)
    # load in selected gas 
    
    gpos = readsnap(snap,'pos','gas',units=1)[gas_select]/h  # positions in ckpc
    gvel = readsnap(snap,'vel','gas',units=1,suppress=1)[gas_select] # vel in physical km/s
    ghalo = np.array(readsnap(snap,'HaloID','gas'),dtype=int)[gas_select]  # Halo ID of gas; 0=not in halo
    # load in all stars+BHs; don't bother with halo selection since most will be in halos anyways
    spos = readsnap(snap,'pos','star',units=1)/h  # star positions in ckpc
    svel = readsnap(snap,'vel','star',units=1,suppress=1) # star vels in physical km/s
    shalo = np.array(readsnap(snap,'HaloID','star'),dtype=int)  # Halo ID of stars
    nstartot = np.uint64(len(spos))
    try:
        bpos = readsnap(snap,'pos','bndry',units=1)/h  # BH positions in ckpc
        bvel = readsnap(snap,'vel','bndry',units=1,suppress=1) # BH vels in physical km/s
        bhalo = np.array(readsnap(snap,'HaloID','bndry'),dtype=int)  # Halo ID of BHs
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
    print('fof6d : Loaded %d (of %d) gas + %d stars + %g bh = %d total particles [t=%.2f s]'%(len(gpos),ngastot,len(spos),len(bpos),len(pindex),time.time()-t0))
    return pos,vel,haloID,pindex,ngastot,nstartot,nbhtot,gas_select

# Returns array galindex to original particles order in snapshot, and splits into separate arrays
# for gas, stars, and BHs.  Returns a list of galaxy IDs for all gas, stars, and BHs in
# snapshot, with galaxy ID = -1 for particles not in a fof6d galaxy
def reset_order(galindex,pindex,ngas,nstar,nbh,snap):
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

    return gindex,sindex,bindex

def periodic(x1,x2,L):  # periodic distance between scalars x1 and k2
    dx = x1-x2
    if dx>0.5*L: return L-dx
    else: return dx

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
    if t==None: text = "\r[{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), np.round(progress*100,2), status)
    else: text = "\r[{0}] {1}% [t={2} s] {3}".format( "#"*block + "-"*(barLength-block), np.round(progress*100,2), np.round(t,2), status)
    sys.stdout.write(text)
    sys.stdout.flush()


#=========================================================
# 6DFOF ROUTINES
#=========================================================

# set up kernel table
def kernel_table(h,ntab=1000):
    kerneltab = np.zeros(ntab+1)
    hinv = 1./h
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

def fof6d(igrp,groups,poslist,vellist,kerneltab,t0,Lbox,fof_LL,vel_LL=None,nfof=0):
    # find neighbors of all particles within fof_LL
    istart = groups[igrp][0]  # starting particle index for group igrp
    iend = groups[igrp][1]
    nactive = iend-istart
    if nactive < mingrp: return [0,[]]
    neigh = NearestNeighbors(radius=fof_LL)  # set up neighbor finder
    neigh.fit(poslist)  # do neighbor finding
    nlist = neigh.radius_neighbors(poslist)  # get neighbor properties (radii, indices)

    # compute velocity criterion for neighbors, based on local velocity dispersion 
    if vel_LL != None:
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
    counter = 0
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

def fof_sorting(groups,pos,vel,haloID,pindex,fof_LL,Lbox,idir):
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
                posrange = [min(pos[idir][groups[grpcount][0]:i]),max(pos[idir][groups[grpcount][0]:i])]
                grpcount += 1
            oldpos = pos[idir][i]
    assert grpcount>1,'fof6d : Unable to separate groups via sorting; exiting.'
    # Remove groups that have less than mingrp particles (gas+star)
    oldgroups = groups[:]
    groups = []
    for igrp in range(len(oldgroups)):
        istart = oldgroups[igrp][0]  
        iend = oldgroups[igrp][1]
        if iend-istart >= mingrp: groups.append(oldgroups[igrp])
    return groups

def fofrad(snap,nproc,mingrp,LL_factor,vel_LL):
    t0 = time.time()
    Lbox = readheader(snap,'boxsize')
    h = readheader(snap,'h')
    Lbox = Lbox/h  # to kpc
    n_side = int(readhead(snap,'dmcount')**(1./3.)+0.5)
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

    # within each group (i.e. halo), create sub-groups of particles via directional sorting, ala FOFRAD
    print('fof6d : FOF via sorting beginning with %d halos/groups [t=%.2f s]'%(len(groups),time.time()-t0))
    for idir in range(len(pos)):  # sort in each direction, find groups within sorted list
        groups = fof_sorting(groups,pos,vel,haloID,pindex,fof_LL,Lbox,idir)
        print('fof6d : Axis %d approx FOF via sorting found %d groups [t=%.2f s]'%(idir,len(groups),time.time()-t0))

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
        for igrp in igser: results[igrp] = fof6d(igrp,groups,pos.T[groups[igrp][0]:groups[igrp][1]],vel.T[groups[igrp][0]:groups[igrp][1]],kerneltab,t0,Lbox,fof_LL,vel_LL)
        print('\nfof6d : Doing %d groups with npart>%d on %d cores (progressbar approximate) [t=%.2f s]'%(len(igpar),npart_par,nproc,time.time()-t0))
        if len(igpar)>0: 
            import multiprocessing
            from joblib import Parallel, delayed
            results_par = Parallel(n_jobs=nproc)(delayed(fof6d)(igrp,groups,pos.T[groups[igrp][0]:groups[igrp][1]],vel.T[groups[igrp][0]:groups[igrp][1]],kerneltab,t0,Lbox,fof_LL,vel_LL) for igrp in igpar)

        for i in range(len(igpar)): results[igpar[i]] = results_par[i]
    else:
        print('fof6d : Doing %d groups on single core [t=%.2f s]'%(len(groups),time.time()-t0))
        for igrp in range(len(groups)): 
            results[igrp] = fof6d(igrp,groups,pos.T[groups[igrp][0]:groups[igrp][1]],vel.T[groups[igrp][0]:groups[igrp][1]],kerneltab,t0,Lbox,fof_LL,vel_LL)

    # insert galaxy IDs into particle lists
    nfof = 0
    for igrp in range(len(groups)):
        result = results[igrp]
        istart = groups[igrp][0]  # starting particle index for group igrp
        iend = groups[igrp][1]
        posgrp = pos.T[groups[igrp][0]:groups[igrp][1]]
        galindex[istart:iend] = np.where(result[1]>=0,result[1]+nfof,-1)  # for particles in groups, increment group ID with counter (nfof)
        nfof += result[0]

    gindex,sindex,bindex = reset_order(galindex,pindex,ngastot,nstartot,nbhtot,snap)

    if 0:
        spos = readsnap(snap,'pos','star',units=1)/h  # star positions in ckpc
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

    print('fof6d : Doing snapfile: %s'%snapfile)

# Set up multiprocessing
    if nproc == 0:   # use all available cores
        num_cores = multiprocessing.cpu_count()
    if nproc != 1:   # if multi-core, set up Parallel processing
        import multiprocessing
        from joblib import Parallel, delayed
        num_cores = multiprocessing.cpu_count()
        if nproc < 0: print('fof6d : Using %d cores (all but %d)'%(num_cores+nproc+1,-nproc-1) )
        if nproc > 1: 
            print('progen : Using %d of %d cores'%(nproc,num_cores))
        else: print('progen : Using single core')
        if nproc>8: print('fof6d : FYI you are using nproc=%d. nproc>8 tends to give minimal or negative benefit.'%nproc)

    # find friends of friends groups 
    gas_index,star_index,bh_index,t0 = fofrad(snapfile,nproc,mingrp,LL_factor,vel_LL)   # returns galaxy indices for *all* gas, stars, bh
    

    #return statements
    nparts = np.array([len(gas_index),len(star_index),len(bh_index)])
    return nparts,gas_index,star_index,bh_index

