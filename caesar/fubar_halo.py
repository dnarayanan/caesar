import numpy as np
import h5py
from caesar.group import create_new_group, group_types
from caesar.property_manager import get_property, get_particles_for_FOF, get_high_density_gas_indexes
from caesar.property_manager import ptype_ints
from caesar.utils import calculate_local_densities
from caesar.fof6d import *

import six
from yt.funcs import mylog
from yt.extern.tqdm import tqdm
from yt.units.yt_array import uconcatenate, YTArray
from yt.data_objects.octree_subset import YTPositionArray
from yt.utilities.lib.contour_finding import ParticleContourTree
from yt.geometry.selection_routines import AlwaysSelector


def fubar_halo(obj):
    """Halo-by-halo group finding procedure.
    Romeel Davé March 2020

    FUBAR stands for Friends-of-friends Unbinding after Rockstar; the
    name is no longer valid, but it stuck.  Here we perform an FOF
    operation for each grouping and create the master caesar lists.

    For halos we consider dark matter + gas + stars.  For galaxies
    however, we only consider high density gas and stars (dust and
    blackholes if included).

    For clouds we consider all gas particles.
    
    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Main caesar object.
    group_type : str
        Can be either 'halo', 'galaxy' or 'cloud'; determines what objects
        we find with FOF.

    """
 
    #pdb.set_trace()

    from caesar.fof6d import fof6d
    from caesar.group import get_group_properties
    from caesar.fubar import get_mean_interparticle_separation

    # Process halos
    halos = fof6d(obj,'halo')  #instantiate a fof6d object
    halos.MIS = get_mean_interparticle_separation(obj).d # also computes omega_baryon and related quantities
    halos.load_haloid()
    halos.obj.data_manager._member_search_init(select=halos.haloid)  # load particle info, but only those selected to be in a halo
    halos.plist_init()  # sort halo IDs
    halos.load_lists()  # create halos, load particle indexes for halos
    get_group_properties(halos,halos.obj.halo_list)  # compute halo properties

    # Find galaxies, or load galaxy membership info
    if not obj.simulation.baryons_present: return  # if no baryons, we're done
    if 'fof6d' in obj._kwargs and obj._kwargs['fof6d']:
        fof6d_flag = True
    else:
        return  # no galaxy finder specified; currently fof6d is the only option
    if ('fof6d_file' in obj._kwargs and obj._kwargs['fof6d_file'] is not None):
        fof6d_flag = halos.load_fof6dfile()  # load galaxy ID's from fof6d_file
    if fof6d_flag:
        halos.run_fof6d('galaxy')  # run fof6d on halos to find galaxies
        halos.save_fof6dfile()  # save fof6d info
        #snapname = ('%s/%s'%(obj.simulation.fullpath,obj.simulation.basename))
        #nparts,gas_index,star_index,bh_index = run_fof_6d(snapname,16,0.02,1.0,nproc)

    # Process galaxies
    galaxies = fof6d(obj,'galaxy')  #instantiate a fof6d object
    galaxies.plist_init(parent=halos)  # get particle list for computing galaxy properties
    if galaxies.nparttot == 0: return  # plist_init didn't find enough particles to group
    galaxies.load_lists(parent=halos)  # create galaxy_list, load particle index lists for galaxies
    get_group_properties(galaxies,galaxies.obj.galaxy_list)  # compute galaxy properties

    # Find and process clouds
    if ('fofclouds' in obj._kwargs) and obj._kwargs['fofclouds']:
        galaxies.run_fof6d('cloud')  # run fof6d to find cloudid's
        galaxies.load_lists('cloud')  # load particle index lists for galaxies
        clouds = fof6d(obj,'cloud')  #instantiate a fof6d object
        clouds.plist_init(parent=galaxies)  # initialize comptutation of cloud properties
        if clouds.nparttot == 0: return  # plist_init didn't find enough particles to group
        get_group_properties(clouds)  # compute cloud properties
        galaxies.load_lists('cloud')
   
    # reset particle lists to have original snapshot ID's; must do this after all group processing is finished
    reset_global_particle_IDs(obj)
    # load global lists
    load_global_lists(obj)

    return


plist_dict = dict( gas='glist', star='slist', bh='bhlist', dust='dlist', dm='dmlist', dm2='dm2list')

def reset_global_particle_IDs(obj):
    ''' Maps particle lists from currently loaded ID's to the ID's corresponding to the full snapshot '''

    # determine full particle numbers in snapshot
    from caesar.property_manager import has_ptype, get_property
    offset = np.zeros(len(obj.ptypes)+1,dtype=np.int64)
    for ip,p in enumerate(obj.ptypes):
        if not has_ptype(obj, p):
            continue
        count = len(get_property(obj, 'mass', p))
        if p == 'gas': 
            offset[ip+1] = offset[ip] + obj.simulation.ngas
            obj.simulation.ngas = count
        elif p == 'star': 
            offset[ip+1] = offset[ip] + obj.simulation.nstar
            obj.simulation.nstar = count
        elif p == 'bh': 
            offset[ip+1] = offset[ip] + obj.simulation.nbh
            obj.simulation.nbh = count
        elif p == 'dust': 
            offset[ip+1] = offset[ip] + obj.simulation.ndust
            obj.simulation.ndust = count
        elif p == 'dm': 
            offset[ip+1] = offset[ip] + obj.simulation.ndm
            obj.simulation.ndm = count
        elif p == 'dm2': 
            offset[ip+1] = offset[ip] + obj.simulation.ndm2
            obj.simulation.ndm2 = count

    # reset lists
    for group_type in obj.group_types:
        group_list = 'obj.%s_list'%group_type
        for ip,p in enumerate(obj.ptypes):
            if not has_ptype(obj, p):
                continue
            for group in eval(group_list):
                part_list = 'group.%s'%plist_dict[p]
                mylist = eval(part_list)
                mylist = obj.data_manager.indexes[mylist+offset[ip]]
                if p == 'gas': group.glist = mylist
                if p == 'star': group.slist = mylist
                if p == 'bh': group.bhlist = mylist
                if p == 'dust': group.dlist = mylist
                if p == 'dm': group.dmlist = mylist
                if p == 'dm2': group.dm2list = mylist

    return

def load_global_lists(obj):
    # initialize global particle lists
    from caesar.property_manager import has_ptype

    for group_type in obj.group_types:
        glist  = np.full(obj.simulation.ngas, -1, dtype=np.int32)
        slist  = np.full(obj.simulation.nstar, -1, dtype=np.int32)
        bhlist = np.full(obj.simulation.nbh, -1, dtype=np.int32)
        dlist  = np.full(obj.simulation.ndust, -1, dtype=np.int32)
        dmlist = np.full(obj.simulation.ndm, -1, dtype=np.int32)
        dm2list = np.full(obj.simulation.ndm2, -1, dtype=np.int32)

        group_list = 'obj.%s_list'%group_type
        for group in eval(group_list):
            for ip,p in enumerate(obj.ptypes):
                if not has_ptype(obj, p):
                    continue
                part_list = 'group.%s'%plist_dict[p]
                if p == 'gas': glist[eval(part_list)] = group.GroupID
                if p == 'star': slist[eval(part_list)] = group.GroupID
                if p == 'bh': bhlist[eval(part_list)] = group.GroupID
                if p == 'dust': dlist[eval(part_list)] = group.GroupID
                if p == 'dm': dmlist[eval(part_list)] = group.GroupID
                if p == 'dm2': dm2list[eval(part_list)] = group.GroupID

        setattr(obj.global_particle_lists, '%s_glist'  % group_type, glist)
        setattr(obj.global_particle_lists, '%s_slist'  % group_type, slist)
        setattr(obj.global_particle_lists, '%s_bhlist' % group_type, bhlist)
        setattr(obj.global_particle_lists, '%s_dlist'  % group_type, dlist)
        setattr(obj.global_particle_lists, '%s_dmlist' % group_type, dmlist)
        setattr(obj.global_particle_lists, '%s_dm2list'% group_type, dm2list)

    return


###################################################################################

    ''' DEFUNCT STUFF (see fubar.py) '''

    if group_type == 'galaxy':


        if ('fof6d' in obj._kwargs and obj._kwargs['fof6d'] == True):

            #set default parameters
            mingrp = 16
            LL_factor = 0.02
            vel_LL=1.0
            nproc = 1
            LL = get_mean_interparticle_separation(obj) * get_b(obj, group_type)  # get MIS and omega_baryon
            if ('fof6d_mingrp' in obj._kwargs and obj._kwargs['fof6d_mingrp'] is not None):
                mingrp = obj._kwargs['fof6d_mingrp']
            if ('fof6d_LL_factor' in obj._kwargs and obj._kwargs['fof6d_LL_factor'] is not None):
                LL_factor = obj._kwargs['fof6d_LL_factor']
            if ('fof6d_vel_LL' in obj._kwargs and obj._kwargs['fof6d_vel_LL'] is not None):
                vel_LL = obj._kwargs['fof6d_vel_LL']
            if ('nproc' in obj._kwargs and obj._kwargs['nproc'] is not None):
                nproc = obj._kwargs['nproc']

            snapname = ('%s/%s'%(obj.simulation.fullpath,obj.simulation.basename))
            mylog.info("Running FOF6D")
            nparts,gas_index,star_index,bh_index = run_fof_6d(snapname,mingrp,LL_factor,vel_LL,nproc)
            fof_tags = np.concatenate((gas_index,star_index,bh_index))
            high_rho_indexes = get_high_density_gas_indexes(obj)
            if ('fof6d_outfile' in obj._kwargs):
                fof6d_file = obj._kwargs['fof6d_outfile']
                mylog.info('Writing fof6d particle group info to %s' % fof6d_file)
                with h5py.File(fof6d_file,'w') as hf:  # overwrites existing fof6d group file
                    hf.create_dataset('nparts',data=nparts, compression=1)
                    hf.create_dataset('gas_index',data=gas_index, compression=1)
                    hf.create_dataset('star_index',data=star_index, compression=1)
                    hf.create_dataset('bh_index',data=bh_index, compression=1)
                    hf.close()
            #assert(obj.simulation.ngas == len(gas_index)) & (obj.simulation.nstar == len(star_index)) & (obj.simulation.nbh == len(bh_index)),'[fubar/fubar]: Assertion failed: Wrong number of particles in fof6d calculation'
            
        elif ('fof6d_file' in obj._kwargs and obj._kwargs['fof6d_file'] is not None):
            # use galaxy info from fof6d hdf5 file
            fof6d_file = obj._kwargs['fof6d_file']
            LL = get_mean_interparticle_separation(obj) * get_b(obj, group_type)  # get MIS and omega_baryon
            import os
            if os.path.isfile(fof6d_file):
                mylog.info('Galaxy IDs from fof6d file %s'%fof6d_file)
            else:
                mylog.info('fof6d file %s not found!' % fof6d_file)
            hf = h5py.File(fof6d_file,'r')
            npfof6d = hf['nparts']
            assert (obj.simulation.ngas==npfof6d[0])&(obj.simulation.nstar==npfof6d[1])&(obj.simulation.nbh==npfof6d[2]),'Assertion failed: Wrong number of particles in fof6d file: %s'%npfof6d
            gas_indexes = hf['gas_index']
            star_indexes = hf['star_index']
            bh_indexes = hf['bh_index']
            fof_tags = np.concatenate((gas_indexes,star_indexes,bh_indexes))

        else: 
            # here we want to perform 3D FOF on high density gas + stars
            mylog.info('Groups based on YT 3DFOF')
            high_rho_indexes = get_high_density_gas_indexes(obj)
            pos0 = pos
            pos  = np.concatenate(( pos0[obj.data_manager.glist][high_rho_indexes], pos0[obj.data_manager.slist]))
            if obj.data_manager.blackholes:
                pos  = np.concatenate(( pos, pos0[obj.data_manager.bhlist]))
            if obj.data_manager.dust:
                pos  = np.concatenate(( pos, pos0[obj.data_manager.dlist]))
            LL = get_mean_interparticle_separation(obj) * get_b(obj, group_type)
            fof_tags = fof(obj, pos, LL, group_type=group_type)
            gtags = np.full(obj.simulation.ngas, -1, dtype=np.int64)
            gtags[high_rho_indexes] = fof_tags[0:len(high_rho_indexes)]
            fof_tags = np.concatenate((gtags,fof_tags[len(high_rho_indexes)::]))

    elif group_type == 'cloud':

        #don't run if there's no baryons
        if not obj.simulation.baryons_present:
            return
            
        #also don't run if fofclouds isn't set
        if ('fofclouds' not in obj._kwargs) or (obj._kwargs['fofclouds'] == False):
            mylog.warning('No clouds: fofclouds either not set, or is set to false: not performing 3D group search for GMCs')
            return
        
        # here we want to perform FOF on all gas
        pos = pos[obj.data_manager.glist]
        LL = get_mean_interparticle_separation(obj) * get_b(obj, group_type)
        if ('ll_cloud' in obj._kwargs) and isinstance(obj._kwargs['ll_cloud'],(int,float)):
            LL = obj._ds.quan(float(obj._kwargs['ll_cloud']),'kpccm')
        fof_tags = fof(obj, pos, LL, group_type=group_type)

    elif group_type == 'halo':
        if ('fof_from_snap' in obj._kwargs and obj._kwargs['fof_from_snap']==1):
            mylog.info('Using Halo fof ID from snapshots')
            fof_tags = obj.data_manager.haloid - 1
        else:
            LL = get_mean_interparticle_separation(obj) * get_b(obj, group_type)
            fof_tags = fof(obj, pos, LL, group_type=group_type, **kwargs)

    else: 
        mylog.warning('group-type %s not recognized'%group_type)


    tag_sort = np.argsort(fof_tags)

    unique_groupIDs = np.unique(fof_tags)
    groupings = {}
    for GroupID in unique_groupIDs:
        if GroupID < 0: continue
        groupings[GroupID] = create_new_group(obj, group_type)

    if len(groupings) == 0:
        mylog.warning('No %s found!' % group_types[group_type])
        return
    
    tags = fof_tags
        
    nparts = len(tags)
    for i in range(0,nparts):
        index = tag_sort[i]
        tag   = tags[index]
        if tag < 0: continue
        groupings[tag]._append_global_index(index)

    if unbind: mylog.info('Unbinding %s' % group_types[group_type])

    for v in tqdm(groupings.values(),
                  total=len(groupings),
                  desc='Processing %s' % group_types[group_type]):
        v._process_group()

    n_invalid = 0
    group_list = []
    for v in six.itervalues(groupings):
        if not v._valid:
            n_invalid += 1
            continue
        group_list.append(v)

        
   
    mylog.info('Disregarding %d invalid %s (%d left)' % (n_invalid, group_types[group_type], len(group_list)))
        
    # sort by mass
    group_list.sort(key = lambda x: x.masses['total'], reverse=True)
    for i in range(0,len(group_list)):
        group_list[i].GroupID = i

        
        if not hasattr(group, 'unbound_indexes'):
            continue
        
        glist[group.unbound_indexes[ptype_ints['gas']]]  = -2
        slist[group.unbound_indexes[ptype_ints['star']]] = -2
        dmlist[group.unbound_indexes[ptype_ints['dm']]]  = -2
        #dmlist[group.unbound_indexes[ptype_ints['bh']]]  = -2
        bhlist[group.unbound_indexes[ptype_ints['bh']]]  = -2
        dlist[group.unbound_indexes[ptype_ints['dust']]]  = -2
            
