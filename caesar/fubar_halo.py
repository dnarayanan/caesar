import numpy as np
import h5py
from caesar.group import group_types
from caesar.property_manager import get_property, get_particles_for_FOF, get_high_density_gas_indexes
from caesar.property_manager import ptype_ints
from caesar.utils import calculate_local_densities
from caesar.fof6d import run_fof_6d

import six
from yt.funcs import mylog
from yt.units.yt_array import uconcatenate, YTArray
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

    # set up number of processors
    obj.nproc = 1  # defaults to single core
    if 'nproc' in obj._kwargs:
        obj.nproc = int(obj._kwargs['nproc'])
    if obj.nproc != 1:
        import joblib
        if obj.nproc < 0:
            obj.nproc += joblib.cpu_count()+1
        if obj.nproc == 0:
            obj.nproc = joblib.cpu_count()
    mylog.info('member_search() running on %d cores'%obj.nproc)
    obj.load_haloid = False
    if 'haloid' in obj._kwargs and 'snap' in obj._kwargs['haloid']:
        obj.load_haloid = True

    # Process halos
    halos = fof6d(obj,'halo')  #instantiate a fof6d object
    halos.MIS = get_mean_interparticle_separation(obj).d # also computes omega_baryon and related quantities
    halos.load_haloid()
    halos.obj.data_manager._member_search_init(select=halos.haloid)  # load particle info, but only those selected to be in a halo
    if not halos.plist_init():  # not enough halo particles found, nothing to do!
        return  
    halos.load_lists()  # create halos, load particle indexes for halos
    if len(halos.obj.halo_list) == 0:  # no valid halos found
        mylog.warning('No valid halos found! Aborting member search')
        return  
    get_group_properties(halos,halos.obj.halo_list)  # compute halo properties

    # Find galaxies, or load galaxy membership info
    if not obj.simulation.baryons_present:  # if no baryons, we're done
        return
    fof6d_flag = True
    if 'fof6d' in obj._kwargs and not obj._kwargs['fof6d']:  # fof6d for galaxies/clouds not requested
        mylog.warning('fof6d not requested, and no other galaxy finder available! Aborting member search')
        return 
    if 'fof6d_file' in obj._kwargs and obj._kwargs['fof6d_file'] is not None:
        fof6d_flag = halos.load_fof6dfile()  # load galaxy ID's from fof6d_file
    if fof6d_flag:
        halos.run_fof6d('galaxy')  # run fof6d on halos to find galaxies
        halos.save_fof6dfile()  # save fof6d info
        #snapname = ('%s/%s'%(obj.simulation.fullpath,obj.simulation.basename))
        #nparts,gas_index,star_index,bh_index = run_fof_6d(snapname,16,0.02,1.0,nproc)

    # Process galaxies
    galaxies = fof6d(obj,'galaxy')  #instantiate a fof6d object
    galaxies.plist_init(parent=halos)  # get particle list for computing galaxy properties
    if galaxies.nparttot == 0: # plist_init didn't find any particles in a galaxy
        mylog.warning('Not enough eligible galaxy particles found!')
        return  
    galaxies.load_lists(parent=halos)  # create galaxy_list, load particle index lists for galaxies
    get_group_properties(galaxies,galaxies.obj.galaxy_list)  # compute galaxy properties
    if ('fsps_bands' in obj._kwargs) and obj._kwargs['fsps_bands'] is not None:
        from caesar.pyloser.pyloser import photometry
        galphot = photometry(obj,galaxies.obj.galaxy_list)
        galphot.run_pyloser()

    # Find and process clouds
    if ('fofclouds' in obj._kwargs) and obj._kwargs['fofclouds']:
        galaxies.run_fof6d('cloud')  # run fof6d to find cloudid's
        galaxies.load_lists('cloud')  # load particle index lists for galaxies
        clouds = fof6d(obj,'cloud')  #instantiate a fof6d object
        clouds.plist_init(parent=galaxies)  # initialize comptutation of cloud properties
        if clouds.nparttot == 0: return  # plist_init didn't find enough particles to group
        galaxies.load_lists('cloud')
        get_group_properties(clouds,clouds.obj.cloud_list)  # compute cloud properties
   
    # reset particle lists to have original snapshot ID's; must do this after all group processing is finished
    reset_global_particle_IDs(obj)
    # load global lists
    load_global_lists(obj)

    return


plist_dict = dict( gas='glist', star='slist', bh='bhlist', dust='dlist', dm='dmlist', dm2='dm2list', dm3='dm3list')

def reset_global_particle_IDs(obj):
    ''' Maps particle lists from currently loaded ID's to the ID's corresponding to the full snapshot '''

    # determine full particle numbers in snapshot
    from caesar.property_manager import has_ptype, get_property
    offset = np.zeros(len(obj.data_manager.ptypes)+1,dtype=np.int64)
    for ip,p in enumerate(obj.data_manager.ptypes):
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
        elif p == 'dm3':
            offset[ip+1] = offset[ip] + obj.simulation.ndm3
            obj.simulation.ndm3 = count

    # reset lists
    for group_type in obj.group_types:
        group_list = 'obj.%s_list'%group_type
        for ip,p in enumerate(obj.data_manager.ptypes):
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
                if p == 'dm3': group.dm3list = mylist

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
        if 'dm2' in obj.data_manager.ptypes: dm2list = np.full(obj.simulation.ndm2, -1, dtype=np.int32)
        if 'dm3' in obj.data_manager.ptypes: dm3list = np.full(obj.simulation.ndm3, -1, dtype=np.int32)

        group_list = 'obj.%s_list'%group_type
        for group in eval(group_list):
            for ip,p in enumerate(obj.data_manager.ptypes):
                if not has_ptype(obj, p):
                    continue
                part_list = 'group.%s'%plist_dict[p]
                if p == 'gas': glist[eval(part_list)] = group.GroupID
                if p == 'star': slist[eval(part_list)] = group.GroupID
                if p == 'bh': bhlist[eval(part_list)] = group.GroupID
                if p == 'dust': dlist[eval(part_list)] = group.GroupID
                if p == 'dm': dmlist[eval(part_list)] = group.GroupID
                if p == 'dm2': dm2list[eval(part_list)] = group.GroupID
                if p == 'dm3': dm3list[eval(part_list)] = group.GroupID

        setattr(obj.global_particle_lists, '%s_glist'  % group_type, glist)
        setattr(obj.global_particle_lists, '%s_slist'  % group_type, slist)
        setattr(obj.global_particle_lists, '%s_bhlist' % group_type, bhlist)
        setattr(obj.global_particle_lists, '%s_dlist'  % group_type, dlist)
        setattr(obj.global_particle_lists, '%s_dmlist' % group_type, dmlist)
        if 'dm2' in obj.data_manager.ptypes: setattr(obj.global_particle_lists, '%s_dm2list'% group_type, dm2list)
        if 'dm3' in obj.data_manager.ptypes: setattr(obj.global_particle_lists, '%s_dm3list'% group_type, dm3list)

    return
