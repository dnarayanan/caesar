import numpy as np
import h5py
from caesar.group import create_new_group, group_types
from caesar.property_manager import get_property, get_high_density_gas_indexes
from caesar.property_manager import ptype_ints
from caesar.utils import calculate_local_densities
from caesar.fof6d import run_fof_6d

import six
from yt.funcs import mylog
from yt.extern.tqdm import tqdm
from yt.utilities.lib.contour_finding import ParticleContourTree
from yt.geometry.selection_routines import AlwaysSelector


"""
## RS TESTING TEMP ##
class FOFGroup(object):
    def __init__(self, index):
        self.index = index
        self.halos = []

class RSHalo(object):
    def __init__(self, index, x,y,z, indexes, mass):
        self.index = index
        self.x = x
        self.y = y
        self.z = z
        self.indexes = indexes
        self.mass = mass
#####################
"""

def fof(obj, positions, LL, group_type=None):
    """Friends of friends.

    Perform 3D friends of friends via yt's ParticleContourTree method.
    
    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Object containing the yt_dataset parameter.
    positions : np.ndarray
        Nx3 position array of the particles to perform the FOF on.
    LL : float
        Linking length for the FOF procedure.

    Returns
    -------
    group_tags : np.ndarray
        Returns an integer array containing the GroupID that each 
        particle belongs to.  GroupIDs of -1 mean the particle is 
        *not* grouped.

    """
    ## TEMP RS WORK ##
    # only use DM for now
    #positions = positions[obj.data_manager.dmlist]
    ##################
    
    #if group_type is not None:
    #    mylog.info('Performing 3D FOF on %d positions for %s identification' %
    #               (len(positions), group_type))
                                                                             
    pct = ParticleContourTree(LL)

    pos = YTPositionArray(obj.yt_dataset.arr(positions, obj.units['length']))
    #pos = YTPositionArray(pdata['pos'])
    ot  = pos.to_octree()

    #ot  = [c._current_chunk.objs[0] for c in obj._dd.chunks([], 'all')][0]
    #group_tags = pct.identify_contours(ot.oct_handler, ot.domain_ind, pdata['pos'],
    #                                   np.arange(0,len(pdata['pos']),dtype=np.int64),
    #                                   0,0)

    group_tags = pct.identify_contours(
        ot,
        ot.domain_ind(AlwaysSelector(None)),
        positions,
        np.arange(0,len(positions),dtype=np.int64),
        0,0
    )    
    
    return group_tags

    """  (PAY NO ATTENTION TO THE MAN BEHIND THE CURTAIN)
    ## RS
    # example script: http://paste.yt-project.org/show/edxCFtSMU4EC1funxpvl/
    from yt.analysis_modules.halo_finding.rockstar.rockstar_groupies import RockstarGroupiesInterface

    # RS setup
    ind       = np.argsort(group_tags)
    ds        = obj.yt_dataset
    rgi       = RockstarGroupiesInterface(ds)
    force_res = obj.simulation.boxsize / float(obj.simulation.effective_resolution) / 50.
    rgi.setup_rockstar(ds.mass_unit * ds.parameters['MassTable'][1], force_res=force_res.d, min_halo_size=20)
    print('running rs')

    ## REGULAR ROCKSTAR ##
    velocities = obj.data_manager.vel[obj.data_manager.dmlist]    
    pcounts = rgi.make_rockstar_fof(ind, group_tags, positions, velocities)
    #print('pcounts:',pcounts)
    
    ## ROCKSTAR GALAXIES ##
    #velocities = obj.data_manager.vel
    #masses = obj.data_manager.mass
    #ptypes = obj.data_manager.ptype
    #pcounts = rgi.make_rockstar_fof(ind, group_tags, positions, velocities, masses, ptypes)

    rs_halos = rgi.return_halos()

    np.savez('rs_data.npz', halos=rs_halos, group_tags=group_tags)
    
    import sys
    sys.exit()
    
    ## common
    rs_halos   = rgi.return_halos()
    nhalos     = len(rs_halos['num_p'])
    fof_groups = []
    tag_index  = -1
    grp_index  = -1
    unique_group_tags = np.unique(group_tags)[1::] # remove -1

    for i in range(0,nhalos):
        halo = rs_halos[i]
        #if halo['num_p'] == 0 or halo['num_p'] < 32:
        #    continue

        # new FOF group
        if halo['p_start'] == 0:            
            ## append the previous group to the list before starting a new one
            if tag_index > -1: fof_groups.append(fof_group)
            tag_index += 1
            grp_index += 1
            g_tag      = unique_group_tags[tag_index]
            fof_group  = FOFGroup(grp_index)
        global_indexes = np.where(group_tags == g_tag)[0]

        # bullshit
        local_indexes  = global_indexes[halo['p_start']:halo['p_start']+halo['num_p']]

        if len(global_indexes) < halo['num_p']:
            print len(global_indexes), halo['num_p']
        
        mymass = np.sum(masses[halo['p_start']:halo['p_start']+halo['num_p']])
        if mymass == 0 and halo['num_p'] != 0:
            #print global_indexes
            print mymass, len(global_indexes), halo['p_start'],halo['num_p']

        #if halo['num_p'] > halo['num_child_particles']:   ## DOES NOT HAPPEN
        #print len(global_indexes),halo['num_p'],halo['num_child_particles'],halo['p_start']
        #if halo['num_p'] == halo['num_child_particles']:
        #    print halo['flags']
        #    tally += 1
            
        
        new_halo = RSHalo(grp_index, halo['pos_x'], halo['pos_y'], halo['pos_z'], local_indexes, np.sum(masses[local_indexes]))
        fof_group.halos.append(new_halo)

    parents, children = [],[]
    for group in fof_groups:
        group_masses   = [i.mass for i in group.halos]
        max_mass_index = np.argmax(group_masses)
        for i in range(0,len(group.halos)):
            if i == max_mass_index:
                parents.append(group.halos[i])
            else:
                children.append(group.halos[i])

    
        
    parent_halos = []
    child_halos = []
    
    nhalos = len(halos['num_p'])
    gti = -1
    fof_groups = []
    fof_group = FOFGroup(gti)
    for i in range(0,nhalos):
        this_halo = halos[i]
        if this_halo['num_p'] == 0 or this_halo['num_p'] < 32:
            continue
        
        if this_halo['p_start'] == 0:
            if gti > 0:
                fof_groups.append(fof_group)
            gti += 1
            fof_group = FOFGroup(gti)
        group_tag = group_tags[gti]
        global_indexes = np.where(group_tags == group_tag)[0]
        local_indexes  = global_indexes[this_halo['p_start']:this_halo['p_start']+this_halo['num_p']]

        h = RSHalo(group_tag,this_halo['pos_x'],this_halo['pos_y'],this_halo['pos_z'],local_indexes, np.sum(masses[local_indexes]))
        #h = RSHalo(gti,this_halo['pos_x'],this_halo['pos_y'],this_halo['pos_z'],pos,this_halo['num_p'])

        fof_group.halos.append(h)


    for grp in fof_groups:
        grp_masses = [i.mass for i in grp.halos]

        #num_p_halo = [i.num_p for i in grp.halos]        
        #maxnp = np.argmax(num_p_halo)
        maxnp = np.argmax(grp_masses)
        
        for i in range(0,len(grp.halos)):
            h = grp.halos[i]
            if i == maxnp:
                parent_halos.append(h)
            else:
                child_halos.append(h)

    parent_group_tags = np.full(len(positions),-1,dtype=np.int64)
    child_group_tags  = np.full(len(positions),-1,dtype=np.int64)
                
    import caesar.vtk_vis as vtk
    v = vtk.vtk_render()
    
    v.point_render(pos, color=[1,0,0],alpha=0.5)
    
    xpos = [i.x for i in parent_halos]
    ypos = [i.y for i in parent_halos]
    zpos = [i.z for i in parent_halos]
    hpos = np.column_stack((xpos,ypos,zpos))

    v.point_render(hpos, color=[0,0,1],psize=5)

    xpos = [i.x for i in child_halos]
    ypos = [i.y for i in child_halos]
    zpos = [i.z for i in child_halos]
    hpos = np.column_stack((xpos,ypos,zpos))

    v.point_render(hpos, color=[0,1,1],psize=3)
    
    #for i in range(0,len(parents)):
    #    v.place_label(hpos[parents][i], '%d %d' % (num_p[parents][i], halos['num_child_particles'][i]))

    largest = np.argmax(halos['num_p'])
    halo_pos = np.column_stack((halos['pos_x'],halos['pos_y'],halos['pos_z']))
    
    v.render(focal_point=halo_pos[largest])
    
    return halos
    """



def get_ptypes(obj, group_type):
    """Unused function."""
    ptypes = ['dm','gas','star']
    
    if 'blackholes' in obj._kwargs and obj._kwargs['blackholes']:
        ptypes.append('bh')

    if 'dust' in obj._kwargs and obj._kwargs['dust']:
        ptypes.append('dust')
    
    if group_type == 'galaxy':
        ptypes.remove('dm')
    if obj._ds_type.grid:
        ptypes.remove('gas')
    return ptypes


def get_mean_interparticle_separation(obj):
    """Calculate the mean interparticle separation and Omega Baryon.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Main caesar object.

    Returns
    -------
    mips : float
        Mean inter-particle separation used for calculating FOF's b
        parameter.

    """    
    if hasattr(obj.simulation, 'mean_interparticle_separation'):
        return obj.simulation.mean_interparticle_separation

    if obj.yt_dataset.cosmological_simulation == 0:
        mylog.info('Non-cosmological data set detected -- setting units as specified in fubar.py')
        UT = obj.yt_dataset.current_time.to('s/h')
        UL = obj.yt_dataset.length_unit.to('cm/h')
        GRAV = obj.yt_dataset.quan(6.672e-8, 'cm**3/(g*s**2)')
    else:
        UT = obj.yt_dataset.time_unit.to('s/h')/obj.yt_dataset.scale_factor
        UL = obj.yt_dataset.length_unit.to('cmcm/h')
        GRAV = obj.yt_dataset.quan(6.672e-8, 'cmcm**3/(g*s**2)')

    UM = obj.yt_dataset.mass_unit.to('g/h')

    
    G = GRAV / UL**3 * UM * UT**2  ## to system units
    Hubble = obj.yt_dataset.quan(3.2407789e-18, 'h/s') * UT

    dmmass = get_property(obj, 'mass', 'dm').to('code_mass')
    ndm    = len(dmmass)
    dmmass = np.sum(dmmass)

    gmass  = obj.yt_dataset.arr(np.array([0.0]), 'code_mass')
    smass  = obj.yt_dataset.arr(np.array([0.0]), 'code_mass')
    bhmass = obj.yt_dataset.arr(np.array([0.0]), 'code_mass')
    dustmass = obj.yt_dataset.arr(np.array([0.0]), 'code_mass')
    
    from caesar.property_manager import has_ptype
    if has_ptype(obj, 'gas'):
        gmass = get_property(obj, 'mass', 'gas').to('code_mass')
    if has_ptype(obj, 'star'):
        smass = get_property(obj, 'mass', 'star').to('code_mass')
    if obj.data_manager.blackholes and has_ptype(obj, 'bh'):
        bhmass= get_property(obj, 'mass', 'bh').to('code_mass')        
    if obj.data_manager.dust and has_ptype(obj, 'dust'):
        dustmass= get_property(obj, 'mass', 'dust').to('code_mass')        
    bmass = np.sum(gmass) + np.sum(smass) + np.sum(bhmass) + np.sum(dustmass)


    """
    DM = obj.data_manager
    dmmass = DM.mass[DM.dmlist]
    gmass  = DM.mass[DM.glist]
    smass  = DM.mass[DM.slist]
    bhmass = DM.mass[DM.bhlist]

    ndm = len(dmmass)
    
    dmmass = obj.yt_dataset.quan(np.sum(dmmass), obj.units['mass']).to('code_mass')
    bmass  = obj.yt_dataset.quan(np.sum(gmass) + np.sum(smass) + np.sum(bhmass), obj.units['mass']).to('code_mass')
    """
    #if its an idealized simulation, then there's no cosmology and we just take z=0 Planck15 values
    if obj.yt_dataset.cosmological_simulation == 0:
        from astropy.cosmology import Planck15
        Om = Planck15.Om0
        Ob = Planck15.Ob0
    else:
        Om = obj.yt_dataset.cosmology.omega_matter
        Ob = (bmass / (bmass + dmmass) * Om).d
    
    rhodm = ((Om - Ob) * 3.0 * Hubble**2 / (8.0 * np.pi * G)).d
    rhodm = obj.yt_dataset.quan(rhodm, 'code_mass/code_length**3')
    
    mips  = ((dmmass / ndm / rhodm)**(1./3.)).to(obj.units['length'])
    efres = int(obj.simulation.boxsize.d / mips.d)
    
    obj.simulation.omega_baryon = float(Ob)
    obj.simulation.effective_resolution = efres
    obj.simulation.mean_interparticle_separation = mips

    mylog.info('Calculated Omega_Baryon=%g and %d^3 effective resolution' % (Ob, efres))
    
    return obj.simulation.mean_interparticle_separation


def get_b(obj, group_type):
    """Function to return *b*, the fraction of the mean interparticle
    separation.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Main caesar object.
    group_type : str
        Can be either 'halo' or 'galaxy' or 'cloud'; determines what objects
        we find with FOF.

    Returns
    -------
    b : float
        Fraction of the mean interparticle separation used for FOF 
        linking length.

    """
    b = 0.2

    if 'b_halo' in obj._kwargs and isinstance(obj._kwargs['b_halo'], (int, float)):
        b = float(obj._kwargs['b_halo'])

    if group_type == 'galaxy':
        if 'b_galaxy' in obj._kwargs and isinstance(obj._kwargs['b_galaxy'], (int, float)):
            b = float(obj._kwargs['b_galaxy'])
        else:
            b *= 0.1  

    if group_type == 'cloud':
        if 'b_cloud' in obj._kwargs and isinstance(obj._kwargs['b_cloud'], (int, float)):
            b = float(obj._kwargs['b_cloud'])
        else:
            b *= 0.1 #BOBBY CONVERSATION: SET UP A CONFIG FILE THAT HAS A DEFAULT SET OF PARAMETERS -- WHETHER WE WANT TO RUN HALOS, GALAXIES ETC.  AND THEN HAVE THE CODE AUTMOATICALLY TURN ON CLOUDS IF WE SET -B_CLOUD linking lengths
            
    mylog.info('Using b=%g for %s' % (b, group_types[group_type]))
    return b
    
    
def fubar(obj, group_type, **kwargs):
    """Group finding procedure.

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
    pos = obj.data_manager.pos

    unbind = False        
    unbind_str = 'unbind_%s' % group_types[group_type]
    if unbind_str in obj._kwargs and \
        isinstance(obj._kwargs[unbind_str], bool):
        unbind = obj._kwargs[unbind_str]
    setattr(obj.simulation, unbind_str, unbind)

    if group_type == 'galaxy':
        if not obj.simulation.baryons_present:
            return


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
        #print 'fof_tags',len(fof_tags[fof_tags>=0]),max(fof_tags),np.shape(fof_tags),fof_tags[fof_tags>=0]

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

        
    # initialize global lists
    glist  = np.full(obj.simulation.ngas,  -1, dtype=np.int32)
    slist  = np.full(obj.simulation.nstar, -1, dtype=np.int32)
    dmlist = np.full(obj.simulation.ndm,   -1, dtype=np.int32)
    bhlist = np.full(obj.simulation.nbh,   -1, dtype=np.int32)
    dlist  = np.full(obj.simulation.ndust,  -1, dtype=np.int32)
    
    for group in group_list:
        glist[group.glist]   = group.GroupID
        slist[group.slist]   = group.GroupID
        dmlist[group.dmlist] = group.GroupID
        bhlist[group.bhlist] = group.GroupID
        dlist[group.dlist]   = group.GroupID
        
        if not hasattr(group, 'unbound_indexes'):
            continue
        
        glist[group.unbound_indexes[ptype_ints['gas']]]  = -2
        slist[group.unbound_indexes[ptype_ints['star']]] = -2
        dmlist[group.unbound_indexes[ptype_ints['dm']]]  = -2
        #dmlist[group.unbound_indexes[ptype_ints['bh']]]  = -2
        bhlist[group.unbound_indexes[ptype_ints['bh']]]  = -2
        dlist[group.unbound_indexes[ptype_ints['dust']]]  = -2
            
    setattr(obj.global_particle_lists, '%s_glist'  % group_type, glist)
    setattr(obj.global_particle_lists, '%s_slist'  % group_type, slist)
    setattr(obj.global_particle_lists, '%s_dmlist' % group_type, dmlist)
    setattr(obj.global_particle_lists, '%s_bhlist' % group_type, bhlist)
    setattr(obj.global_particle_lists, '%s_dlist'  % group_type, dlist)
    
    calculate_local_densities(obj, group_list)
    
    if group_type == 'halo':
        obj.halos  = group_list
        obj.nhalos = len(obj.halos)
        #for ig in range(obj.nhalos):
        #    if ig < 5: print('%d: dm %g gas %g star %g bh %g [%g %g %g] r200 %g vc %g sig %g %g %g'%(ig,np.log10(obj.halos[ig].masses['dm']),np.log10(obj.halos[ig].masses['gas']),np.log10(obj.halos[ig].masses['stellar']),np.log10(obj.halos[ig].masses['bh']),obj.halos[ig].pos[0],obj.halos[ig].pos[1],obj.halos[ig].pos[2],obj.halos[ig].radii['r500c'],obj.halos[ig].virial_quantities['circular_velocity'],obj.halos[ig].velocity_dispersions['gas'],obj.halos[ig].velocity_dispersions['stellar'],obj.halos[ig].velocity_dispersions['dm']))
    elif group_type == 'galaxy':
        obj.galaxies  = group_list
        obj.ngalaxies = len(obj.galaxies)
    if group_type == 'cloud':
        obj.clouds  = group_list
        obj.nclouds = len(obj.clouds)
