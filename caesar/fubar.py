import numpy as np
from caesar.group import create_new_group, group_types
from caesar.property_getter import get_property, get_particles_for_FOF, get_high_density_gas_indexes
from caesar.property_getter import ptype_ints
from caesar.utils import calculate_local_densities

from yt.extern import six
from yt.funcs import mylog
from yt.extern.tqdm import tqdm
from yt.units.yt_array import uconcatenate, YTArray
from yt.data_objects.octree_subset import YTPositionArray
from yt.utilities.lib.contour_finding import ParticleContourTree
from yt.geometry.selection_routines import AlwaysSelector
#from yt.analysis_modules.halo_finding.rockstar.rockstar_groupies import RockstarGroupiesInterface

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
    if group_type is not None:
        mylog.info('Performing 3D FOF on %d positions for %s identification' %
                   (len(positions), group_type))
                                                                             
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

    """
    ## RS
    ds  = obj.yt_dataset
    rgi = RockstarGroupiesInterface(ds)
    rgi.setup_rockstar(ds.mass_unit * ds.parameters['MassTable'][1], force_res = LL / 0.2 / 50.)
    ind = np.argsort(group_tags)
    print('running rs')
    
    print(pdata.keys())
    #pcounts = rgi.make_rockstar_fof(ind, group_tags, pdata['pos'], pdata['vel'], pdata['mass'], pdata['ptype'])
    pcounts = rgi.make_rockstar_fof(ind, group_tags, pdata['pos'], pdata['vel'])
    print('pcounts:',pcounts)
    #rgi.output_halos()
    halos = rgi.return_halos()
    #import ipdb; ipdb.set_trace()
    return halos
    """
    
    return group_tags



def get_ptypes(obj, group_type):
    """Unused function."""
    ptypes = ['dm','gas','star']
    
    if 'blackholes' in obj._kwargs and obj._kwargs['blackholes']:
        ptypes.append('bh')
    
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
    
    UT = obj.yt_dataset.time_unit.to('s/h')/obj.yt_dataset.scale_factor
    UL = obj.yt_dataset.length_unit.to('cmcm/h')
    UM = obj.yt_dataset.mass_unit.to('g/h')
    GRAV = obj.yt_dataset.quan(6.672e-8, 'cmcm**3/(g*s**2)')
    
    G = GRAV / UL**3 * UM * UT**2  ## to system units
    Hubble = obj.yt_dataset.quan(3.2407789e-18, 'h/s') * UT

    dmmass = get_property(obj, 'mass', 'dm').to('code_mass')
    ndm    = len(dmmass)
    dmmass = np.sum(dmmass)

    gmass  = obj.yt_dataset.arr(np.array([0.0]), 'code_mass')
    smass  = obj.yt_dataset.arr(np.array([0.0]), 'code_mass')
    bhmass = obj.yt_dataset.arr(np.array([0.0]), 'code_mass')

    from .property_getter import has_ptype
    if has_ptype(obj, 'gas'):
        gmass = get_property(obj, 'mass', 'gas').to('code_mass')
    if has_ptype(obj, 'star'):
        smass = get_property(obj, 'mass', 'star').to('code_mass')
    if obj.data_manager.blackholes and has_ptype(obj, 'bh'):
        bhmass= get_property(obj, 'mass', 'bh').to('code_mass')        
    bmass = np.sum(gmass) + np.sum(smass) + np.sum(bhmass)

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
        Can be either 'halo' or 'galaxy'; determines what objects
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
            b *= 0.2
        
    mylog.info('Using b=%g for %s' % (b, group_types[group_type]))
    return b
    
    
def fubar(obj, group_type, **kwargs):
    """Group finding procedure.

    FUBAR stands for Friends-of-friends Unbinding after Rockstar; the
    name is no longer valid, but it stuck.  Here we perform an FOF
    operation for each grouping and create the master caesar lists.

    For halos we consider dark matter + gas + stars.  For galaxies
    however, we only consider high density gas and stars (and
    blackholes if included).

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Main caesar object.
    group_type : str
        Can be either 'halo' or 'galaxy'; determines what objects
        we find with FOF.

    """
    LL = get_mean_interparticle_separation(obj) * get_b(obj, group_type)
    
    pos = obj.data_manager.pos

    if group_type == 'galaxy':
        if not obj.simulation.baryons_present:
            return

        # here we want to perform FOF on high density gas + stars
        high_rho_indexes = get_high_density_gas_indexes(obj)
        pos  = np.concatenate((
            pos[obj.data_manager.glist][high_rho_indexes],
            pos[obj.data_manager.slist],
            pos[obj.data_manager.bhlist]
        ))        
        
    fof_tags = fof(obj, pos, LL, group_type=group_type)

    if group_type == 'galaxy':
        gtags = np.full(obj.ngas, -1, dtype=np.int64)
        gtags[high_rho_indexes] = fof_tags[0:len(high_rho_indexes)]
        fof_tags = np.concatenate((gtags,fof_tags[len(high_rho_indexes)::]))
        
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

    
    for v in tqdm(groupings.itervalues(),
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
    glist  = np.full(obj.ngas,  -1, dtype=np.int32)
    slist  = np.full(obj.nstar, -1, dtype=np.int32)
    dmlist = np.full(obj.ndm,   -1, dtype=np.int32)

    for group in group_list:
        glist[group.glist]   = group.GroupID
        slist[group.slist]   = group.GroupID
        dmlist[group.dmlist] = group.GroupID

        if not hasattr(group, 'unbound_indexes'):
            continue
        
        glist[group.unbound_indexes[ptype_ints['gas']]]  = -2
        slist[group.unbound_indexes[ptype_ints['star']]] = -2
        dmlist[group.unbound_indexes[ptype_ints['dm']]]  = -2
            
    setattr(obj.global_particle_lists, '%s_glist'  % group_type, glist)
    setattr(obj.global_particle_lists, '%s_slist'  % group_type, slist)
    setattr(obj.global_particle_lists, '%s_dmlist' % group_type, dmlist)       
    
    calculate_local_densities(obj, group_list)
    
    if group_type == 'halo':
        obj.halos  = group_list
        obj.nhalos = len(obj.halos)
    elif group_type == 'galaxy':
        obj.galaxies  = group_list
        obj.ngalaxies = len(obj.galaxies)
