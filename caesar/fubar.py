from .group import create_new_group
from .property_getter import get_property, get_particles_for_FOF
from yt.units.yt_array import uconcatenate, YTArray
from yt.data_objects.octree_subset import YTPositionArray
from yt.utilities.lib.contour_finding import ParticleContourTree
import numpy as np
from yt.geometry.selection_routines import AlwaysSelector

def fof(obj, pdata, LL):
    pct = ParticleContourTree(LL)

    #pos = YTPositionArray(YTArray(pdata['pos'],'kpccm',registry=obj.yt_dataset.unit_registry))
    pos = YTPositionArray(pdata['pos'])
    ot  = pos.to_octree()

    #ot  = [c._current_chunk.objs[0] for c in obj._dd.chunks([], 'all')][0]
    #group_tags = pct.identify_contours(ot.oct_handler, ot.domain_ind, pdata['pos'],
    #                                   np.arange(0,len(pdata['pos']),dtype=np.int64),
    #                                   0,0)

    group_tags = pct.identify_contours(
        ot,
        ot.domain_ind(AlwaysSelector(None)),
        pdata['pos'],
        np.arange(0,len(pdata['pos']),dtype=np.int64),
        0,0
    )

    return group_tags

def get_ptypes(obj, find_type):
    ptypes = ['dm','gas','star']
    if find_type == 'galaxy':
        ptypes.remove('dm')
    if obj._ds_type.grid:
        ptypes.remove('gas')
    return ptypes


def get_mean_interparticle_separation(obj):
    UT = obj.yt_dataset.time_unit.to('s/h')/obj.yt_dataset.scale_factor
    UL = obj.yt_dataset.length_unit.to('cmcm/h')
    UM = obj.yt_dataset.mass_unit.to('g/h')
    GRAV = obj.yt_dataset.quan(6.672e-8, 'cmcm**3/(g*s**2)')
    
    G = GRAV / UL**3 * UM * UT**2  ## to system units
    Hubble = obj.yt_dataset.quan(3.2407789e-18, 'h/s') * UT

    dmmass = get_property(obj, 'mass', 'dm').to('code_mass')
    ndm    = len(dmmass)
    dmmass = np.sum(dmmass)

    gmass = get_property(obj, 'mass', 'gas').to('code_mass')
    smass = get_property(obj, 'mass', 'star').to('code_mass')
    bmass = np.sum(gmass) + np.sum(smass)

    Om = obj.yt_dataset.cosmology.omega_matter
    Ob = (bmass / (bmass + dmmass) * Om).d
    rhodm = ((Om - Ob) * 3.0 * Hubble**2 / (8.0 * np.pi * G)).d
    rhodm = obj.yt_dataset.quan(rhodm, 'code_mass/code_length**3')

    mips = ((dmmass / ndm / rhodm)**(1./3.)).to(obj.units['length'])
    return mips
    
def fubar(obj, find_type, **kwargs):
    ptypes = get_ptypes(obj, find_type)        
    pdata  = get_particles_for_FOF(obj, ptypes, find_type)
    LL     = get_mean_interparticle_separation(obj) * 0.2
    tags   = fof(obj, pdata, LL)

    tag_sort = np.argsort(tags)
    tags     = tags[tag_sort]
    pos      = pdata['pos'][tag_sort]
    vel      = pdata['vel'][tag_sort]
    mass     = pdata['mass'][tag_sort]
    ptype    = pdata['ptype'][tag_sort]
    indexes  = pdata['indexes'][tag_sort]

    # create unique groups
    groupings = {}
    unique_groupIDs = np.unique(tags)

    print len(unique_groupIDs)
    import sys
    sys.exit()
    
              
    for GroupID in unique_groupIDs:
        if GroupID < 0:
            continue
        groupings[GroupID] = create_new_group(obj, find_type)

    # assign particles to each group    
    for i in range(0,pdata['nparticles']):
        if tags[i] < 0:
            continue
        groupings[tags[i]].particle_indexes.append(i)

    
