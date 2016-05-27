
import caesar.vtk_vis as vtk
from caesar.utils import rotator

def group_vis(group, rotate=True):
    """Function to visualize a :class:`group.Group` with VTK.

    Parameters
    ----------
    group : :class:`group.Group`
        Group to visualize.
    rotate : boolean
        If true the positions are rotated so that the angular momentum 
        vector is aligned with the z-axis.

    """
    DM = group.obj.data_manager    
    v  = vtk.vtk_render()

    if group.ngas > 0:
        pos = DM.pos[DM.glist[group.glist]]
        if rotate:
            pos = rotator(pos, group.rotation_angles['ALPHA'],
                          group.rotation_angles['BETA'])
        v.point_render(pos, color=[0,0,1])
        
    if group.nstar > 0:
        pos = DM.pos[DM.slist[group.slist]]
        if rotate:
            pos = rotator(pos, group.rotation_angles['ALPHA'],
                          group.rotation_angles['BETA'])        
        v.point_render(pos, color=[1,1,0])

    if hasattr(group, 'ndm') and group.ndm > 0:
        pos = DM.pos[DM.dmlist[group.dmlist]]
        if rotate:
            pos = rotator(pos, group.rotation_angles['ALPHA'],
                          group.rotation_angles['BETA'])        
        v.point_render(pos, color=[1,0,0])
    
    v.render(focal_point=group.pos.d)


def sim_vis(obj, ptypes = ['dm','star','gas'],
            halo_only=True, galaxy_only=False,
            annotate_halos=False, annotate_galaxies=False):
    """Function to visualize an entire simulation with VTK.
    
    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Simulation object to visualize.
    ptypes : list
        List containing one or more of the following: 
        'dm','gas','star', which dictates which particles to render.
    halo_only : boolean
        If True only render particles belonging to halos.
    galaxy_only: boolean
        If True only render particles belonging to galaxies.  Note 
        that this overwrites ``halo_only``.
    annotate_halos : boolean, list, int, optional
        Add labels to the render at the location of halos annotating
        the group ID and total mass.  If True then all halos are
        annotated, if an integer list then halos of those indexes
        are annotated, and finally if an integer than the most massive
        N halos are annotated.
    annotate_galaxies : boolean, list, int, optional
        Add labels to the render at the location of galaxies 
        annotating the group ID and total mass.  If True then all
        galaxies are annotated, if an integer list then galaxies of 
        those indexes are annotated, and finally if an integer than 
        the most massive N galaxies are annotated.

    """    
    import numpy as np
    
    if galaxy_only:
        halo_only = False
        if 'dm' in ptypes:
            ptypes.remove('dm')

    gpl = obj.global_particle_lists            
    DM  = obj.data_manager
    v   = vtk.vtk_render()

    if obj.simulation.ngas > 0 and 'gas' in ptypes:
        pos = DM.pos[DM.glist]
        if halo_only:
            pos = pos[np.where(gpl.halo_glist > -1)[0]]
        elif galaxy_only:
            pos = pos[np.where(gpl.galaxy_glist > -1)[0]]
        v.point_render(pos, color=[0,0,1])
        
    if obj.simulation.nstar > 0 and 'star' in ptypes:
        pos = DM.pos[DM.slist]
        if halo_only:
            pos = pos[np.where(gpl.halo_slist > -1)[0]]
        elif galaxy_only:
            pos = pos[np.where(gpl.galaxy_slist > -1)[0]]
        v.point_render(pos, color=[1,1,0])

    if obj.simulation.ndm > 0 and 'dm' in ptypes:
        pos = DM.pos[DM.dmlist]
        if halo_only:
            pos = pos[np.where(gpl.halo_dmlist > -1)[0]]
        v.point_render(pos, color=[1,0,0])

    def annotate_group(group_list, annotation, txtcolor):
        if isinstance(annotation, bool):
            indexes = [i for i in range(0,len(group_list))]
        elif isinstance(annotation, list):
            indexes = annotation
        elif isinstance(annotation, int):
            indexes = [i for i in range(0, annotate_halos)]

        for i in indexes:
            if i >= len(group_list): continue
            group = group_list[i]
            v.place_label(group.pos, '%d :: %0.2e Msun' %
                          (group.GroupID, group.masses['total']),
                          text_color=txtcolor)

        
    if annotate_halos and 'dm' in ptypes:
        annotate_group(obj.halos, annotate_halos, [0.75,0.75,0])
    if annotate_galaxies:
        annotate_group(obj.galaxies, annotate_galaxies, [0, 0.75, 0.75])
        
    v.render()
