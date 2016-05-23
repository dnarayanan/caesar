
import caesar.vtk_vis as vtk
from caesar.utils import rotator

def group_vis(group, rotate=True):

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
    
    v.render()
