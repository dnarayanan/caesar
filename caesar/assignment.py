import numpy as np

def assign_galaxies_to_halos(obj):
    """Assign galaxies to halos.

    This function compares galaxy_glist + galaxy_slist with halo_glist
    + halo_slist to determine which halo the majority of particles
    within each galaxy lie.  Finally we assign the .galaxies list to
    each halo.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Object containing the galaxies and halos lists.    

    """
    if not obj._has_galaxies:
        return
    
    h_glist = obj.global_particle_lists.halo_glist
    h_slist = obj.global_particle_lists.halo_slist
    
    for galaxy in obj.galaxies:
        glist = h_glist[galaxy.glist]
        slist = h_slist[galaxy.slist]

        combined = np.hstack((glist,slist))
        valid = np.where(combined > -1)[0]
        combined = combined[valid]

        galaxy.parent_halo_index = -1
        if len(combined) > 0:
            galaxy.parent_halo_index = np.bincount(combined).argmax()

    for halo in obj.halos:
        halo.galaxy_index_list = []

    for i in range(0,obj.ngalaxies):
        galaxy = obj.galaxies[i]
        if galaxy.parent_halo_index > -1:
            obj.halos[galaxy.parent_halo_index].galaxy_index_list.append(i)

            
def assign_central_galaxies(obj):
    """Assign central galaxies.

    Iterate through halos and consider the most massive galaxy within
    a central and all other satellites.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Object containing the galaxies to assign centrals.  Halos 
        must already be assigned via `assign_galaxies_to_halos`.

    """
    if not obj._has_galaxies:
        return
    
    obj.central_galaxies   = []
    obj.satellite_galaxies = []

    for halo in obj.halos:
        if len(halo.galaxy_index_list) == 0:
            continue

        galaxy_masses = np.array([s.masses['total'] for s in halo.galaxies])
        central_index = np.argmax(galaxy_masses)
        obj.galaxies[halo.galaxy_index_list[central_index]].central = True
