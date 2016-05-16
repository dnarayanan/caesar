import numpy as np

def link_galaxies_and_halos(obj):
    """Link galaxies and halos to one another.

    This function creates the links between galaxy-->halo and
    halo-->galaxy objects.  Is run during creation, and loading in of
    each CAESAR file.

    """
    if not obj._has_galaxies:
        return

    # halos
    for halo in obj.halos:
        halo.galaxies = []
        for galaxy_index in halo.galaxy_index_list:
            halo.galaxies.append(obj.galaxies[galaxy_index])
    
    # galaxies
    for galaxy in obj.galaxies:
        if galaxy.parent_halo_index > -1:
            galaxy.halo = obj.halos[galaxy.parent_halo_index]
        else:
            galaxy.halo = None
            

def create_sublists(obj):
    """Create sublists of objects.

    Will create the sublists:
        - central_galaxies
        - satellite_galaxies
        - unassigned_galaxies (those without a halo)

    """
    if not obj._has_galaxies:
        return
    
    obj.central_galaxies   = []
    obj.satellite_galaxies = []
    
    # assign halo sub lists
    for halo in obj.halos:
        halo.satellite_galaxies = []
        for galaxy in halo.galaxies:
            if galaxy.central:
                halo.central_galaxy = galaxy
            else:
                halo.satellite_galaxies.append(galaxy)

    # assign galaxy sub lists
    for galaxy in obj.galaxies:
        if galaxy.central:
            galaxy.satellites = galaxy.halo.satellite_galaxies
            obj.central_galaxies.append(galaxy)
        elif galaxy.halo is not None:
            galaxy.satellites = []
            obj.satellite_galaxies.append(galaxy)
        else:
            if not hasattr(obj, 'unassigned_galaxies'):
                obj.unassigned_galaxies = []
            obj.unassigned_galaxies.append(galaxy)
            
