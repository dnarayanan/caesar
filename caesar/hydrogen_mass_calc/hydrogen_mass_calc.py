## SHELL FILES FOR DOCUMENTATION PURPOSES           ##
## if you want to see the source please examine     ##
## caesar/hydrogen_mass_calc/hydrogen_mass_calc.pyx ##

def check_values(obj):
    """Check to make sure that we have the required fields available to
    perform the hydrogen mass frac calculation.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Main CAESAR object.
    
    Returns
    -------
    bool
        Returns True if all fields are present, False otherwise.

    """
    pass


def hydrogen_mass_calc(obj,**kwargs):
    """Calculate the neutral and molecular mass contents of SPH particles.
    
    For non star forming gas particles assigned to halos we calculate
    the neutral fraction based on equations from Popping+09 and
    Rahmati+13.  If H2 block is not present in the simulation file we
    estimate the neutral and molecular fraciton via Leroy+08.  Once
    these fractions are calculated we assign HI/H2 masses to galaxies
    & halos based on their mass-weighted distances.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Main CAESAR object.

    Returns
    -------
    HImass, H2mass : np.ndarray, np.ndarray
        Contains the HImass and H2 mass of each individual particle.

    """
    pass


def assign_halo_gas_to_galaxies(
        ## internal to the halo
        internal_galaxy_pos,
        internal_galaxy_mass,
        internal_glist,
        internal_galaxy_index_list,
        ## global values
        galaxy_glist,
        grhoH,
        gpos,
        galaxy_HImass,
        galaxy_H2mass,
        HImass,
        H2mass,
        low_rho_thresh,
        boxsize,
        halfbox
):    
    """Function to assign halo gas to galaxies.

    When we assign galaxies in CAESAR, we only consider dense gas.
    But when considering HI gas however, it is often desirable to
    also consider low-density gas 'outside' of the galaxy.  This
    function calculates the mass weighted distance to each galaxy
    within a given halo and assigns low-density gas to the 'nearest'
    galaxy.

    Typically called from :func:`hydrogen_mass_calc.hydrogen_mass_calc`.

    """
    pass
