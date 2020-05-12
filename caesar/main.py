import numpy as np

from caesar.property_manager import DatasetType
from caesar.particle_list import ParticleListContainer
from caesar.simulation_attributes import SimulationAttributes

import six
from yt.funcs import mylog, get_hash


class CAESAR(object):
    """Master CAESAR class.

    CAESAR objects contain all references to halos, galaxies, and
    clouds for a single snapshot.  Its output format is portable and
    global object statistics can be examined without the raw
    simulation file.
    
    Parameters
    ----------
    ds : yt dataset, optional
        A dataset via ``ds = yt.load(snapshot)``
    mass : str, optional
        Mass unit to store data with. Defaults to 'Msun'.
    length : str, optional
        Length unit to store data with. Defaults to 'kpccm'.
    velocity : str, optional
        Velocity unit to store data with. Defaults to 'km/s'.
    time : str, optional
        Time unit to store data with. Defaults to 'yr'.
    temperature : str, optional
        Temperature unit to store data with. Defaults to 'K'.

    Examples
    --------
    >>> import caesar
    >>> obj = caesar.CAESAR()

    """
    def __init__(self, ds=0, *args, **kwargs):        
        self._args   = args
        self._kwargs = kwargs
        self._ds     = 0
        self._dm     = 0

        self.units = dict(
            mass='Msun',
            length='kpccm',
            velocity='km/s',
            time='yr',
            temperature='K'
        )

        # check for unit overrides
        for k,v in six.iteritems(kwargs):
            if k.lower() in self.units:
                self.units[k.lower()] = v

        self.global_particle_lists = ParticleListContainer(self)
        self.simulation = SimulationAttributes()
        self.yt_dataset = ds

        self.nhalos    = 0
        self.ngalaxies = 0
        self.nclouds = 0
        self.halos = []
        self.galaxies = []
        self.group_types = []
        
        self.reset_default_returns()
        
    @property
    def yt_dataset(self):
        """The yt dataset to perform actions on."""
        if isinstance(self._ds, int):
            raise Exception('No yt_dataset assigned!\nPlease assign '\
                            'one via `obj.yt_dataset=<YT DATASET>` ' \
                            'to load particle/field data from disk.')
        return self._ds
    @yt_dataset.setter
    def yt_dataset(self,value):
        if value == 0: return

        if not hasattr(value, 'dataset_type'):
            raise IOError('not a yt dataset?')

        infile = '%s/%s' % (value.fullpath, value.basename)
        

        if hasattr(self, 'hash'):
            if isinstance(self.hash, np.bytes_):
                self.hash = self.hash.decode('utf8')

            hash = get_hash(infile)
            if hash != self.hash:
                raise IOError('hash mismatch!')
            else:
                self._ds = value
        else:
            self._ds  = value
            self.hash = get_hash(infile)

        self._ds = value
        self._ds_type = DatasetType(self._ds)
        self._assign_simulation_attributes()
        
    @property
    def _has_galaxies(self):
        """Checks if any galaxies are present."""
        if self.ngalaxies > 0:
            return True
        else:
            return False

    @property
    def _has_clouds(self):
        """Checks if any clouds are present."""
        if self.nclouds > 0:
            return True
        else:
            return False
        
    @property
    def data_manager(self):
        """On demand DataManager class."""
        if isinstance(self._dm, int):
            from caesar.data_manager import DataManager
            self._dm = DataManager(self)
        return self._dm
        
    def _assign_simulation_attributes(self):
        """Populate the `caesar.simulation_attributes.SimulationAttributes`
        class."""
        self.simulation.create_attributes(self)


    def reset_default_returns(self, group_type='all'):
        """Reset the default returns for object dictionaries.
    
        This function resets the default return quantities for CAESAR 
        halo/galaxy/cloud objects including ``mass``, ``radius``, ``sigma``, 
        ``metallicity``, and ``temperature``.
    
        Parameters
        ----------
        obj : :class:`main.CAESAR`
            Main CAESAR object.
        group_type : {'all', 'halo', 'galaxy', 'cloud'}, optional
            Group to reset return values for.

        """
        self._default_returns = {}
        dr = dict(
            mass = 'total',
            radius = 'total',
            metallicity = 'mass_weighted',
            temperature = 'mass_weighted',
        )        
        if group_type == 'halo' or group_type == 'all':
            dr['sigma'] = 'dm'
            self._default_returns['halo'] = dr
        if group_type == 'galaxy' or group_type == 'all':
            dr['sigma'] = 'stellar'
            self._default_returns['galaxy'] = dr
        if group_type == 'cloud' or group_type == 'all':
            dr['sigma'] = 'gas'
            self._default_returns['galaxy'] = dr 
            
    def _set_default_returns(self, group_type, category, value):
        """Generic default return setter."""
        from caesar.group import category_mapper, group_types
        if group_type == 'halo':     group = self.halos[0]
        elif group_type == 'galaxy': group = self.galaxies[0]
        elif group_type == 'cloud': group = self.clouds[0]
        
        if category not in category_mapper.keys():
            raise ValueError('%s not a valid category!  Must pick one of %s' %
                             (category, category_mapper.keys()))
        if value not in getattr(group, category_mapper[category]):
            raise ValueError('%s not a valid value!  Must pick one of %s' %
                             (value, getattr(group, category_mapper[category]).keys()))
        
        mylog.warning('Setting default %s return for %s to "%s"' %
                      (category, group_types[group_type], value))
        self._default_returns[group_type][category] = value                
        
    def set_default_halo_returns(self, category, value):
        """Set the default return quantity for a given halo attribute.
        
        Parameters
        ----------
        category : str
            The attribute to redirect to a different quantity.
        value : str
            The internal name of the new quantity which must be 
            present in the dictinoary
        
        """
        self._set_default_returns('halo', category, value)
        
    def set_default_galaxy_returns(self, category, value):
        """Set the default return quantity for a given galaxy
        attribute.
        
        Parameters
        ----------
        category : str
            The attribute to redirect to a different quantity.
        value : str
            The internal name of the new quantity which must be 
            present in the dictinoary
    
        """
        self._set_default_returns('galaxy', category, value)


    def set_default_cloud_returns(self, category, value):
        """Set the default return quantity for a given cloud
        attribute.
        
        Parameters
        ----------
        category : str
            The attribute to redirect to a different quantity.
        value : str
            The internal name of the new quantity which must be 
            present in the dictinoary
    
        """
        self._set_default_returns('cloud', category, value)

        
    def _assign_objects(self):
        """Assign galaxies to halos, and central galaxies."""
        import caesar.assignment as assign
        assign.assign_galaxies_to_halos(self)
        assign.assign_central_galaxies(self)
        assign.assign_clouds_to_galaxies(self)
        
    def _link_objects(self):
        """Link galaxies to halos and create sublists."""
        import caesar.linking as link
        link.link_galaxies_and_halos(self)
        link.link_clouds_and_galaxies(self)
        link.create_sublists(self)


    def save(self, filename):
        """Save CAESAR file.

        Parameters
        ----------
        filename : str
            The name of the output file.

        Examples
        --------
        >>> obj.save('output.hdf5')

        """        
        from caesar.saver import save
        save(self, filename)
    
    def member_search(self, *args, **kwargs):
        """Meat and potatoes of CAESAR.

        This method is responsible for loading particle/field data
        from disk, creating halos, galaxies and clouds, linking objects
        together, and finally calculating HI/H2 masses if necessary.

        Parameters 
        ----------
        unbind_halos : boolean, optional
            Unbind halos?  Defaults to False
        unbind_galaxies : boolean, optional
            Unbind galaxies?  Defaults to False
        b_halo : float, optional
            Quantity used in the linking length (LL) for halos.
            LL = mean_interparticle_separation * b_halo.  Defaults to 
            ``b_halo = 0.2``.
        b_galaxy : float, optional
            Quantity used in the linking length (LL) for galaxies.
            LL = mean_interparticle_separation * b_galaxy.  Defaults 
            to ``b_galaxy = b_halo * 0.2``.
        ll_cloud : float, optional
            Quantity used in the linking length (LL) for clouds in 
            comoving kpc (kpccm).
        fofclouds: boolean,optional
            Indicates if we're running 3D fof on clouds. Default is that this
            is set to false
        fof6d: boolean, optional
            Indicates if we're running galaxy finding with 6D FOF vs 
            the default of 3D FOF
        fof6d_LL_factor: float, optional        
            Sets linking length for fof6d
        fof6d_mingrp: float, optional
            Sets minimum group size for fof6d
        fof6d_velLL: float, optional
            Sets linking length for velocity in fof6d
        nproc: int, optional
            Sets number of processors for fof6d and progen_rad
        blackholes : boolean, optional
            Indicate if blackholes are present in your simulation.  
            This must be toggled on manually as there is no clear 
            cut way to determine if PartType5 is a low-res particle, 
            or a black hole.
        dust : boolean, optional
            Indicate if active dust particles are present in your simulation.  
            This must be toggled on manually as there is no clear 
            cut way to determine if PartType3 is a low-res particle, 
            or an active dust particle.
        lowres : list, optional
            If you are running ``CAESAR`` on a Gadget/GIZMO zoom
            simulation in HDF5 format, you may want to check
            each halo for low-resolution contamination.  By passing
            in a list of particle types (ex. [2,3,5]) we will check
            ALL objects for contamination and add the 
            ``contamination`` attribute to all objects.  Search
            distance defaults to 2.5x radii['total'].

        Examples
        --------
        >>> obj.member_search(blackholes=False)

        """
        import caesar.assignment as assign
        import caesar.linking as link

        self._args   = args
        self._kwargs = kwargs

        if 'v01_member_search' in self._kwargs and self._kwargs['v01_member_search']:
            from caesar.fubar import fubar
            self.data_manager._member_search_init()
            fubar(self, 'halo')
            fubar(self, 'galaxy')
            fubar(self,'cloud')
        else:
            from caesar.fubar_halo import fubar_halo
            fubar_halo(self)
            assign.assign_galaxies_to_halos(self)
            assign.assign_clouds_to_galaxies(self)

        link.link_galaxies_and_halos(self)
        link.link_clouds_and_galaxies(self)
        assign.assign_central_galaxies(self)
        link.create_sublists(self)

        #import caesar.hydrogen_mass_calc as mass_calc
        #mass_calc.hydrogen_mass_calc(self)

        from caesar.zoom_funcs import all_object_contam_check
        all_object_contam_check(self)


    def vtk_vis(self, **kwargs):
        """Method to visualize an entire simulation with VTK.
        
        Parameters
        ----------
        obj : :class:`main.CAESAR`
            Simulation object to visualize.
        ptypes : list
            List containing one or more of the following: 
            'dm','gas','star', which dictates which particles to 
            render.
        halo_only : boolean
            If True only render particles belonging to halos.
        galaxy_only: boolean
            If True only render particles belonging to galaxies.  Note 
            that this overwrites ``halo_only``.
        annotate_halos : boolean, list, int, optional
            Add labels to the render at the location of halos 
            annotating the group ID and total mass.  If True then all 
            halos are annotated, if an integer list then halos of 
            those indexes are annotated, and finally if an integer 
            than the most massive N halos are annotated.
        annotate_galaxies : boolean, list, int, optional
            Add labels to the render at the location of galaxies 
            annotating the group ID and total mass.  If True then all
            galaxies are annotated, if an integer list then galaxies 
            of those indexes are annotated, and finally if an integer 
            than the most massive N galaxies are annotated.
        
        """    
        self.data_manager.load_particle_data()
        from caesar.vtk_funcs import sim_vis
        sim_vis(self, **kwargs)

        
    def galinfo(self, top=10):
        """Method to print general info for the most massive galaxies
        identified via CAESAR.

        Parameters
        ----------
        top : int, optional
            Number of results to print.  Defaults to 10.

        Notes
        -----
        This prints to terminal, and is meant for use in an 
        interactive session.

        """
        from caesar.utils import info_printer
        info_printer(self, 'galaxy', top)

    def haloinfo(self, top=10):
        """Method to print general info for the most massive halos
        identified via CAESAR.

        Parameters
        ----------
        top : int, optional
            Number of results to print.  Defaults to 10.

        Notes
        -----
        This prints to terminal, and is meant for use in an 
        interactive session.

        """
        from caesar.utils import info_printer
        info_printer(self, 'halo', top)
        
        
    def cloudinfo(self, top=10):
        """Method to print general info for the most massive clouds
        identified via CAESAR.

        Parameters
        ----------
        top : int, optional
            Number of results to print.  Defaults to 10.

        Notes
        -----
        This prints to terminal, and is meant for use in an 
        interactive session.

        """
        from caesar.utils import info_printer
        info_printer(self, 'cloud', top)
