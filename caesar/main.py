from caesar.property_getter import DatasetType
from caesar.particle_list import ParticleListContainer
from caesar.simulation_attributes import SimulationAttributes

from yt.funcs import mylog, get_hash

VERSION = '0.1'

class CAESAR(object):
    """Master CAESAR class.

    CAESAR objects contain all references to halos and galaxies for
    a single snapshot.  Its output format is portable and global
    object statistics can be examined without the raw simulation file.
    
    Parameters
    ----------
    ds : yt dataset, optional
        A dataset via ``ds = yt.load(snapshot)``

    Examples
    --------
    >>> import caesar
    >>> obj = caesar.CAESAR()

    """
    def __init__(self, ds=0, *args, **kwargs):        
        self._args   = args
        self._kwargs = kwargs
        self._ds     = 0

        self.units = dict(
            mass='Msun',
            length='kpccm',
            velocity='km/s',
            time='yr',
            temperature='K'
        )

        self.global_particle_lists = ParticleListContainer(self)
        self.simulation = SimulationAttributes()
        self.yt_dataset = ds

        self.nhalos    = 0
        self.ngalaxies = 0
        
    @property
    def yt_dataset(self):
        """The yt dataset to perform actions on."""
        return self._ds
    @yt_dataset.setter
    def yt_dataset(self,value):
        if value == 0: return

        if not hasattr(value, 'dataset_type'):
            raise IOError('not a yt dataset?')

        infile = '%s/%s' % (value.fullpath, value.basename)

        if hasattr(self, 'hash'):
            hash = get_hash(infile)
            if hash != self.hash:
                raise IOError('hash mismatch!')
            else:
                self._ds = value
        else:
            self._ds  = value
            self.hash = get_hash(infile)

        self._ds_type = DatasetType(self._ds)
        self._assign_simulation_attributes()

    @property
    def _has_galaxies(self):
        """Checks if any galaxies are present."""
        if self.ngalaxies > 0:
            return True
        else:
            return False

    def _load_data(self):
        """Performs disk IO for particle/field data."""
        if hasattr(self, 'DataManager'):
            return
        if isinstance(self.yt_dataset, int):
            raise Exception("No yt_dataset assigned!")
        from caesar.data_manager import DataManager
        self.data_manager = DataManager(self)
        
    def _assign_simulation_attributes(self):
        """Populate the `caesar.simulation_attributes.SimulationAttributes`
        class."""
        self.simulation.create_attributes(self)

    def _assign_objects(self):
        """Assign galaxies to halos, and central galaxies."""
        import caesar.assignment as assign
        assign.assign_galaxies_to_halos(self)
        assign.assign_central_galaxies(self)
        
    def _link_objects(self):
        """Link galaxies to halos and create sublists."""
        import caesar.linking as link
        link.link_galaxies_and_halos(self)
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
        from disk, creating halos and galaxies, linking objects
        together, and finally calculating HI/H2 masses if necessary.

        Parameters 
        ---------- 
        blackholes : boolean 
            Indicate if blackholes are present in your simulation.  
            This must be toggled on manually as there is no clear 
            cut way to determine if PartType5 is a low-res particle, 
            or a black hole.
        
        b_halo : float
            Quantity used in the linking length (LL) for halos.
            LL = mean_interparticle_separation * b_halo.  Defaults to 
            ``b_halo = 0.2``.

        b_galaxy : float
            Quantity used in the linking length (LL) for galaxies.
            LL = mean_interparticle_separation * b_galaxy.  Defaults 
            to ``b_galaxy = b_halo * 0.2``.

        Examples
        --------
        >>> obj.member_search(blackholes=False)

        """
        self._args   = args
        self._kwargs = kwargs

        self._load_data()
        
        from caesar.fubar import fubar
        fubar(self, 'halo')
        fubar(self, 'galaxy')

        import caesar.assignment as assign
        import caesar.linking as link
        assign.assign_galaxies_to_halos(self)
        link.link_galaxies_and_halos(self)
        assign.assign_central_galaxies(self)
        link.create_sublists(self)
        
        import caesar.hydrogen_mass_calc as mass_calc
        mass_calc.hydrogen_mass_calc(self)


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
        
        """    
        self._load_data()
        from caesar.vtk_funcs import sim_vis
        sim_vis(self, **kwargs)
        
