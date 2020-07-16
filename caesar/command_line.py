import argparse
import h5py
import os
import numpy as np



def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Input file or input directory')
    parser.add_argument('-o', '--output', type=str, help='Output file name')
    parser.add_argument('-b_halo',   type=float, help='Halo linking length')
     #NOTES FROM BOBBY: why not -b_cloud througout?
    parser.add_argument('-ll_cloud',   type=float, help='Cloud linking length in kpccm')
    parser.add_argument('-b_galaxy', type=float, help='Galaxy linking length')
    parser.add_argument('-fof6d',help='Set 6D FOF for galaxies',
                        dest = 'OPTIONS', action='append_const',const='fof6d')
    parser.add_argument('-fof6d_file',type =str, help='Set fof6d filename for reading (if it exists) or writing (if not)')
    parser.add_argument('-fofclouds',help='Bool: Turn on 3D FOF for clouds',
                        dest = 'OPTIONS', action='append_const',const='fofclouds')
    parser.add_argument('-fof6d_mingrp',type=float,help='Set min group size for fof6d')
    parser.add_argument('-fof6d_LL_factor',type=float,help='Set linking length factor for fof6d')
    parser.add_argument('-fof6d_vel_LL',type=float,help='Set velocity linking length for fof6d, in units of local vel disp')
    parser.add_argument('-nproc', type=int, help='Set number of processors for fof6d and group property calculation', default=1)
    parser.add_argument('-bh', '--blackholes', help='Black holes present?',
                        dest='OPTIONS', action='append_const', const='blackholes')
    parser.add_argument('-d', '--dust', help='Active dust present?',
                        dest='OPTIONS', action='append_const', const='dust')
    parser.add_argument('-uh', '--unbind_halos', help='Unbind halos?',
                        dest='OPTIONS', action='append_const', const='unbind_halos')
    parser.add_argument('-ug', '--unbind_galaxies', help='Unbind galaxies?',
                        dest='OPTIONS', action='append_const', const='unbind_galaxies')
    parser.add_argument('-lr', '--lowres', type=int, help='Lowres particle types (Gadget/GIZMO HDF5 ONLY)', nargs='+')
    parser.add_argument('-q', '--quick', action="store_true", default=False, help='Use the quick-loading CAESAR backend if loading an existing output file')
    parser.add_argument('--use-the-old-and-slow-loader', action="store_true", default=False)
    args = parser.parse_args()

    if args.quick:
        import warnings
        warnings.warn('The quick-loader is now the default behavior. The -q and --quick flags will be removed soon.', stacklevel=2)

    var_dict = vars(args)
    if args.OPTIONS is not None:
        for opt in args.OPTIONS:
            if opt not in var_dict:
                var_dict[opt] = True

    if os.path.isdir(args.input):
        run_multiple_caesar(args.input, var_dict)
        return

    if not os.path.isfile(args.input):
        raise IOError('%s not a valid file!' % args.input)
    
    caesar_file = False
    try:
        hd = h5py.File(args.input, 'r')
        if 'caesar' in hd.attrs.keys():
            caesar_file = True
        hd.close()
    except:
        pass

    if caesar_file:
        if args.use_the_old_and_slow_loader:
            import IPython
            from .old_loader import load
            obj = load(args.input)

            print('')
            print("CAESAR file loaded into the 'obj' variable")
            print('')

            IPython.embed()
        else:
            open_caesar_file(args.input)
    else:
        run_caesar(args.input, var_dict)


def open_caesar_file(infile):
    import IPython
    from .loader import load
    obj = load(infile)

    IPython.embed(header="CAESAR file loaded into the 'obj' variable")


def run_caesar(infile, args):
    import yt
    
    if args['output'] is not None:
        if args['output'].endswith('.hdf5'):
            outfile = args['output']
        else:
            outfile = '%s.hdf5' % args['output']

    elif infile.endswith('.bin') or infile.endswith('.dat'):
        outfile = 'caesar_%s.hdf5' % (infile[:-4])

    else:
        if 'snapshot_' in infile:
            outfile = infile.replace('snapshot_','caesar_')
        elif 'snap_' in infile:
            outfile = infile.replace('snap_','caesar_')
        else:
            outfile = 'caesar_%s' % (infile) 

    from .main import CAESAR
    ds = yt.load(infile)
    if ds.cosmological_simulation == 1:
        obj = CAESAR(yt.load(infile))
    else:
        try:
            import pygadgetreader as pygr
        except ImportError:
            raise ImportError('Whoops! It looks like you need to install pygadgetreader: https://bitbucket.org/rthompson/pygadgetreader/src/default/')

        print('Figuring out the box size for a non-cosmological simulation')
        #find the min/max coordinates and use a bbox
        pos_dm = pygr.readsnap(infile,'pos','dm')
        pos_gas = pygr.readsnap(infile,'pos','gas')
        pos_stars = pygr.readsnap(infile,'pos','stars')
        

        #BOBBY NOTES: JUST LOAD THIS IN YT4.0 TO AVOID DEPENDENCIES ON PYGR
        maxpos = np.max(np.concatenate((pos_stars.flatten(),pos_dm.flatten(),pos_gas.flatten()),axis=0))
        minpos = np.min(np.concatenate((pos_stars.flatten(),pos_dm.flatten(),pos_gas.flatten()),axis=0))
        boxsize = np.max([np.absolute(maxpos),np.absolute(minpos)])
        bbox = [[-boxsize,boxsize],
                [-boxsize,boxsize],
                [-boxsize,boxsize]]
        obj = CAESAR(yt.load(infile,bounding_box = bbox))
        
    obj.member_search(**args)
    obj.save(outfile)


def run_multiple_caesar(dir, args):
    import glob

    # look for hdf5 files
    infiles = glob.glob('*.hdf5')
    if len(infiles) == 0:
        infiles = glob.glob('*.bin')
    if len(infiles) == 0:
        raise IOError('Could not locate any hdf5 or bin files in %s!' % dir)


    for f in infiles:
        try:
            run_caesar(f, args)
        except:
            print('failed on %s' % f)
            pass
        
