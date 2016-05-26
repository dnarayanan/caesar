import argparse
import h5py
import os

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Input file or input directory')
    parser.add_argument('-o', '--output', type=str, help='Output file name')
    parser.add_argument('-b_halo',   type=float, help='Halo linking length')
    parser.add_argument('-b_galaxy', type=float, help='Galaxy linking length')
    parser.add_argument('-bh', '--blackholes', help='Black holes present?',
                        dest='OPTIONS', action='append_const', const='blackholes')
    parser.add_argument('-lr', '--lowres', type=int, help='Lowres particle types (Gadget/GIZMO HDF5 ONLY)', nargs='+')
    args = parser.parse_args()

    var_dict = vars(args)
    if args.OPTIONS is not None:
        for opt in args.OPTIONS:
            if opt not in var_dict:
                var_dict[opt] = True
            
    if os.path.isdir(args.input):
        run_multiple_caesar(args.input, var_dict)
        return

    if not os.path.isfile(args.input):
        raise IOException('not a valid file!')
    
    caesar_file = False
    try:
        hd = h5py.File(args.input, 'r')
        if 'caesar' in hd.attrs.keys() and hd.attrs['caesar']:
            caesar_file = True
        hd.close()
    except:
        pass
    
    if caesar_file:
        open_caesar_file(args.input)        
    else:
        run_caesar(args.input, var_dict)

        
def open_caesar_file(infile):
    import IPython
    from loader import load
    obj = load(infile)

    print('')
    print("CAESAR file loaded into the 'obj' variable")
    print('')

    IPython.embed()

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
        outfile = 'caesar_%s' % (infile) 

    from .main import CAESAR

    obj = CAESAR(yt.load(infile))
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
        
