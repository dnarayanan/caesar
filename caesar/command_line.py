import argparse
import h5py
import os

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input file or input directory', type=str)
    args = parser.parse_args()
    
    input = args.input

    if os.path.isdir(input):
        run_multiple_caesar(input)
        return

    if not os.path.isfile(input):
        raise IOException('not a valid file!')
    
    caesar_file = False
    try:
        hd = h5py.File(input, 'r')
        if 'caesar' in hd.attrs.keys() and hd.attrs['caesar']:
            caesar_file = True
        hd.close()
    except:
        pass
    
    if caesar_file:
        open_caesar_file(input)        
    else:
        run_caesar(input)

        
def open_caesar_file(infile):
    import IPython
    from loader import load
    obj = load(infile)

    print('')
    print("CAESAR file loaded into the 'obj' variable")
    print('')

    IPython.embed()

def run_caesar(infile):
    import yt
    
    #if outfile is None:
    if infile.endswith('.bin') or infile.endswith('.dat'):
        outfile = 'caesar_%s.hdf5' % (infile[:-4])
    else:
        outfile = 'caesar_%s' % (infile) 

    from .main import CAESAR
        
    obj = CAESAR(yt.load(infile))
    obj.member_search()
    obj.save(outfile)


def run_multiple_caesar(dir):
    import glob

    # look for hdf5 files
    infiles = glob.glob('*.hdf5')
    if len(infiles) == 0:
        infiles = glob.glob('*.bin')
    if len(infiles) == 0:
        raise IOError('Could not locate any hdf5 or bin files in %s!' % dir)


    for f in infiles:
        try:
            run_caesar(f)
        except:
            print 'failed on %s' % f
            pass
        
