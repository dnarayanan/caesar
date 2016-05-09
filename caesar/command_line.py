import argparse
import h5py
import os

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', help='input file', type=str)
    args = parser.parse_args()
    
    infile = args.infile

    caesar_file = False
    try:
        hd = h5py.File(infile, 'r')
        if 'caesar' in hd.attrs.keys() and hd.attrs['caesar']:
            caesar_file = True
        hd.close()
    except:
        pass
    
    if caesar_file:
        open_caesar_file(infile)        
    else:
        run_caesar(infile)

def open_caesar_file(infile):
    import IPython
    from loader import load
    print('')
    print("CAESAR file loaded into the 'obj' variable")
    print('')
    obj = load(infile)
    
    IPython.embed()

def run_caesar(infile):
    import yt
    ds = yt.load(infile)
    
    #if outfile is None:
    if infile.endswith('.bin') or infile.endswith('.dat'):
        outfile = 'caesar_%s.hdf5' % (infile[:-4])
    else:
        outfile = 'caesar_%s' % (infile) 

    from .caesar import CAESAR
        
    obj = CAESAR(yt.load(infile))
    obj.member_search()
    obj.save(outfile)
