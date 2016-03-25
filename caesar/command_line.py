import argparse

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', help='input file', type=str)
    args = parser.parse_args()
    
    print args.infile

    from main import CAESAR
    n = CAESAR(args.infile, test='yay')
    
