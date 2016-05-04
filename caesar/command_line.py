import argparse
import yt

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', help='input file', type=str)
    args = parser.parse_args()
    
    print args.infile

    from caesar import CAESAR
    obj = CAESAR(yt.load(args.infile), test='yay')
    obj.member_search()
    
