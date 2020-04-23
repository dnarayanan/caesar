
'''
This command-line script reduces snapshots for a simulation 
    SIM (e.g. m50n512) and wind model WIND (e.g. s50).
OVERWRITE tells it whether to skip doing a snapshot if the 
    Caesar file exists.

It assumes the snapshots are located in /home/USERNAME/data/SIM/WIND.
It creates a subdirectory Groups that holds the Caesar and fof6d files.
You can provide a snapshot list if desired, or else let it try to
    do all the available snapshots.

Requires reduce.py, which is the actual driver (which can also be
    used by itself).  This script mostly just compiles SNAPLIST.

example usage:
    % python REDUCE_.py m50n512 s50 True 78 105 151

Romeel Davé, 23 Apr 2020
'''

import sys
import os
import reduce as r

# to reduce a list of snapshots
SIM  = sys.argv[1]
WIND = sys.argv[2]
OVERWRITE = int(sys.argv[3])
if len(sys.argv) == 5:
    SNAPLIST = [int(i) for i in sys.argv[4:]]
else:
    SNAPLIST = range(10,152,1)

# set up directories
HOMEDIR = os.path.expanduser('~')
SNAPDIR = '%s/data/%s/%s'%(HOMEDIR,SIM,WIND)  # may need to change this based on your file organization
CAESARLOC = 'Groups'
FOF6DLOC = 'Groups'
CAESARDIR = '%s/%s' % (SNAPDIR,CAESARLOC)  # put Caesar files in 'Groups' subdirectory
os.makedirs(CAESARDIR,exist_ok=True)

# check that snaps exist
SNAPLIST_ALL = SNAPLIST.copy()
for SNAP in SNAPLIST_ALL:
    SNAPFILE = '%s/snap_%s_%03d.hdf5' % (SNAPDIR, SIM, SNAP)
    if not os.path.exists(SNAPFILE):
        SNAPLIST.remove(SNAP)

# check for already done Caesar files
if not OVERWRITE:
    SNAPLIST_ALL = SNAPLIST.copy()
    for SNAP in SNAPLIST_ALL:
        CAESARFILE = '%s/Groups/%s_%03d.hdf5' % (SNAPDIR, SIM, SNAP)
        if os.path.exists(CAESARFILE):
            SNAPLIST.remove(SNAP)

# special behavior: if negative, do first N available snaps
if int(sys.argv[4]) <= -1: 
    SNAPLIST = SNAPLIST[:-int(sys.argv[3])]


print('SNAPLIST=',SNAPLIST )

if len(SNAPLIST) > 0:
    r.reduce(SNAPLIST, SIM, SNAPDIR, CAESARLOC=CAESARLOC, FOF6DLOC=FOF6DLOC) 

