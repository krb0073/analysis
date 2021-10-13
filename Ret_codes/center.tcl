# neeed to load in the op
#mol load pdb opm.pdb
# make a pdb for only chain A
set CHA [atomselect top 'protein and chain A']
$CHA writepdb opm-A.pdb

mol delete 0

mol load pdb opm-A.pdb

set A [atomselect top all ]

set center [measure center $A weight mass]
set mver [vesub (0 0 0) $A]
$A moveby $mver
set A [atomselect top all ]
$A writepdb cnt_xtal.pdb
# load in the water
mol load pdb DOW/dowser_all.pdb

# set variable for the center of the protein
set B [atomselect top all ]
$B moveby $mver

set B [atomselect top all ]
$B writepdb c_wat.pdb
