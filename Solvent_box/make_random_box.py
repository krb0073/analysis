# @Author: Kyle Billings <kbillings>
# @Date:   2021-09-07T09:05:22-04:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: final_mox_maker.py
# @Last modified by:   kbillings
# @Last modified time: 2021-09-07T09:56:41-04:00
""" this code takes a given pdb of a solvent moleculae an places the a items
at random points in a given space this verson follows https://github.com/GrossfieldLab/loos/blob/main/Packages/OptimalMembraneGenerator/add_molecules.py"""
import numpy as np
import matplotlib.pyplot as plt
import loos
import loos.pyloos
import random
import sys
# comand line args go here
single_pdb = sys.argv[1]
number_of_mol = float(sys.argv[2])
box_side_lenght = float(sys.argv[3])
out_pdb = sys.argv[4]

# vatiable that i define befor the loops

# max number of trys before we exit
max = 2000
trial = 0
good = 0
#cnt
# load the pdb
system = loos.PDB(single_pdb)
box = loos.GCoord(box_side_lenght, box_side_lenght, box_side_lenght) # point of the farets point
half_box = box_side_lenght *.5 # half the box

solv = loos.AtomicGroup() # empty atomic group instance to append molecules

# start ton add stuff in
while good < number_of_mol:
    trial +=1

    # find the raondom point to move the item
    xt,yt,zt = random.uniform(-half_box, half_box) ,random.uniform(-half_box, half_box) ,random.uniform(-half_box, half_box)
    trans = loos.GCoord(xt,yt,zt)
    cop = system.copy() # making a copy the mol
    cop.centerAtOrigin() # moving to the orgin
    cop.translate(trans) # ,ove the the random chosen point
    # rotate the mol picking a random atomts corrd from cop a random degres
    cop.rotate(cop.centroid(),random.randint(-180,180))
    if not solv.contactWith(6.,cop,box):
        # fix resid number
        good += 1
        for i in range(len(cop)):
            cop[i].resid(good)
        solv.append(cop)

print("Placed ", good, " molecules in ", trial, " trials: ",good/trial * 100, " %")

solv.renumber()
solv.clearBonds()
pdb = loos.PDB.fromAtomicGroup(solv)

with open(out_pdb,"w") as out:
    out.write(str(pdb))
