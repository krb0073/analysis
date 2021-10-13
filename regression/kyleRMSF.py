# @Author: Kyle Billings <kbillings>
# @Date:   2020-06-26T17:35:56-04:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: kyleRMSF.py
# @Last modified by:   kbillings
# @Last modified time: 2020-06-29T19:55:36-04:00
import sys
import numpy as np
import loos
import loos.pyloos
PSF=str(sys.argv[1]) # sys argument for PSF
sel=sys.argv[2] # selection
dataFile=sys.argv[3] # file with the random frames
DCD=sys.argv[4] # dcd with all the frames in it
def grabCol(FILE,col1,delim=' '): # this will be all the data in the file
    import csv
    allVals=[]
    with open( FILE , 'r' ) as csvfile:
        plots = csv.reader(csvfile, delimiter=delim)
        for row in plots:
            if "#"  not in row:
                allVals.append(float(row[col1]))
    return allVals
data = grabCol(dataFile,0) # the data is only in the first column
s = loos.createSystem(PSF) # loos systenm temp.psf
mainSel =  loos.selectAtoms(s , sel) # making the atom sel
# split by atom
cnt = 0 # counter for the first frame
Traj =loos.pyloos.Trajectory(DCD,s,skip=1,stride=1,) # traj in loos
for i in data: # froe frami in the list of random frames
    x=[] # psotion arrays
    y=[]
    z=[]
    Traj.readFrame(int(i)) # read the given frame
    for A in mainSel: # for atom in the selection
        pos = A.coords() # get the  atom coords
        x.append(pos[0]) # add the postion to their array
        y.append(pos[1])
        z.append(pos[2])
    if cnt == 0: # on pass one
        px = np.array((x)) # build start array
        py = np.array((y))
        pz = np.array((z))
        cnt += 1 # now on a frame larger than one
    else:
        px = np.row_stack((px,x)) # add row to p matries
        py = np.row_stack((py,y))
        pz = np.row_stack((pz,z))
# Rref will be the avarge postion of each
AVGX = np.mean(px ,axis = 0) # find the avarge postion which will be used as the
## ref psotion
AVGY = np.mean(py ,axis = 0)
AVGZ = np.mean(pz, axis = 0)
# r^2 matrix
# get the shape of the AVG
numRes = px.shape[0] # lenght of the nubmer of resiures
numFrame = px.shape[1] # lenght for number of frames
#Rmatrix = np.zeros((numFrame,numRes)) # empty array
diffx = px - AVGX
diffy = py - AVGY
diffz = pz - AVGZ
R_2 = np.square(diffx) + np.square(diffy) + np.square(diffz)
#print(R_2.shape)
AVG_R_2 = np.mean(R_2,axis=0)
RMSF = np.sqrt(AVG_R_2)
#print(RMSF)
AUC = 0
AUC_ring =0
ringAtom = ['C1', 'C2' , 'C3','C4', 'C5' ,'C6' ,'C18','C16','C17']
for i in range(len(mainSel) -1 ):
    print(mainSel[i].resid(),mainSel[i].name(), RMSF[i])
    AUC += RMSF[i]
    if mainSel[i].name() in ringAtom:
        AUC_ring += RMSF[i]
print('#AUC {0} AUC_ring {1}'.format(AUC,AUC_ring))
