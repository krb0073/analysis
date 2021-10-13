# @Author: Kyle Billings <kbillings>
# @Date:   2021-01-13T15:17:42-05:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: RMSF_runner.py
# @Last modified by:   kbillings
# @Last modified time: 2021-01-14T13:41:00-05:00
import numpy as np
import pandas as pd
import os
import sys
import glob
import random
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
import shutil
import statsmodels.api as sm
import loos
import loos.pyloos
path = os.path.abspath('../goodRangeDCD/')
seed = 1600877851
st = 0
end =  558940
num = 10
def scramble(array,seedNum,axis=-1):
    import numpy as np
    np.random.seed()
    swapped = array.swapaxes(axis,-1)
    n = array.shape[axis]
    idx = np.random.choice(n, n, replace=False)
    swapped = swapped[..., idx]
    return swapped.swapaxes(axis, -1)
def run_RMSF_rand(seed,num_of_sets,start,end,path,sel = 'resname =~ "^(RET|RTNH)" || (resname == "LYS" && resid == 216)'):
    ringAtom = ['C1', 'C2' , 'C3','C4', 'C5' ,'C6' ,'C18','C16','C17']
    # getting the state for the file name
    DIR = str(seed)+"_RMSF/"
    try:
        os.mkdir(DIR)
    except:
        shutil.rmtree(DIR)
        os.mkdir(DIR)
    path_to_DIR = os.path.abspath(DIR)
    print(path_to_DIR)
    np.random.seed(seed)
    number = end - start
    frame_list= np.arange(start,end)
    current_loc = os.getcwd()
    full_path = os.path.abspath(os.getcwd()+path)
    os.chdir(path)
    dcd_files= glob.glob('*goodAll.dcd')
    dcd_files.sort()
    psf_files = glob.glob('*good*.pdb')
    psf_files.sort()
    for psf , dcd in zip(psf_files,dcd_files):
        state = dcd.split("/")[0].split("_")[0]
        print(state)
        print(psf,dcd,"is running RMSF")
        system = loos.createSystem(psf)
        Traj = loos.pyloos.Trajectory(dcd,system)
        ret_atoms = loos.selectAtoms(system,sel)
        ATOM_NAMES = []
        for atom in range(len(ret_atoms.getCoords())):
            ATOM = ret_atoms.getAtom(atom)
            print(ATOM.name(),ATOM.resname(),ATOM.resid())
            X = []
            Y = []
            Z = []
            ATOM_NAMES.append(ATOM.name())
            for frame in frame_list:
                Traj.readFrame(frame)
                x,y,z = ret_atoms.getCoords()[atom]
                X.append(x)
                Y.append(y)
                Z.append(z)
            X = np.array(X).reshape(num_of_sets,-1)
            Y = np.array(Y).reshape(num_of_sets,-1)
            Z = np.array(Z).reshape(num_of_sets,-1)
            # with these built we will now use the seed we have to split the thing on
            mixed_X = scramble(X,seed,axis=1)
            mixed_Y = scramble(Y,seed,axis=1)
            mixed_Z = scramble(Z,seed,axis=1)
            # gathere the mean vaule of each row
            mean_x, mean_y,mean_z = np.average(mixed_X,axis=1) , np.average(mixed_Y,axis=1) , np.average(mixed_Z,axis=1)
            diff = (mixed_X-mean_x[:,None]) # substract the arrays
            diff_squared = diff**2
            mean_diff = np.average(diff_squared,axis = 1)
            RMSF = mean_diff**0.5
            if atom == 0:
                ALL_DATA = np.array(RMSF)
            else:
                ALL_DATA = np.column_stack((ALL_DATA,RMSF))
            if ATOM.name() in ringAtom:
                if ATOM.name() == ringAtom[0]:
                    RING_ONLY = np.array(RMSF)
                else:
                    RING_ONLY = np.column_stack((RING_ONLY,RMSF))
        n_array_DATA = np.array_split(ALL_DATA,num_of_sets)
        n_array_RING = np.array_split(RING_ONLY,num_of_sets)
        header = "#" + " ".join(ATOM_NAMES)
        n = 1
        for d ,r in zip(n_array_DATA,n_array_RING):
            print(path_to_DIR+"/" +state + f"_{seed}_set_{n}.dat")
            file = open(path_to_DIR+"/" +state + f"_{seed}_set_{n}.dat" ,'w')
            file.write(header)
            vals = [str(x) for x in d]
            np.set_printoptions(linewidth=np.inf)
            file.write(vals[0].strip('\n').strip('[').strip(']'))
            file.write('\n')
            file.write(f"AUC {np.sum(d)} \n")
            file.write(f"ring_AUC {np.sum(r)} \n")
            file.close()
            n += 1
run_RMSF_rand(seed,num,st,end,path,sel = 'resname =~ "^(RET|RTNH)" || (resname == "LYS" && resid == 216)')
