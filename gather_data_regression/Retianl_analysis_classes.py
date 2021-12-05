
import loos
import loos.pyloos
import sys
import numpy as np
import os
import multiprocessing as mp
from math import atan2


def dihedral(selections):
    """ find the dihedral angel between two points """
    # loos Gcoord into numpy array

    p1 = np.array([x for x in selections[0].centroid()])
    p2 = np.array([x for x in selections[1].centroid()])
    p3 = np.array([x for x in selections[2].centroid()])
    p4 = np.array([x for x in selections[3].centroid()])
    # bais vectors
    b1 = -1.0*(p2-p1)
    b2 = p3 - p2
    b3 = p4 - p3
    # find cross poduction
    b1xb2 = np.cross(b1,b2)
    b2xb3 = np.cross(b2,b3)
    # find normal between them
    b1b2xb2b3 = np.cross(b1xb2,b2xb3)
    b2_norm = b2/np.linalg.norm(b2)
    dot1 = np.dot(b1b2xb2b3,b2_norm)
    dot2 = np.dot(b1xb2 ,b2xb3)
    ans =  np.degrees(np.arctan2(dot1,dot2))
    return ans  # not this will have


def angel(selections):
    """given the selections from loos we will find the user given angel"""

    p1 = np.array([x for x in selections[0].centroid()])
    p2 = np.array([x for x in selections[1].centroid()])
    p3 = np.array([x for x in selections[2].centroid()])
    # makeing the vectos to fins the angel assume p2 is between  p1 and p3
    p1xp2 = p1 - p2
    p3xp2 = p3 - p2
    # now we will find the angels that I want
    top = np.dot(p1xp2,p3xp2)
    bottom =   np.linalg.norm(p1xp2) * np.linalg.norm(p3xp2)
    return np.degrees(np.arccos(top/bottom))

def unit_point_orination(normal,selctions):
    # change to np array
    normal = np.array(([x for x in nomral])) # unit vector alread
    # assumeiing p2 is the one we cant to point to
    p1 = np.array([x for x in selections[0].centroid()])
    p2 = np.array([x for x in selections[1].centroid()])
    p2xp1 = (p2 - p1) /np.linalg.norm(p2-p1) # convet to unit vector
    cos0 = normal * p2xp1 # they are both unit vector aleardy there normalized
    return np.degrees(np.arccos(cos0))

def startup(psf,dcd,skip=0,stride=1):
    model = loos.createSystem(psf)
    return model , loos.pyloos.Trajectory(dcd,model,skip=skip,stride=stride)

def multicore_async(function,maps,cores=2):
    """ a easy way to run any fuicntion in python synsocis will
    not work unless you have a working function to pass it to  """
    pool = mp.Pool(processes=cores)
    L = pool.starmap_async(function,maps)
    pool.close()
    pool.join()


class readConfig:
    def __init__(self,FILE):
        self.trajs = [] #
        self.dihe = [] #
        self.user_angs = [] #
        self.orination = [] #
        self.ret_name ='resname =~ "^(RET|RTNH)$" || (resname == "LYS" && resid == 216)' #
        self.skip = 0#
        self.stride = 1#
        self.lipid_sel = "resname =~ '^(POPE|POPG)$'"#
        self.models = []
        self.system_name = []
        self.out = './'
        self.n_cores = 2
    # now we open the file to read this
        file = open(FILE)
        # read the lines to find what we need
        for line in file.readlines():
            if line.startswith("#") or line.isspace() or len(line) == 0 or "@" in line:
                continue

            elif line.upper().startswith("DIHEDRAL"):
                # compile a list of dihedral
                atoms = line.split()[1:] #grab to the end
                if len(atoms) != 4:
                    print("wrong number of atoms given dihedral .... exiting\n")
                    sys.exit(0)
                atoms = ' '.join(atoms)
                self.dihe.append(atoms) # add this string to the list

            elif line.upper().startswith("ANGLE"):
                atoms = line.split()[1:] #grab to the end
                if len(atoms) != 3:
                    print("wrong number of atoms given for a user angle .... exiting\n")
                    sys.exit(0)
                atoms = ' '.join(atoms)
                self.user_angs.append(atoms) # should be therr atom names

            elif line.upper().startswith("ORINATION"):
                atoms = line.split()[1:] #grab to the end
                atoms = ' '.join(atoms)
                self.orination.append(atoms) # should be therr atom names

            elif line.upper().startswith("SKIP"):
                self.skip = int(line.split()[1])

            elif line.upper().startswith("STRIDE"):
                self.stride = int(line.split()[1])

            elif line.upper().startswith("RETINAL"):
                self.ret_name = line.split()[:1]


            elif line.upper().startswith("LIPID"):
                self.lipid_sel = ' '.join(line.split()[1:])

            elif line.upper().startswith("SYSTEM"):
                _ ,name ,model , traj = line.split()
                try:
                    os.path.isfile(model)
                    os.path.isfile(traj)
                except:
                    print("one of the the given file is not real ... exiting")
                    sys.exit(0)
                self.models.append(os.path.abspath(model))
                self.trajs.append(os.path.abspath(traj))
                self.system_name.append(name)
            elif line.upper().startswith("OUT"):
                out_dir = line.split()[1]
                try:
                    os.path.isdir(out_dir)
                except:
                    print(f'path is not real {out} ... exiting \n ')
                    sys.exit()
                self.out = os.path.abspath(out_dir)
            elif line.upper().startswith("CORES"):
                nc = int(line.split()[1])
                self.n_cores = nc

class CollectData(readConfig):
    """ run all of the files"""
    def __init__(self,*args,**kwargs):
        self.calc_type = None
        super().__init__(*args,**kwargs)

    def selector(self,atom_list,model):
        """ pass this a list of the atoms that you want and return loos selections"""
        return [loos.selectAtoms(model, f'{self.ret_name} && name == "{atom}"') for atom in atom_list.split()]

    def runner(self,psf,dcd,atom_list,system_name,calc_type):
        """ runs all of the calactions """
        # make the systems
        model , traj = startup(psf,dcd)
        # make selections
        selections = self.selector(atom_list,model)

        if calc_type == 0:
            # this is forb dihe
            fx = dihedral
            tag = 'dihe'
        elif calc_type == 1:
            fx = angel
            tag = 'ang'
        elif calc_type == 2:
            lipid_sel = loos.selectAtoms(model,self.lipid_sel)
            fx = unit_point_orination
            tag = 'ort'

        atom_names = '-'.join(atom_list.split())
        fname  = f'{system_name}_{tag}_{atom_names}.dat'
        with open(fname,'w') as log:
            for frame in traj:

                # we can now just run the claa
                if calc_type != 2:
                    ans = fx(selections)
                else:
                    norm = lipid_sel.lip.principalAxes()[0]
                    ans = fx(norm,selections)
                #print(f'{traj.index() * self.stride + self.skip} {ans}\n')
                line = f'{self.stride * traj.index() + self.skip} {ans}\n'
                log.write(line)
    def main(self):
        """ main block to run all code """
        for psf ,dcd ,name in  zip(self.models,self.trajs,self.system_name):
            if len(self.dihe) != 0:
                calc_type = 0
                # make a list of maps
                maps = []
                for d in self.dihe:
                    maps.append((psf,dcd,d,name,calc_type))
                multicore_async(self.runner,maps,cores=self.n_cores)
            if len(self.user_angs) != 0:
                maps = []
                self.calc_type = 1
                for a in self.user_angs:
                    maps.append((psf,dcd,a,name,calc_type))
                multicore_async(self.runner,maps,cores=self.n_cores)

            if len(self.orination) != 0:
                maps = []
                self.calc_type = 2
                for o in self.orination:
                    maps.append((psf,dcd,o,name,calc_type))
                multicore_async(self.runner,maps,cores=self.n_cores)

            print("done",psf,dcd)
if __name__ == '__main__':
    config = sys.argv[1]
    CollectData(config).main()
