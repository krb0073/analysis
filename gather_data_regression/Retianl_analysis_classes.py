# @Author: Kyle Billings <kbillings>
# @Date:   2021-05-20T16:33:48-04:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: Retianl_analysis.py
# @Last modified by:   kbillings
# @Last modified time: 2021-06-03T15:53:27-04:00


import loos
import loos.pyloos
import sys
import numpy as np
import os
import multiprocessing as mp
# all the function i need to use in one place
config = sys.argv[1]
def dihe_points(p1 , p2 , p3 , p4):
"""turns four points into numpy arrays"""
    val1 = p1.centerOfMass()
    val2 = p2.centerOfMass()
    val3 = p3.centerOfMass()
    val4 = p4.centerOfMass()
    def toNp(point):
        p = np.zeros(3)
        p[:] = [point[0] , point[1] , point[2]]
        return p
    P1= toNp(val1)
    P2 = toNp(val2)
    P3 = toNp(val3)
    P4 = toNp(val4)
    return P1 , P2 , P3 , P4
def calc_q_vectors(P1,P2,P3,P4):
"""Function to calculate q vectors"""
    import numpy as np
# Calculate coordinates for vectors q1, q2 and q3
    q1 = np.subtract(P2,P1) # b - a
    q2 = np.subtract(P3,P2) # c - b
    q3 = np.subtract(P4,P3) # d - c
    return q1,q2,q3
def calc_cross_vectors(q1,q2,q3):
    """Function to calculate cross vectors"""
    import numpy as np
    # Calculate cross vectors
    q1_x_q2 = np.cross(q1,q2)
    q2_x_q3 = np.cross(q2,q3)
    return q1_x_q2, q2_x_q3

def calc_nornals(q1_x_q2,q2_x_q3):
    """Function to calculate normal vectors to planes"""
    import numpy as np
    # Calculate normal vectors
    n1 = q1_x_q2/np.sqrt(np.dot(q1_x_q2,q1_x_q2))
    n2 = q2_x_q3/np.sqrt(np.dot(q2_x_q3,q2_x_q3))
    return n1,n2

def calc_orthogonal_unit_vectors(n2,q2):
"""Function to calculate orthogonal unit vectors"""
    import numpy as np
    # Calculate unit vectors
    u1 = n2
    u3 = q2/(np.sqrt(np.dot(q2,q2)))
    u2 = np.cross(u3,u1)
    return u1,u2,u3

def calc_dihedral_angle(n1,u1,u2,u3):
    """Function to calculate dihedral angle"""
    import numpy as np
    import math
    # Calculate cosine and sine
    cos_theta = np.dot(n1,u1)
    sin_theta = np.dot(n1,u2)
    # Calculate theta
    theta = -math.atan2(sin_theta,cos_theta) # it is different from Fortran math.atan2(y,x)
    theta_deg = np.degrees(theta)
    if theta_deg < 0:
        theta_deg += 360
    return theta_deg

def obtain_dihedral_angel(sel1,sel2,sel3,sel4):
""" using  four selections output a diehdral angle"""
    p1 , p2 , p3 , p4 = dihe_points(sel1, sel2 ,sel3, sel4)
    q1,q2,q3 = calc_q_vectors(p1,p2,p3,p4)
    # Call calc_cross_vectors(q1,q2,q3) function
    q1_x_q2, q2_x_q3 = calc_cross_vectors(q1,q2,q3)
    # Call calc_nornalss(q1_x_q2,q2_x_q3) function
    n1, n2 = calc_nornals(q1_x_q2,q2_x_q3)
    # Call calc_orthogonal_unit_vectors(n2,q2) function
    u1,u2,u3 = calc_orthogonal_unit_vectors(n2,q2)
    # Call calc_dihedral_angle(u1,u2,u3) function
    theta_deg = calc_dihedral_angle(n1,u1,u2,u3)
    return theta_deg
def toNp(point):
"""make any point into a numpy array"""
    p = np.zeros(3)
    p[:] = [point[0] , point[1] , point[2]]
    return p
def ANG(p1,p2,p3):
"""find the angle between 3 atoms"""
    ba = p1 - p2
    bc = p3 - p2
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    deg = np.degrees(angle)
    return deg
def obtain_user_angel(sel1,sel2,sel3):
"""using Three selections find the angle between the points"""
    val1 = sel1.centerOfMass()
    val2 = sel2ProcessPoolExecutor(max_workers=3).centerOfMass()
    val3 = sel3.centerOfMass()
    P1= toNp(val1)
    P2 = toNp(val2)
    P3 = toNp(val3)
    angel = ANG(P1,P2,P3)
    return angel

def vectorize(A,B): # f(x)n to make a vector
    return np.subtract(A,B) # substracthe two point from eachother
def vectorAngles(A,B): # to find the angle
    """ find the angle between 2 vectors"""
    # https://newtonexcelbach.com/2014/03/01/the-angle-between-two-vectors-python-version/
    cosang = np.dot(A,B)
    sinang = np.linalg.norm(np.cross(A,B))
    return  np.arctan2(sinang, cosang) * 180 / np.pi
class readConfig:
"""Class to read a confing flie to limit the number of arugments"""
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
class system_setup(readConfig):

    def load_traj(self,model,traj):
        self.model_name = model
        self.traj_name = traj
        try:
            self.model = loos.createSystem(self.model_name)
        except:
            print("model was not able to be loaded")
            sys.exit(0)
        # now with the psf in we can do the same for the traj
        try:
            self.traj = loos.pyloos.Trajectory(self.traj_name,self.model,skip=self.skip,stride = self.stride)
        except:
            print("traj could not not be loaded")
            sys.exit(0)
    def find_dihe_overtime(self,dihe_sel,system_name,psf,dcd):
"""from that atoms given the the config file we find the DIHE over time """
        self.load_traj(psf,dcd)
        atom1 ,atom2,atom3,atom4 = dihe_sel.split()
        Sel1 = loos.selectAtoms(self.model, self.ret_name +' '+f'&& name == "{atom1}"')
        Sel2 = loos.selectAtoms(self.model,self.ret_name +' '+f'&& name == "{atom2}"')
        Sel3 = loos.selectAtoms(self.model, self.ret_name +' '+f'&& name == "{atom3}"')
        Sel4 = loos.selectAtoms(self.model, self.ret_name +' '+f'&& name == "{atom4}"')
        int_frame = 0 + self.skip
        atoms_in_dihe = "_".join(dihe_sel.split())

        fname = f"{system_name}_dihe_{atoms_in_dihe}.dat"
        file = open(self.out + "/" + fname , 'w')
        for frame in self.traj:
            p1 , p2 , p3 , p4 = dihe_points(Sel1, Sel2 ,Sel3, Sel4)
            q1,q2,q3 = calc_q_vectors(p1,p2,p3,p4)
            q1_x_q2, q2_x_q3 = calc_cross_vectors(q1,q2,q3)
            n1, n2 = calc_nornals(q1_x_q2,q2_x_q3)
            u1,u2,u3 = calc_orthogonal_unit_vectors(n2,q2)
            theta_deg = calc_dihedral_angle(n1,u1,u2,u3)
            ans = f"{int_frame} {theta_deg} \n"
            file.write(ans)
            int_frame += self.stride
        file.close()
    def find_angel_overtime(self,ang_sel,system_name,psf,dcd):
"""find the anagle using the config"""
        self.load_traj(psf,dcd)
        atom1 ,atom2 ,atom3 = ang_sel.split()
        Sel1 = loos.selectAtoms(self.model, self.ret_name +' '+f'&& name == "{atom1}"')
        Sel2 = loos.selectAtoms(self.model,self.ret_name +' '+f'&& name == "{atom2}"')
        Sel3 = loos.selectAtoms(self.model, self.ret_name +' '+f'&& name == "{atom3}"')
        f = 0 +  self.skip
        atoms_in_ang = "_".join(ang_sel.split())
        fname = f"{system_name}_ang_{atoms_in_ang}.dat"
        file = open(self.out + "/" + fname , 'w')
        for frame in self.traj:
            val1 = Sel1.centerOfMass()
            val2 = Sel2.centerOfMass()
            val3 = Sel3.centerOfMass()
            P1= toNp(val1)
            P2 = toNp(val2)
            P3 = toNp(val3)
            ang = ANG(P1,P2,P3)
            ans = f"{f} {ang}\n"
            file.write(ans)
            f += self.stride
        file.close()
    def find_ornatation_to_normal(self,orination_atoms,system_name,psf,dcd):
"""find the ornatation to the bilayer normal using the selection in the confing"""
        self.load_traj(psf,dcd)
        lipid = loos.selectAtoms(self.model,self.lipid_sel)
        atom1 , atom2 = orination_atoms.split()
        Sel1 = loos.selectAtoms(self.model, self.ret_name +' '+f'&& name == "{atom1}"')
        Sel2 = loos.selectAtoms(self.model,self.ret_name +' '+f'&& name == "{atom2}"')
        f = 0 +  self.skip
        atoms_in_or = "_".join(orination_atoms.split())
        fname = f"{system_name}_orin_{atoms_in_or}.dat"
        file = open(self.out + "/" + fname , 'w')
        for frame in self.traj:
            COM = lipid.centerOfMass()
            tempCOM = np.array((COM[0],COM[1],COM[2])) # COM to array
            box = (frame.periodicBox()) # getting the pbc mesurmwents
            Xbox = box[0]/2 # from the center half the box mesument should be to +
            Ybox = box[1]/2 ## this is done for each
            v1 = np.array((Xbox,COM[1],COM[2])) # vector with on different in x
            v2 = np.array((COM[0],Ybox,COM[2])) # diff in Y
            vector1 = vectorize(tempCOM,v1) # make  vectors
            vector2 = vectorize(tempCOM,v2)
            normal = np.cross(vector1,vector2) #cross product to find the normal should
            ## be in Z only if centered
            ########################################
            # using the atom supplied by the user make a vector
            p1 = Sel1.centroid()
            p2 = Sel2.centroid()
            point1 = np.array((p1[0],p1[1],p1[2]))
            point2 = np.array((p2[0],p2[1],p2[2]))
            vector3 = vectorize(point1,point2)
            angle = vectorAngles(vector3,normal) # angel between the methyl & norm
            ans = f'{f} {angle}\n'
            file.write(ans)
            f += self.stride
    def main(self):
        for psf, dcd , name in zip(self.models,self.trajs,self.system_name):
            if len(self.dihe) != 0:
                maps = []
                for d in self.dihe:
                    maps.append((d,name,psf,dcd))
                pool = mp.Pool(processes=self.n_cores)
                L = pool.starmap_async(self.find_dihe_overtime,maps)
                pool.close()
                pool.join()
                maps = []
            if len(self.user_angs) != 0:
                for a in self.user_angs:
                    maps.append((a,name,psf,dcd))
                pool = mp.Pool(processes=self.n_cores)
                L = pool.starmap_async(self.find_angel_overtime,maps)
                pool.close()
                pool.join()
                maps = []
            if len(self.orination) != 0:
                for o in self.orination:
                    maps.append((o,name,psf,dcd))
                pool = mp.Pool(processes=self.n_cores)
                L = pool.starmap_async(self.find_ornatation_to_normal,maps)
                pool.close()
                pool.join()


if __name__ in "__main__":
	run  = system_setup(config)
	run.main()
