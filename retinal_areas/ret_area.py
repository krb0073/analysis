import numpy as np
import loos
import loos.pyloos
import sys
from scipy.spatial import ConvexHull, convex_hull_plot_2d
""" this scirpt find the area of the ring of retial give a psf and DCD and file prefix using
Convexhull """
def find_ring_pos(psf,dcd):
    """ load dcd into the computer the make a matrix of X,Y,Z to use """
    model = loos.createSystem(psf)
    traj = loos.pyloos.Trajectory(dcd,model)
    # We are find the postion using the vector from c15
    base = loos.selectAtoms(model,'resname =~ "^(RET|RTNH)$" && name == "C15"')
    ring = loos.selectAtoms(model,'resname =~ "^(RET|RTNH)$" && name =~ "C[1-6]"')
    # make one array that is size 3 by the number of frames
    pos = np.zeros((len(traj),3))
    for frame in traj:
        # find the com for each
        base_c = base.centroid()
        ring_c = ring.centroid()
        # find vector postion from that frame
        pos[traj.index()] = [x for x in ring_c - base_c]
    return pos

def find_ring_area(pos):
    """ use convex hull in scipy to find the area of the ring uses"""
    data = pos[:,0:2]
    # we only need those two for this because is about constant
    return ConvexHull(data).volume

def find_retinal_angel(pos):
    """ use the postions of the ring to the ange from the average postion the retinal makes"""
    # we will use x,y,z for this. finding the avg postion
    avg = pos.mean(axis=0)
    # have make avg the same size of the postion using np tile
    cosO = np.zeros((pos.shape[0]))
    mag_avg = np.linalg.norm(avg)
    cnt = 0
    for p in pos:
        # find the cos0 before the arc cos
        top = np.dot(p,avg)
        bottom = np.linalg.norm(p) * mag_avg
        cosO[cnt] = top/bottom
        cnt += 1
    # find the ang by acr cosine
    return np.degrees(np.arccos(cosO))

def main(psf,dcd,prefix):
    """ runs all command and write files to be used later """
    postions = find_ring_pos(psf,dcd)
    area = find_ring_area(postions)
    # now we will write out the postion array
    header = f"""# {psf} {dcd} {sys.argv[0]}
    # Ring Area {area}"""
    np.savetxt(prefix+"_postion_of_ring.txt",postions,header=header)
    # now find the angel of the retinal
    angels = find_retinal_angel(postions)
    header = f"""# {psf} {dcd} {sys.argv[0]}
    # angles from the avg postion
    # AVG ANG {angels.mean()}
    # STD {angels.std()}
    # POSAVG {*pos.mean(axis=1)}"""
    np.savetxt(prefix+"_angel_of_RET.txt",angels)


if __name__ == '__main__':
    psf = sys.argv[1]
    dcd = sys.argv[2]
    prefix = sys.argv[3]
    main(psf,dcd,prefix)
