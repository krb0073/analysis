# @Author: Kyle Billings <kbillings>
# @Date:   2020-06-28T13:56:00-04:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: testVtraj.py
# @Last modified by:   kbillings
# @Last modified time: 2020-06-29T16:35:22-04:00
import sys
import loos
import loos.pyloos

PSF=sys.argv[1]
DCD=sys.argv[2:]
frames = [1,2,5,10,100,50]
s = loos.createSystem(PSF)
Traj =loos.pyloos.Trajectory(DCD[0],s,skip=1,stride=1,) # traj in loos
Vtraj = loos.pyloos.VirtualTrajectory(Traj)
for t in DCD[1:]:
    traj = loos.pyloos.Trajectory(t ,s,skip=1,stride=1)
    Vtraj.append(traj)
for item in frames:
    Vtraj.readFrame(item)
    print("works")

# looking at dir to get the risght tyhing vtraj is notn able to do what i want
