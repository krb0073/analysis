# @Author: Kyle Billings <kbillings>
# @Date:   2020-12-02T17:27:00-05:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: NAMD-Kyle.py
# @Last modified by:   kbillings
# @Last modified time: 2021-02-09T22:35:00-05:00
 # stole from https://github.com/GrossfieldLab/loos/blob/main/Packages/OptimalMembraneGenerator/NAMD.py
#!/usr/bin/env python3
import subprocess
import sys
import os.path
import loos
import loos.pyloos
import sys
import configparser
import shutil
class NAMD_KB:

    def __init__(self, psf_file, start_pdb, end_pdb, param_file, box,
                 command):
        self.psf_file = psf_file
        self.start_pdb = start_pdb
        self.end_pdb = end_pdb
        self.param_file = param_file

        if command is None:
            self.command = "/opt/bin/namd2 +p10"
        else:
            self.command = command


        # in case we're going to do constraints, construct this filename
        self.cons_k_filename = (self.start_pdb[:-4] + ".cons.pdb").split("/")[-1]
        self.x_box = box.x()
        self.y_box = box.y()
        self.z_box = box.z()

    def update_box(self, box):
        self.x_box = box.x()
        self.y_box = box.y()
        self.z_box = box.z()

    def construct_header(self):
        lines = []
        lines.append("structure " + self.psf_file)
        lines.append("coordinates " + self.start_pdb)
        lines.append("outputname " + self.end_pdb)
        lines.append("paratypecharmm on")
        for p in self.param_file:
            lines.append("parameters " + p)
        lines.append('set outputname ' + self.end_pdb + '\n')
        string = "\n".join(lines)
        return string

    def construct_box(self):
        lines = []
        lines.append("cellBasisVector1 " + str(self.x_box) + " 0 0")
        lines.append("cellBasisVector2 0 " + str(self.y_box) + " 0")
        lines.append("cellBasisVector3 0 0 " + str(self.z_box))

        string = "\n".join(lines)
        string += "\n"
        return string

    def construct_mini(self, num_iter=100):
        line = "minimize " + str(num_iter) + "\n"
        return line

    def construct_dyn(self, num_iter=100):
        line = "run " + str(num_iter) + "\n"
        return line
    def construct_restart(self,input_name):
        lines = []
        lines.append("set inputname "+ input_name)
        lines.append("""binCoordinates $inputname.coor;
binVelocities  $inputname.vel;
extendedSystem $inputname.xsc;""" )
        string = "\n".join(lines)
        string += "\n"
        return string

    def write_inputfile(self, filename, nsteps=100):
        file = open(filename, "w")
        file.write(self.construct_header())
        file.write(self.construct_constraints())
        file.write(self.construct_box())
        file.write(self.construct_mini(nsteps))
        file.write(self.construct_dyn(nsteps))
        file.close()
    def write_restraintfile(self, directory, atomicgroup,sele, spring=10.0):
        pdb = loos.PDB.fromAtomicGroup(atomicgroup.copy())
        spring = loos.GCoord(spring,0.,0.)
        zero = loos.GCoord(0.,0.,0.)
        for atom in pdb:
            atom.coords(spring)
             #print (atom.coords)
        heavy = loos.selectAtoms(pdb, sele)
        for thing in heavy:
            thing.coords(zero)
        pdb_file = open(os.path.join(directory, self.cons_k_filename), "w")
        pdb_file.write(str(pdb))
        #print('done')
        pdb_file.close()
    def construct_constraints(self):
        lines = [ "constraints on",
                  "selectConstraints on",
                  "selectConstrZ on",
                  "conskcol X"
                ]
        line = "consref " + self.start_pdb
        lines.append(line)

        line = "conskfile " + self.cons_k_filename
        lines.append(line)

        line = "\n".join(lines)
        line += "\n"
        return line

    def run_namd(self, inputfilename, outfilename):
        """
        Run namd on 4 processors, and report any failure to stderr
        """
        outfile = open(outfilename, "w")
        call = (str(self.command) + " " + str(inputfilename)).split()
        try:
            subprocess.check_call(call,stdout=outfile)
        except subprocess.CalledProcessError:
            sys.stderr.write("NAMD call failed, inp = %s, out = %s\n" %
                             (inputfilename, outfilename))
            sys.exit(-1)
    def MIN(self):

        return """restartfreq        1000;                # 500 steps = every 1ps
dcdfreq           1000;

dcdUnitCell        yes;


xstFreq           1000;

outputEnergies     1000;

outputTiming      1000;

# These are specified by CHARMM
exclude             scaled1-4


1-4scaling          1.0
switching            on
vdwForceSwitching   yes;


# You have some freedom choosing the cutoff
cutoff              12.0;              # may use smaller, maybe 10., with PME
switchdist          10.0;              # cutoff - 2.
                                       # switchdist - where you start to switch
                                       # cutoff - where you stop accounting for nonbond interactions.
                                       # correspondence in charmm:
                                       # (cutnb,ctofnb,ctonnb = pairlistdist,cutoff,switchdist)
pairlistdist        16.0;              # stores the all the pairs with in the distance it should be larger
                                       # than cutoff( + 2.)
stepspercycle       20;                # 20 redo pairlists every ten steps
pairlistsPerCycle    2;                # 2 is the default
                                       # cycle represents the number of steps between atom reassignments
                                       # this means every 20/2=10 steps the pairlist will be updated

# Integrator Parameters
timestep            1.0;
rigidBonds          all;
nonbondedFreq       1;
fullElectFrequency  1;

# PME (for full-system periodic electrostatics)
PME                yes;
PMEInterpOrder       6;                # interpolation order (spline order 6 in charmm)
PMEGridSpacing     1.0;                # maximum PME grid space / used to calculate grid size

wrapWater            on;               # wrap water to central cell
wrapAll              on;               # wrap other molecules too
"""
    def EQ(self):
        return """ restartfreq        1000;                # 500 steps = every 1ps
dcdfreq           1000;
dcdUnitCell        yes;                # the file will contain unit cell info in the style of
                                       # charmm dcd files. if yes, the dcd files will contain
                                       # unit cell information in the style of charmm DCD files.
xstFreq           1000;                # XSTFreq: control how often the extended systen configuration
                                       # will be appended to the XST file
outputEnergies     1000;                # 125 steps = every 0.25ps
                                       # The number of timesteps between each energy output of NAMD
outputTiming      1000;                # The number of timesteps between each timing output shows
                                       # time per step and time to completion

# These are specified by CHARMM
exclude             scaled1-4



1-4scaling          1.0
switching            on
vdwForceSwitching   yes;               # New option for force-based switching of vdW
                                       # if both switching and vdwForceSwitching are on CHARMM force
                                       # switching is used for vdW forces.

# You have some freedom choosing the cutoff
cutoff              12.0;              # may use smaller, maybe 10., with PME
switchdist          10.0;              # cutoff - 2.
                                       # switchdist - where you start to switch
                                       # cutoff - where you stop accounting for nonbond interactions.
                                       # correspondence in charmm:
                                       # (cutnb,ctofnb,ctonnb = pairlistdist,cutoff,switchdist)
pairlistdist        16.0;              # stores the all the pairs with in the distance it should be larger
                                       # than cutoff( + 2.)
stepspercycle       20;                # 20 redo pairlists every ten steps
pairlistsPerCycle    2;                # 2 is the default
                                       # cycle represents the number of steps between atom reassignments
                                       # this means every 20/2=10 steps the pairlist will be updated

# Integrator Parameters
rigidBonds          all;               # Bound constraint all bonds involving H are fixed in length
nonbondedFreq   1;                 # nonbonded forces every step
fullElectFrequency  1;                 # PME every step

# PME (for full-system periodic electrostatics)
PME                yes;
PMEInterpOrder       6;                # interpolation order (spline order 6 in charmm)
PMEGridSpacing     1.0;                # maximum PME grid space / used to calculate grid size

wrapWater            on;               # wrap water to central cell
wrapAll              on;               # wrap other molecules too


stochRescale on;
stochRescalePeriod .5;
stochRescaleTemp 310;
LangevinPistonTemp 310;



# Pressure and volume control
useGroupPressure       yes;            # use a hydrogen-group based pseudo-mol$
                                       # has less fluctuation, is needed for ri$
useFlexibleCell        yes;            # yes for anisotropic system like membr$
useConstantRatio       yes;            # keeps the ratio of the unit cell in t$

#pistion
LangevinPiston on;
LangevinPistonTarget 1.01325;
LangevinPistonPeriod 200;
LangevinPistonDecay 100;
"""

    def HEAT(self):
        return """
restartfreq        1000;                # 500 steps = every 1ps
dcdfreq           1000;
dcdUnitCell        yes;                # the file will contain unit cell info in the style of
                                       # charmm dcd files. if yes, the dcd files will contain
                                       # unit cell information in the style of charmm DCD files.
xstFreq           1000;                # XSTFreq: control how often the extended systen configuration
                                       # will be appended to the XST file
outputEnergies     1000;                # 125 steps = every 0.25ps
                                       # The number of timesteps between each energy output of NAMD
outputTiming      1000;                # The number of timesteps between each timing output shows
            # @endcond
                           # time per step and time to completion
# These are specified by CHARMM
exclude             scaled1-4          # non-bonded exclusion policy to use "none,1-2,1-3,1-4,or scaled1-4"
                                       # 1-2: all atoms pairs that are bonded are going to be ignored
                                       # 1-3: 3 consecutively bonded are excluded
                                       # scaled1-4: include all the 1-3, and modified 1-4 interactions
                                       # electrostatic scaled by 1-4scaling factor 1.0
                                       # vdW special 1-4 parameters in charmm parameter file.
1-4scaling          1.0
switching            on
vdwForceSwitching   yes;               # New option for force-based switching of vdW
                                       # if both switching and vdwForceSwitching are on CHARMM force
                                       # switching is used for vdW forces.

# You have some freedom choosing the cutoff
cutoff              12.0;              # may use smaller, maybe 10., with PME
switchdist          10.0;              # cutoff - 2.
                                       # switchdist - where you start to switch
                                       # cutoff - where you stop accounting for nonbond interactions.
                                       # correspondence in charmm:
                                       # (cutnb,ctofnb,ctonnb = pairlistdist,cutoff,switchdist)
pairlistdist        16.0;              # stores the all the pairs with in the distance it should be larger
                                       # than cutoff( + 2.)
stepspercycle       20;                # 20 redo pairlists every ten steps
pairlistsPerCycle    2;                # 2 is the default
                                       # cycle represents the number of steps between atom reassignments
                                       # this means every 20/2=10 steps the pairlist will be updated

# Integrator Parameters
timestep            1.0;               # fs/step
rigidBonds          all;               # Bound constraint all bonds involving H are fixed in length
nonbondedFreq       1;                 # nonbonded forces every step
fullElectFrequency  1;                 # PME every step

# PME (for full-system periodic electrostatics)
PME                yes;
PMEInterpOrder       6;                # interpolation order (spline order 6 in charmm)
PMEGridSpacing     1.0;                # maximum PME grid space / used to calculate grid size

wrapWater            on;               # wrap water to central cell
wrapAll              on;               # wrap other molecules too


stochRescale on;
stochRescalePeriod .5;

useFlexibleCell        yes;            # yes for anisotropic system like membra$
useConstantRatio       yes;            # keeps the ratio of the unit cell in th$
"""
