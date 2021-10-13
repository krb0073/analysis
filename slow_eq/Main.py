# @Author: Kyle Billings <kbillings>
# @Date:   2020-12-07T10:53:42-05:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: Main.py
# @Last modified by:   kbillings
# @Last modified time: 2021-08-11T21:34:55-04:00

import subprocess
import sys
import os.path
import loos
import loos.pyloos
import sys
import configparser
import shutil
sys.path.append('/media/bak12/Analysis/RUN_EQ_PYTHON/')
from NAMD_Kyle import *
from Config_get import *
Config_file =sys.argv[1]
TYPE = sys.argv[2]
config = Read_config(Config_file) # read config
system = loos.createSystem(config.psf)
PDB = loos.PDB(config.pdb).periodicBox()
#def lastFramePDB(DCD):
 #  """ this function will take a given DCD and then grab the last frame of
#    The Traj"""
#    file = open('temp.pdb','w')
#    A = 'frame2pdb ' + '-- ' + sim.psf_file +' ' + DCD  + ' -1'
#    subprocess.Popen(A, stdout=file, shell=True,text=True)
#    file.close()
main_dir = 'Sim_prep' # dir name for the run\
def lastFramePDB(DCD):
    file = open('temp.pdb', 'w')
    Traj = loos.pyloos.Trajectory(DCD ,system)
    Fnum = len(Traj) -1
    last_frame = Traj.readFrame(Fnum)
    pdb = loos.PDB.fromAtomicGroup(last_frame.copy())
    file.write(str(pdb))
    file.close()

def PBCWhoops(failed_run_num,sim_type,dir_loc,sim,sp=20):
    ans ='crash'
    run_type = sim_type.lower()
    last_run = run_type + f".{failed_run_num}"
    run_num = 1
    old = last_run
    # starting the While loop
    while ans == 'crash':
        new_run = run_type + f".{failed_run_num}_{run_num}"
        # now we need to grab the frist and last frame for use
        file = open(f"{last_run}.xst")
        times = []
        for line in file.readlines():
            if line.startswith("#") or line.isspace() or len(line) == 0:
                continue
            else:
                times.append(line.split()[0])

        first_time = times[0]
        last_time = times[-1]
        # build the start of file that is the same for both types
        DCD = old + ".dcd"
        lastFramePDB(DCD)
        ca_sel = '!(' + config.ca + ')'
        sim.write_restraintfile(dir_loc,system,ca_sel,spring=sp)
        sim.start_pdb = 'temp.pdb'
        sim.end_pdb = new_run
        s = sim.construct_header()
        s += f"firsttimestep      {last_time};" + "\n"
        s += sim.construct_restart(old + ".restart")
        # now we will have to sperate by type for a cuple things
        if run_type == 'heat':
            # find the temp that failed on
            final_temp = 310
            all_temp = subprocess.Popen(f'grep stochRescaleTemp {old}.log',shell=True,stdout=subprocess.PIPE,text=True)
            x = [x for x in all_temp.stdout.readlines()][-1]
            temp = x.split()[-1]
            s += sim.HEAT()
            s += sim.construct_constraints()
            if failed_run == 3:
                final_temp = 250
                s += """
            # Pressure and volume control
            useGroupPressure       yes;            # use a hydrogen-group based pseudo-mol$
                                                   # has less fluctuation, is needed for ri$


            # Pistion Parameters
            LangevinPiston on;
            LangevinPistonTarget 1.01325;
            LangevinPistonPeriod 400;
            LangevinPistonDecay 200;\n"""
                s += "for {set i  " + str(temp) +  " } { $i <= "  + str(final_temp)  + "} {incr i} {"
                s += """
            stochRescaleTemp $i;


            LangevinPistonTemp $i;

            run 2000;
            }"""
            elif failed_run == 4:
                final_temp = 310
                s += """
            # Pressure and volume control
            useGroupPressure       yes;            # use a hydrogen-group based pseudo-mol$
                                                   # has less fluctuation, is needed for ri$


            # Pistion Parameters
            LangevinPiston on;
            LangevinPistonTarget 1.01325;
            LangevinPistonPeriod 200;
            LangevinPistonDecay 100;\n"""
                s += "for {set i  " + str(temp) +  " } { $i <= "  + str(final_temp)  + "} {incr i} {"
                s += """
            stochRescaleTemp $i;


            LangevinPistonTemp $i;

            run 2000;
            }

            """

            # now restart the thing
        elif run_type == "eq":
            step = 2
            total_steps = 1000000
            if failed_run_num == 1:
                step = 1
                total_steps = 500000
            # compute total time left
            time_left = int(total_steps) - (int(last_time) - int(first_time))# number of steps ran
            s += sim.EQ()
            s += f"timestep            {step}; "
            s += sim.construct_constraints()
            s += sim.construct_dyn(num_iter=time_left)#time_left) # ns
        file = open(f'{new_run}.inp','w')
        file.write(s)
        file.close()
        ans = PBC_crash(f'{new_run}.inp',f'{new_run}.log')
        A = run_num
        run_num += 1
    return f"{failed_run_num}_{run_num}" # this will be the start of the old run
 # start EQ
def startUp():
    """ creates the dir or removes then maked the dir if the dir is already their"""
    try:
        os.mkdir(main_dir)
    except:
        print('Sim_perp is already a dir removing to start over')
        shutil.rmtree(main_dir)
        os.mkdir(main_dir)
    os.chdir(main_dir)
# if they pass the checks in the read config we cand assume the file will be okay
system = loos.createSystem(config.psf)
# the pdb will have perodic box info so lets get that too
box = loos.PDB(config.pdb).periodicBox()## a pbc
if box.x() == 99999: # we have to build a guess to what the box will be
# 99999 is the defult for no box
    box_min , box_max = loos.PDB(config.pdb).boundingBox()
    print(box_min)
    # Making the start box size
    X = abs(box_min.x()) + abs(box_max.x())
    Y = abs(box_min.y()) + abs(box_max.y())
    Z = abs(box_min.z()) + abs(box_max.z())
    box = loos.GCoord(X,Y,Z)
    # note this will be a bit big
def runMin():
    """ Runs the Min process from JOe after checking the the dir is right"""
    # checking that the dir starting is the right one
    loc = os.getcwd()[-8:]
    if loc != main_dir:
        sys.stderr.write("Min will make the starting dir for you")
    # builbing the dir and entering
    try:
        os.mkdir('Min')
        os.chdir('Min')
    except:
        shutil.rmtree('Min')
        os.mkdir('Min')
        os.chdir('Min')
    # now we will have to build the config file for the Min one
    # intialize the NAMD_KB class
    sim = NAMD_KB(config.psf,config.pdb,'min.1',config.parameters,box,config.namd_binary)
    s = sim.construct_header()
    s += "firsttimestep      0;" + "\n"
    s += "temperature        25" + "\n"
    s += sim.construct_box()
    s += sim.MIN()
    # get the min location
    MIN_loc = os.getcwd()
    sim.write_restraintfile(MIN_loc, system, config.salt_sel + " ||" + config.water_sel,spring=500)
    s += sim.construct_constraints()
    s += sim.construct_mini(num_iter=20000)
    file = open('min.1.inp','w')
    file.write(s)
    file.close()
    sim.run_namd('min.1.inp','min.1.log')
    # now we can do min 2- 5
    not_backone = config.protein_sel + "&&"+"!(" + config.backbone + ")"
    not_ca = config.protein_sel + "&&"+"!(" + config.ca + ")"
    sel_list = [config.lipid_sel,config.protein_sel,not_backone,not_ca]
    for min_run , sel in zip(range(2,6),sel_list):
        print(min_run)
        last_run = min_run -1
        out_name = f"min.{min_run}"
        DCD = f'min.{last_run}.dcd'
        lastFramePDB(DCD)
        sim.start_pdb = 'temp.pdb'
        sp = 500
        nstep = 20000
        # getting the last time step
        file = open(f"min.{last_run}.restart.xsc",'r')
        for line in file.readlines():
            if line.startswith("#") or line.isspace() or len(line) == 0:
                continue
            else:
                last_frame = line.split()[0]
        file.close()
        if min_run > 3:
            nstep = 10000
            sp = 20
        sim.end_pdb = out_name
        s = sim.construct_header()
        s += f"firsttimestep      {last_frame}" + "\n"
        s += sim.construct_restart(f"min.{last_run}")
        s += sim.MIN()
        sim.write_restraintfile(MIN_loc,system,sel,spring=sp)
        s += sim.construct_constraints()
        s += sim.construct_mini(num_iter=nstep)
        file = open(f'min.{min_run}.inp','w')
        file.write(s)
        file.close()
        sim.run_namd(f'min.{min_run}.inp',f'min.{min_run}.log')
    # there is no constraint on the last min function so we dont have to do constraint
    min_run = 6
    last_run = 5
    file = open(f"min.5.restart.xsc",'r')
    for line in file.readlines():
        if line.startswith("#") or line.isspace() or len(line) == 0:
            continue
        else:
            last_frame = line.split()[0]
    file.close()
    sim.end_pdb = 'min.6'
    DCD = 'min.5.dcd'
    lastFramePDB(DCD)
    sim.start_pdb = 'temp.pdb'
    s = sim.construct_header()
    s += f"firsttimestep      {last_frame}" + "\n"
    s += sim.construct_restart(f"min.{last_run}")
    s += sim.MIN()
    s += sim.construct_mini(num_iter=10000)
    file = open(f'min.{min_run}.inp','w')
    file.write(s)
    file.close()
    sim.run_namd(f'min.{min_run}.inp',f'min.{min_run}.log')
    os.chdir('../')
def PBC_crash(inputfilename,outfilename,sim):
    """ when the system crash due to PBC error this fuction will return 'crash'
    instead of exiting out of the program. Aloows us to restart"""
    outfile = open(outfilename, "w")
    ans ='ran'
    call = (str(sim.command) + " " + str(inputfilename)).split()
    try:
            subprocess.check_call( call,stdout=outfile)
    except subprocess.CalledProcessError:
        sys.stderr.write("NAMD call failed, inp = %s, out = %s\n" %(inputfilename, outfilename))
            # checking if the call is becasue of the pbc crash
        pros = subprocess.Popen(f'grep FATAL {outfilename}',shell=True,stdout=subprocess.PIPE,text=True)
        line = pros.stdout.readline().split()
        if 'Periodic' in line:
            print("the Reason the sim failed was PBC...restarting")
            ans ='crash'
        else:
            sys.exit(-1)
    return ans

def runHeat():
    """ Runs the heat process of the sim making sure that the dir is right.
    Once pressure is on We switch to PBC_crash to run namd. Need to make the while
    loop better / a function """
    # testing heat stuff
    loc = os.getcwd()[-8:]
    if loc != main_dir:
        sys.stderr.write("Must launch in the Sim_prep dir ...exiting")
        sys.exit(1)
        # builbing the dir and entering
    try:
        os.mkdir('Heat')
        os.chdir('Heat')
    except:
        shutil.rmtree('Heat')
        os.mkdir('Heat')
        os.chdir('Heat')
    sim = NAMD_KB(config.psf,config.pdb,'heat.1',config.parameters,box,config.namd_binary)
    final_min = '../Min/min.6'
    DCD = '../Min/min.6.dcd'
    lastFramePDB(DCD)
    sim.start_pdb = 'temp.pdb'
    file = open(final_min + '.restart.xsc', 'r')
    for line in file.readlines():
        if line.startswith("#") or line.isspace() or len(line) == 0:
            continue
        else:
            last_frame = line.split()[0]
    file.close()
    # heat has no contraints
    s = sim.construct_header()
    s += f"firsttimestep      {last_frame};" + "\n"
    s += sim.construct_restart(final_min)
    s += sim.HEAT()
    HEAT_loc = os.getcwd()
    s += """for {set i 26} { $i <= 50 } {incr i} {
    stochRescaleTemp $i;

    run 2000;
    }

    """
    file = open('heat.1.inp','w')
    file.write(s)
    file.close()
    sim.run_namd('heat.1.inp','heat.1.log') # run for 50ps
    # all of the rest of the things we have will ahve CA with 20 kcal restraint ( only need one file)
    # all of the rest of the things we have will ahve CA with 20 kcal restraint ( only need one file)
    sp = 20
    ca_sel = '!(' + config.ca + ')'
    sim.write_restraintfile(HEAT_loc,system,ca_sel,spring=sp)
    # heat 2 no pistion
    file = open( 'heat.1.restart.xsc', 'r')
    for line in file.readlines():
        if line.startswith("#") or line.isspace() or len(line) == 0:
            continue
        else:
            last_frame = line.split()[0]
    file.close()
    sim.end_pdb = 'heat.2'
    DCD = 'heat.1.dcd'
    lastFramePDB(DCD)
    sim.start_pdb = 'temp.pdb'
    s = sim.construct_header()
    s += f"firsttimestep      {last_frame};" + "\n"
    s += sim.construct_restart('heat.1')
    s += sim.HEAT()
    s += sim.construct_constraints()
    s += """for {set i 51} { $i <= 200 } {incr i} {

    stochRescaleTemp $i;

    run 1000;
    }

    """
    file = open('heat.2.inp','w')
    file.write(s)
    file.close()
    sim.run_namd('heat.2.inp','heat.2.log')
    # Heat 3 and four will be trated diffetnet because the pistion
    file = open( 'heat.2.restart.xsc', 'r')
    for line in file.readlines():
        if line.startswith("#") or line.isspace() or len(line) == 0:
            continue
        else:
            last_frame = line.split()[0]
    file.close()
    DCD = 'heat.2.dcd'
    lastFramePDB(DCD)
    sim.start_pdb = 'temp.pdb'
    sim.end_pdb = 'heat.3'
    s = sim.construct_header()
    s += f"firsttimestep      {last_frame};" + "\n"
    s += sim.construct_restart('heat.2')
    s += sim.HEAT()
    s += sim.construct_constraints()
    s += """
    # Pressure and volume control
    useGroupPressure       yes;            # use a hydrogen-group based pseudo-mol$
                                           # has less fluctuation, is needed for ri$


    # Pistion Parameters
    LangevinPiston on;
    LangevinPistonTarget 1.01325;
    LangevinPistonPeriod 400;
    LangevinPistonDecay 200;

    for {set i 201} { $i <= 250 } {incr i} {
    stochRescaleTemp $i;


    LangevinPistonTemp $i;

    run 2000;
    }

    """
    file = open('heat.3.inp','w')
    file.write(s)
    file.close()
    ans = PBC_crash('heat.3.inp','heat.3.log',sim)
    old = 3
    # this will run if the ans to PBC drash is crash
    if ans == 'crash':
        old= PBCWhoops(3,'Heat',HEAT_loc,sim,sp=sp)
    file = open( f'heat.{old}.restart.xsc', 'r')
    for line in file.readlines():
        if line.startswith("#") or line.isspace() or len(line) == 0:
            continue
        else:
            last_frame = line.split()[0]
    file.close()
    DCD = f'heat.{old}.dcd'
    lastFramePDB(DCD)
    sim.start_pdb = 'temp.pdb'
    sim.end_pdb = 'heat.4'
    s = sim.construct_header()
    s += f"firsttimestep      {last_frame};" + "\n"
    s += sim.construct_restart(f'heat.{old}')
    s += sim.HEAT()
    s += sim.construct_constraints()
    s += """# Pressure and volume control
    useGroupPressure       yes;            # use a hydrogen-group based pseudo-mol$


    # Pistion Parameters
    LangevinPiston on;
    LangevinPistonTarget 1.01325;
    LangevinPistonPeriod 200;
    LangevinPistonDecay 100;


    for {set i 251} { $i <= 310 } {incr i} {
    stochRescaleTemp $i;

    LangevinPistonTemp $i;

    run 2000;
    }
    """
    file = open('heat.4.inp','w')
    file.write(s)
    file.close()
    ans = PBC_crash('heat.4.inp','heat.4.log',sim)
    # this will run if the ans to PBC drash is crash
    # this will run if the ans to PBC drash is crash
    if ans == 'crash':
        New = PBCWhoops(1,'EQ',EQ_loc,sim,sp=sp)
    os.chdir('../')

def runEQ():
    loc = os.getcwd()[-8:]
    if loc != main_dir:
        sys.stderr.write("Must launch in the Sim_prep dir ...exiting")
        sys.exit(1)
        # builbing the dir and entering
    try:
        os.mkdir('Eq')
        os.chdir('Eq')
    except:
        shutil.rmtree('Eq')
        os.mkdir('Eq')
        os.chdir('Eq')
        heat_dir = '../Heat/'
    # grabing the number og the last run of heat for the sim
    heat_dir = '../Heat/'
    HH = subprocess.run(f'ls -t {heat_dir}*.dcd',shell=True, stdout=subprocess.PIPE,text=True)
    Last_run = HH.stdout.split("\n")[0].split("/")[2].split(".")[1]
    final_heat = f'../Heat/heat.{Last_run}'
    # build sim obj
    sim = NAMD_KB(config.psf,config.pdb,'eq.1',config.parameters,box,config.namd_binary)
    sp_list = [10,5,1] # list of spring contatns to be used later
    DCD = f'../Heat/heat.{Last_run}.dcd'
    lastFramePDB(DCD)
    sim.start_pdb = 'temp.pdb'
    file = open(final_heat + '.restart.xsc', 'r')
    for line in file.readlines():
        if line.startswith("#") or line.isspace() or len(line) == 0:
            continue
        else:
            last_frame = line.split()[0]
    file.close()
    s = sim.construct_header()
    s += f"firsttimestep      {last_frame};" + "\n"
    s += sim.construct_restart(final_heat)
    s += sim.EQ()
    EQ_loc = os.getcwd()
    s += "timestep            1.0; "
    ca_sel = '!(' + config.ca + ')'
    sp = 20
    sim.write_restraintfile(EQ_loc,system,ca_sel,spring=20)
    s += sim.construct_constraints()
    s += sim.construct_dyn(num_iter=1000000) # ns
    file = open('eq.1.inp','w')
    file.write(s)
    file.close()
    ans = PBC_crash('eq.1.inp','eq.1.log',sim)
    # to make it go into the while loos i nees to say that it failed
    old = 1
    if ans == 'crash':
          old = PBCWhoops(1,'EQ',EQ_loc,sim,sp=sp)
    # doing 2 -4
    for eq , sp in zip(range(2,5),sp_list):
        print(f'eq {eq}')
        DCD = f'eq.{old}.dcd'
        lastFramePDB(DCD)
        sim.start_pdb = 'temp.pdb'
        sim.end_pdb = 'eq.' + str(eq)
        file = open(  f'eq.{old}.restart.xsc', 'r')
        for line in file.readlines():
            if line.startswith("#") or line.isspace() or len(line) == 0:
                continue
            else:
                last_frame = line.split()[0]
        file.close()
        s = sim.construct_header()
        s += f"firsttimestep      {last_frame};" + "\n"
        s += sim.construct_restart(f'eq.{old}')
        s += sim.EQ()
        s += "timestep            1.0; "
        ca_sel = '!(' + config.ca + ')'
        sim.write_restraintfile(EQ_loc,system,ca_sel,spring=sp)
        s += sim.construct_constraints()
        s += sim.construct_dyn(num_iter=1000000) # ns
        file = open(f'eq.{eq}.inp','w')
        file.write(s)
        file.close()
        ans = PBC_crash(f'eq.{eq}.inp',f'eq.{eq}.log',sim)
        old = eq
        if ans == 'crash':
            old = PBCWhoops(eq,'EQ',EQ_loc,sim,sp=sp)
    os.chdir('../')
#############################################################################
if TYPE == 'all':
    startUp()
    runMin()
    runHeat()
    runEQ()
elif TYPE == 'min':
    startUp()
    runMin()
elif TYPE == 'heat':
    runHeat()
elif TYPE == 'eq':
    runEQ()
else:
    print('the sting used for type is not a vvaild option please use all,min,heat,or eq')
