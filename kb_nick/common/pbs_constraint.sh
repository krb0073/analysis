# @Date:   2021-03-23T13:13:20-04:00
# @Last modified time: 2021-03-23T13:32:03-04:00



#!/bin/bash
# Change the things in the square brackets []
# Don't change other stuff :)


#PBS -q insert_queue_here
#PBS -m ae
#PBS -M krb0073@mix.wvu.edu
#PBS -N insert_name_here
#PBS -l nodes=2:ppn=40
#PBS -l walltime=insert_walltime_here:00:00
num=insert_run_num


cd $PBS_O_WORKDIR
working=$PBS_O_WORKDIR
module purge
# number of total cores 
N=76
ppn=19 # number of workers per nodes 
# modules needed 
module load  lang/intel/2018 libs/fftw/3.3.9_intel18 parallel/openmpi/3.1.4_intel18_tm loos/latest 
# print date to the output 
date

# loos stuff for things i do 
module load loos/latest
module load mertz_conda/latest 

# activate conda 
conda activate loos

# loos stuff to get the contraint going 
# im getting about 2 ns per run therfore 13 at 25
# 3 runs at 10 , 5 ,1 3 * 3 = 9
# total is about 22 runs with bacbone contraint 

current_run=$(echo "allTrans_w189f_sys_3_2" | sed 's/_/\n/g' | tail -n 1)
echo "$current_run"
c_path="../../common"
if [ $current_run == "here" ] ;then current_run=0 ;c_path="../common" ;fi
run_num=$("print(int($current_run))" | python )

info=$(python $c_path/get_spring_val.py insert_run_num) # this gives out the spring 
echo "$info"
sp=$(echo $info |awk '{print $1}')
traj=$(echo $info | awk '{print $2}')

struct="$c_path/final.psf"
echo "$traj $sp $struct"
python /users/krb0073/EQ_run/frame_to_pdb_python.py $struct $traj > last.pdb
python ~/EQ_run/python_constraint_1.py $struct last.pdb $sp

# namd excauateable 
MD_NAMD=/scratch/jbmertz/binaries/NAMD_2.14_Source/Linux-x86_64-icc-smp/
NODES=`pwd`/.nodelist
cat $PBS_NODEFILE | perl -e 'while(<>) { chomp; $node{$_}++; } $num = keys %node; print "group main\n"; for (keys %node) { print "host $_ ++cpus $node{$_}\n"; }' > $NODES

### run your executable program
procs=`echo "$N/$ppn" | bc`
procspernode=`echo $procs/2 | bc`
echo "procs: $procs with $procspernode on each node"
job_num=$(basename $(pwd))
prev_job=$(($job_num - 1))

# stuff for the chain 
first_time_step=$(grep -v '#' ../$prev_job/run.restart.xsc |awk '{print $1}')
sed -i "s|insert_firsttimestep_here|$first_time_step|g" run.conf

mpirun --map-by ppr:$procspernode:node ${MD_NAMD}namd2 +setcpuaffinity +ppn${ppn} run.conf > run.log
rm $NODES


