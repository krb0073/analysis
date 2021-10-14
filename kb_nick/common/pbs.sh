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

module purge
# number of total cores 
N=76
ppn=19 # number of workers per nodes 
# modules needed 
module load  lang/intel/2018 libs/fftw/3.3.9_intel18  parallel/openmpi/3.1.4_intel18_tm
# print date to the output 
date

# loos stuff for things i do 
module load loos/latest
module load mertz_conda/latest vmd/1.9.3

# activate conda 
conda activate loos

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

