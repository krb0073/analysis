# @Author: Kyle Billings <kbillings>
# @Date:   2021-03-26T14:20:30-04:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: job_chains.sh
# @Last modified by:   kbillings
# @Last modified time: 2021-03-26T14:22:44-04:00
#!/bin/bash


if [ $# -eq 0 ]; then
     echo "sh $0 benchamrk|info|chain|restart|help steps time queue"
     exit
elif [ $1 = '-h' ] || [ $1 = 'help' ]; then
     echo "This script allows the chaining of NAMD jobs through the Torque submission system (Spruce Knob and Thorny Flat)"
     echo "Several scripts join these for the ease of creating your pbs and config files."
     echo "Modify the scripts in common/ to reflect your parameters, add the files you'll be using to stat from, and add your forcefield files to the toppar/ directory."
     echo "Here's a more detailed breakdown of the command line arguements and how to use this script:"
     echo ""
     echo "First arguement:"
     echo "    benchmark option - runs benchmarking based on the starting files outlined in common/template.conf using the number of nodes in common/pbs.sh"
     echo "    info option - uses the benchmarking info to inform you of the number of jobs that will be in the chain and the number of steps per job"
     echo "    chain option - starts running chained jobs based on the command line arguements and benchmarks"
     echo "    restart option - find the trajecotry within the runs dir that haas the last updated trajectory to restart the chain after a crash"
     echo "steps - the total number of steps to be run over the chained jobs"
     echo "time - the wall time in hours for each job"
     echo "queue - the queue for the job"
     exit
fi

run=$1; total_steps=$2; wall_hours=$3; queue=$4

COMMON=$(pwd)/common
WORK_DIR=$(pwd)
# Conditional to see if we are benchmarking or running
if [ $run = 'benchmark' ]; then
     mkdir -p benchmark
     # Make a benchmarking config
     cp common/template.conf benchmark/run.conf
     # First time step is 0
     sed -i 's|insert_firsttimestep_here|0|g' benchmark/run.conf
     # # of steps
     sed -i 's|insert_steps_here|10000|g' benchmark/run.conf
     # Common location
     sed -i "s|insert_common_here|${COMMON}|g" benchmark/run.conf
     # Number of the job
     sed -i "s|insert_jobnum_here|1|g" benchmark/run.conf

     # Make a benchmarking pbs
     cp common/pbs.sh benchmark/pbs.sh
     # Insert queue
     sed -i 's|insert_queue_here|standby|g' benchmark/pbs.sh
     # Insert walltime
     sed -i 's|insert_walltime_here|1|g' benchamrk/pbs.sh
     # submit pbs
     cd benchmark
     qsub pbs.sh
     cd $WORK_DIR
     echo "Benchmark submitted!"
     exit

elif [ $run = 'info' ]; then
     if ! [ -f benchmark/benchamrk.log ]; then
          echo "Run the benchamrk first!"
          exit
     else
          # Get seconds per step from benchmark log
          seconds_per_step=$(grep Benchmark benchamrk/run.log |cut -f 6-6 -d ' ')
          # Convert to steps per hour
          steps_per_hour=$(echo "print(3600.0/float($seconds_per_step))" |python)
          # Determine steps per job
          steps_per_job=$(echo "import math; print(math.floor(float($steps_per_hour)*float($wall_hours)/1000.0)*1000)" |python)
          # Determine number of chains
          number_of_jobs=$(echo "import math; print(math.ceiling(float($total_steps)/float($steps_per_job)))" |python)

          echo "With the present parameters, the maximum steps completed per job is $stepspj steps."
          echo "This means the total job of $total_steps steps will take $number_of_jobs jobs to be completed."
          echo "Make sure the frequency of the output in the common files is appropriate for the number of steps in each job."
          echo "DCD and restart frequencies should be equivalent."
          exit
     fi
elif [ $run = 'chain' ]; then

     if ! [ -f benchmark/benchamrk.log ]; then
          echo "Run the benchamrk first!"
          exit
     else
          # Get seconds per step from benchmark log
          seconds_per_step=$(grep Benchmark benchamrk/run.log |cut -f 6-6 -d ' ')
          # Convert to steps per hour
          steps_per_hour=$(echo "print(3600.0/float($seconds_per_step))" |python)
          # Determine steps per job
          steps_per_job=$(echo "import math; print(math.floor(float($steps_per_hour)*float($wall_hours)/1000.0)*1000)" |python)
          # Determine number of chains
          number_of_jobs=$(echo "import math; print(math.ceiling(float($total_steps)/float($steps_per_job)))" |python)

          # Loop to create pbs and submit chained jobs
          mkdir -p runs
          for n in $(seq 1 $number_of_jobs); do
               mkdir -p runs/$n

               # Make a run pbs
               cp common/pbs.sh runs/${n}/pbs.sh
               # Insert queue
               sed -i "s|insert_queue_here|${queue}|g" runs/${n}/pbs.sh
               # Insert walltime
               sed -i "s|insert_time_here|${wall_hours}|g" runs/${n}/pbs.sh

               # Make a run config
               cp common/template.conf runs/${n}/run.conf
               # # of steps
               sed -i "s|insert_steps_here|${steps_per_job}|g" runs/${n}/run.conf
               # Common location
               sed -i "s|insert_common_here|${COMMON}|g" runs/${n}/run.conf
               # Number of the job
               sed -i "s|insert_jobnum_here|${n}|g" runs/${n}/run.conf

          done
          # Submit the first job in the chain
          cd runs/1/
          echo -n "Submitting job 1 "
          one=$(qsub pbs.sh)
          cd $WORK_DIR
          # Add the job id to the list
          echo $one >> runs/job_ids.txt

          for n in $(seq 2 $number_of_jobs); do
               cd runs/$n/
               echo -n "$n "
               # Submit the subsequent jobs depending on the previous one
               two=$(qsub -W depend=afterany:$one pbs.sh)
               cd $WORK_DIR
               one=$two
               # Add the job id to the list
               echo $one >> runs/job_ids.txt
          done

          echo "Done!"
     fi

elif [ $run = 'restart' ]; then
	if ! [ -f benchmark/benchamrk.log ]; then
          echo "Run the benchamrk first!"
          exit
	elif ![ -d runs/ ] ;then
		echo "Run Chain frist!" 
		exit 
	else
	  # this will be the same as chain but we have to figure out where we neded 
          # Get seconds per step from benchmark log
          seconds_per_step=$(grep Benchmark benchamrk/run.log |cut -f 6-6 -d ' ')
          # Convert to steps per hour
          steps_per_hour=$(echo "print(3600.0/float($seconds_per_step))" |python)
          # Determine steps per job
          steps_per_job=$(echo "import math; print(math.floor(float($steps_per_hour)*float($wall_hours)/1000.0)*1000)" |python)
          # Determine number of chains
          number_of_jobs=$(echo "import math; print(math.ceiling(float($total_steps)/float($steps_per_job)))" |python)	
	# last good run done for the procject 
	  last_good_run=$(ls -1v runs/*/*.dcd | sed 's/\// /g'| tail -n 1) # this gets the dir number that actaully worked
	 # index number of the run that failes is one more than the ones that ran 
	   failed_starting_index=$(( $last_good_run + 1 )) 
	# rm all dir that did not run right by force 
	  for i in $(seq $failed_starting_index  $number_of_jobs); do rm -rf runs/${i} ;done ; echo "runs that failed were $(seq $failed_starting_index	 $number_of_jobs) ... removed"
	for n in $(seq $failed_starting_index $number_of_jobs); do
               mkdir -p runs/$n
               # Make a run pbs
               cp common/pbs.sh runs/${n}/pbs.sh
               # Insert queue
               sed -i "s|insert_queue_here|${queue}|g" runs/${n}/pbs.sh
               # Insert walltime
               sed -i "s|insert_time_here|${wall_hours}|g" runs/${n}/pbs.sh
               # Make a run config
               cp common/template.conf runs/${n}/run.conf
               # # of steps
               sed -i "s|insert_steps_here|${steps_per_job}|g" runs/${n}/run.conf
               # Common location
               sed -i "s|insert_common_here|${COMMON}|g" runs/${n}/run.conf
               # Number of the job
               sed -i "s|insert_jobnum_here|${n}|g" runs/${n}/run.conf

          done
         cd runs/$failed_starting_index/
          echo -n "Submitting job $failed_starting_index "
          one=$(qsub pbs.sh)
          cd $WORK_DIR
          # Add the job id to the list
          echo $one >> runs/job_ids.txt
	  next_run=$(($failed_starting_index + 1 )) # add one to that index
	  for n in $(seq $next_run $number_of_jobs); do
               cd runs/$n/
               echo -n "$n "
               # Submit the subsequent jobs depending on the previous one
               two=$(qsub -W depend=afterany:$one pbs.sh)
               cd $WORK_DIR
               one=$two
               # Add the job id to the list
               echo $one >> runs/job_ids.txt
          done

          echo "Done!"
     fi


	
else
     echo "Input makes no sense!"
     exit
fi
