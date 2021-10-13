# @Author: Kyle Billings <kbillings>
# @Date:   2020-08-07T14:05:08-04:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: Main_LR_code.sh
# @Last modified by:   kbillings
# @Last modified time: 2020-08-20T14:20:24-04:00
#!/usr/bin/env bash
# run this code in data all file ...yay
start=$(date +%s)
seed=$1 # command line for the random seed so each run will use the same seed
# from here we need to make a dir for thatb random seed
rm -rf goodRangeDCD/RMSF_datasets_${seed}
mkdir goodRangeDCD/RMSF_datasets_${seed}
cd goodRangeDCD/RMSF_datasets_${seed}
# randaimize file for the code
python3.8 /media/bak12/Analysis/regression/LR_randomvalues.py $seed
pwd
echo "done random arrays for RSMF"
# this will make the file 'randvals_seed{}.csv' {} is the seed number
# need to format the  line in the code for the rmsf runs
for line in $(seq 1 10); do
# get the line of the data
head -n ${line}  randvals_seed${seed}.csv | tail -n 1 | tr ',' '\n' | sort -n > $line.dat
done
# now we can run all the states RMSF stuff at once
for i in bR K L M N O ; do
bash /media/bak12/Analysis/regression/run_the_RMSF.sh $i $seed &
done
wait # waits until all is done
echo "DONE random RMSF"
for i in $(seq 1 10); do
  set=$(ls *_set_${i}.*| sort) # collect all the sets sort by letters
  for j in $set; do
    grep "#" $j
  done | awk '{print $2 , $4}'  > my_set_${i}.dat
done
# with the right files made we can run the regressions with that seed
# move back to data all
cd ../../
end=$(date +%s)
tot=$(echo "$end - $start" | bc -l )
echo "Execution time of Prep was $tot seconds."
start=$(date +%s)
python3.8 /media/bak12/Analysis/regression/LR.py $seed # > test_seed_${seed}_run.log
end=$(date +%s)
tot=$(echo "$end - $start" | bc -l )
echo "OLS normal style took $tot "
start=$(date +%s)
python3.8 /media/bak12/Analysis/regression/LR_Kfold_SMOLS.py $seed
end=$(date +%s)
tot=$(echo "$end - $start" | bc -l )
echo "Execution time  of OLS K fold was $tot seconds."
echo "starting the ridige version of the regression"
start=$(date +%s)
python3.8 /media/bak12/Analysis/regression/LR_ridgeoneSet.py $seed
end=$(date +%s)
tot=$(echo "$end - $start" | bc -l )
echo " ridge version ran in ${tot} seconds"
echo "ridge Kfold beingin exuctuion"
#start=$(date +%s)
#python3.8 /media/bak12/Analysis/regression/LR_kfold_ridgeSklearn.py $seed
#end=$(date +%s)
#tot=$(echo "$end - $start" | bc -l )
#echo " the K fold ridge model took $tot"
echo DONE
exit
