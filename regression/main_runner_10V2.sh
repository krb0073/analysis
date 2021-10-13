# @Author: Kyle Billings <kbillings>
# @Date:   2020-09-23T17:29:14-04:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: main_runner_10V2.sh
# @Last modified by:   kbillings
# @Last modified time: 2020-09-23T17:31:09-04:00
# @Author: Kyle Billings <kbillings>
# @Date:   2020-09-09T01:53:45-04:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: main_10_runner.sh
# @Last modified by:   kbillings
# @Last modified time: 2020-09-23T17:31:09-04:00
#!/usr/bin/env bash
# run this code in data all file ...yay
start=$1
seed=$start # command line for the random seed so each run will use the same seed
# from here we need to make a dir for thatb random seed
#rm -rf goodRangeDCD/RMSF_datasets_${seed}
#mkdir goodRangeDCD/RMSF_datasets_${seed}
cd goodRangeDCD/RMSF_datasets_${seed}
# randaimize file for the code
#python3.8 /media/bak12/Analysis/regression/LR_randomvalues.py $seed
#pwd
#echo "done random arrays for RSMF"
# this will make the file 'randvals_seed{}.csv' {} is the seed number
# need to format the  line in the code for the rmsf runs
#for line in $(seq 1 10); do
# get the line of the data
#head -n ${line}  randvals_seed${seed}.csv | tail -n 1 | tr ',' '\n' | sort -n > $line.dat
#done
# now we can run all the states RMSF stuff at once
#for i in bR K L M N O ; do
#bash /media/bak12/Analysis/regression/run_the_RMSF.sh $i $seed &
#done
#wait # waits until all is done
#echo "DONE random RMSF"
#for i in $(seq 1 10); do
#  set=$(ls *_set_${i}.*| sort) # collect all the sets sort by letters
#  for j in $set; do
#    grep "#" $j
#  done | awk '{print $2 , $4}'  > my_set_${i}.dat
#done
cd ../../
# make the name of the file to put them in
P="graphs_${seed}"
rm -rf $P
mkdir  $P
#python3.8 /media/bak12/Analysis/regression/LR.py $seed  "./${P}/" > $P/LR_code_${seed}.log &
python3.8 /media/bak12/Analysis/regression/LR_LOOCV_SMOLS.py $seed "./${P}/" > $P/LOOCV_OLS_${seed}.log &
python3.8 /media/bak12/Analysis/regression/LR_kfold_ridgeSklearn.py $seed  "./${P}/" > $P/Kfold_10_ridge_${seed}.log &
python3.8 /media/bak12/Analysis/regression/LR_Kfold_SMOLS.py $seed "./$P/" > $P/Kfold_10_OLS_${seed}.log &
#python3.8 /media/bak12/Analysis/regression/LR_LOOCV_ridge.py $seed "./$P/" > $P/LOOCV_Ridge_model_${seed}.log &
wait
