# @Author: Kyle Billings <kbillings>
# @Date:   2020-09-09T11:28:53-04:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: ONE_set_runner.sh
# @Last modified by:   kbillings
# @Last modified time: 2020-09-09T11:35:17-04:00
#!/usr/bin/env bash
run_name="MD_only"
path="MD_only"
rm -rf $run_name
mkdir $run_name
python3.8 /media/bak12/Analysis/regression/one_set_LOOCV_ridge.py $run_name "./${path}/" > MD_ONLY_LOOCV_RIDGE.log &
python3.8 /media/bak12/Analysis/regression/one_Set_OLS_MD.py  $run_name "./${path}/" > MD_ONLY_OLS_noCV.log &
python3.8 /media/bak12/Analysis/regression/One_set_with_loocv.py  $run_name "./${path}/" > MD_ONLY_OLS_LOOCV.log &
python3.8 /media/bak12/Analysis/regression/oneset_ridge.py  $run_name "./${path}/" > MD_ONLY_Ridge_noCV.log &
