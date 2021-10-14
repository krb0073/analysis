#!/bin/bash
# command line args 

FILE_loc=$1 # this is for common 
if  [ $1 == "help" ] ;then 
	echo "args FILE_Loc PSF PDB run_to_start_from starting_time_step time_step "
	exit
fi 
loc=$(pwd)
PSF=$2
PDB=$3
RESTART=$4 
NEW_TIME_STEP=$5
time_step=$6

# move the blank common dir to your location 
cp -r ${FILE_loc}/common/ ${loc}
# once we have the file location set we can than sed into the files that items that we need 
sed -i "s/PSF/$PSF/g" common/template.conf
sed -i "s/PDB/$PDB/g" common/template.conf
sed -i "s/RESTART/${RESTART}.restart/g" common/template.conf
sed -i "s/NEW_TIME_STEP/$NEW_TIME_STEP/g" common/template.conf
sed -i "s/STEP/${time_step}/g" common/general.inp 
sed -i "s/PSF/$PSF/g" common/general.inp
sed -i "s/PDB/$PDB/g" common/general.inp
sed -i "s/RESTART/${RESTART}.dcd/g" common/get_spring_val.py
sed -i "s/PSF/$PSF/g" common/pbs_constraint.sh
# now we have to move the filed given above into common for this to work 
cp -t common/ $PSF $PDB ${RESTART}.* 
mv common/pbs_constraint.sh common/pbs.sh
echo "Done the Stuff we need to do"

