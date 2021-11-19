# thiss attachedb the ret to the portein
package require psfgen
# make the segments
foreach S  {  189 185 182} {
	for {set i 1} {$i <= 3} {incr i} {
		resetpsf
		psfcontext reset
		set my_path "/media/bak11/KylesStuff/newer_mutants/cis/$S/$i"
		# now load in the final pdb using the thing
		mol load psf ${my_path}/final.psf pdb ${my_path}/final.pdb
		set A [atomselect top "not (protein or resname RET RTNH)"]
		$A writepsf ${my_path}/temp.psf
		$A writepdb ${my_path}/temp.pdb
		mol delete all
		# now we can make the fixed_pdb and psf
		readpsf ${my_path}/temp.psf
		coordpdb ${my_path}/temp.pdb
		readpsf ${my_path}/regen_ret.psf
		coordpdb ${my_path}/regen_ret.pdb
		writepsf ${my_path}/final_2.psf
		writepdb ${my_path}/final_2.pdb
	}
}

exit
