# VMD code to get the only the protein from each mutant system 
foreach S {182 189 185} { 
	for {set i 1} { $i <= 3} {incr i} {
		set my_path "/media/bak22/BR_Photo/BR/newer_mutants/alltrans/$S/$i"
		mol load psf ${my_path}/final.psf pdb ${my_path}/final.pdb
		set A [atomselect top "protein or resname RET RTNH"]
		$A writepsf ${my_path}/pro_only.psf 
		$A writepdb ${my_path}/pro_only.pdb
		mol delete top 
	}
}
