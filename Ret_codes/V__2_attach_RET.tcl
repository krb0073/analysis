# thiss attachedb the ret to the portein
package require psfgen
topology /media/bak11/toppar/top_all36_prot.rtf
topology /media/bak11/toppar/retinal-pro.str
pdbalias residue HIS HSE
pdbalias atom ILE CD1 CD
pdbalias atom HOH OW OH2
pdbalias residue HOH TIP3
# make the segments
foreach S  {  189 185 182} {
	for {set i 1} {$i <= 3} {incr i} {
		set my_path "/media/bak11/KylesStuff/newer_mutants/cis/$S/$i"
		puts $S
		resetpsf
	# retinal
#	segment RET {pdb ret_only.pdb}
#	coordpdb ret_only.pdb RET
	# proteins
		segment PRO {pdb ${my_path}/regen.pdb}
		coordpdb  ${my_path}/regen.pdb PRO
		patch ASPP PRO:96
		regenerate angles dihedrals
		patch GLUP PRO:194
		regenerate angles dihedrals
		patch SCK1 PRO:216 PRO:301
		regenerate angles dihedrals
		guesscoord
		writepsf ${my_path}/regen_ret.psf
		writepdb ${my_path}/regen_ret.pdb
	}
}

exit
