package require psfgen
topology /media/bak11/toppar/top_all36_prot.rtf
topology /media/bak11/toppar/retinal-pro.str
topology /media/bak11/toppar/top_all36_carb.rtf
topology /media/bak11/toppar/top_all36_cgenff.rtf
topology /media/bak11/toppar/top_all36_lipid.rtf
topology /media/bak11/toppar/top_all36_na.rtf
topology /media/bak11/toppar/toppar_water_ions_namd.str
pdbalias residue HIS HSE
pdbalias atom ILE CD1 CD
pdbalias atom HOH OW OH2
pdbalias residue HOH TIP3
foreach S { 182 185 189 } {
	resetpsf
	readpsf ret_${S}.psf
	coordpdb cntret_${S}.pdb
	segment WAT {
		auto none
		pdb c_wat.pdb
	}
	coordpdb c_wat.pdb WAT
	guesscoord
	writepsf intact_${S}.psf
	writepdb intact_${S}.pdb
	}
exit
