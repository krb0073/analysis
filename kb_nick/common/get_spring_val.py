import sys
run=int(sys.argv[1])
prev= run - 1


if run == 0:
	traj="../../common/min.1.dcd"
	sp = 25
elif run == 1:
	traj="../../common/min.1.dcd"
	sp =25
else:
	traj=f"../{prev}/run.dcd"
	if run <= 13:
		sp = 25
	elif run > 13 and run <= 17:
		sp = 10
	elif run > 17 and run <= 20:
		sp = 5
	elif run > 20 and run <= 24:
		sp = 1
	else:
		sp =0 

print(sp,traj)
