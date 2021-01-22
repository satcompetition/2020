
for file in $@; do
	#echo " ### Positions of Best Main Track Solvers in $file"
	grep -nH "kissat\|relaxed\|cryptominisat\|cadical\|scavel…\|f2trc" $file
	#echo $(grep -n kissat_sat $file) $file
	#echo $(grep -n relaxed_newtech $file) $file 
	#echo $(grep -n cryptominisat_ccnr_lsids $file) $file
	#echo
	#grep -n kissat_unsat $file
	#grep -n cadical_alluip $file
	#grep -n f2trc $file
	#echo
done
