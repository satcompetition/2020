if [ $# -lt 1 -o ! -e $1 ]; then
	echo "Usage: $0 [results]"
	exit 0
fi

file=$1

head -n1 $file > $file.sat; 
grep ",SATISFIABLE" $file >> $file.sat

head -n1 $file > $file.unsat; 
grep ",UNSATISFIABLE" $file >> $file.unsat

echo "_______ SAT _______"
./sum.sh $file.sat 200 | tee $file.sat.res | sort -n -k2

echo "_______ UNSAT _______"
./sum.sh $file.unsat 200 | tee $file.unsat.res | sort -n -k2

echo "_______ ALL _______"
./sum.sh $file 400 | tee $file.res | sort -n -k2

