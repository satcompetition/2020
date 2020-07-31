if [ $# -lt 2 ]; then
	echo "Usage: $0 [results] [total] [penalty]"
	exit 1
fi

if [ $# -eq 3 ]; then
	penalty=$3
else
	penalty=10000
fi

file=$1
total=$2
maxc=$(head -n1 $file | sed 's/,/ /g' | wc -w)

for i in $(seq 3 $maxc); do 
	name=$(head -n1 $file | cut -d, -f$i); 
	times=$(tail -n+2 $file | cut -d, -f$i | xargs); 
	nsolved=$(echo $times | sed 's/ FAILED\| TIMEOUT\| INCORRECT\| UNKNOWN//g' | wc -w); 
	incorrect=$(if [ $(echo $times | grep -v INCORRECT | wc -w) -eq 0 ]; then echo "disqualified"; else echo " "; fi); 
	sum=$(echo $times | sed "s/FAILED\|TIMEOUT\|INCORRECT\|UNKNOWN/$penalty/g" | sed 's/ /+/g'); 

	# par2 fix (sat/unsat tracks, fill 200)
	count=$(echo $sum | sed 's/+/ /g' | wc -w)
	for c in $(seq $((count+1)) $total); do sum="$sum+$penalty"; done
	#echo $sum | sed 's/+/ /g' | wc -w

	par2=$(echo "($sum)/$total" | bc); 
	echo $name $par2 $nsolved $incorrect; 
done
