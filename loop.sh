number=( 15000 30000 45000 60000 )
bools=( False )

for i in "${number[@]}"
do
    for j in "${bools[@]}"
    do
        for k in {1..10} 
        do
            ./run_network.sh $i $j 2
        done
    done
done