number=( 60000 )
bools=( False )

for i in "${number[@]}"
do
    for j in "${bools[@]}"
    do
        python train.py $i $j
        echo "\n"
    done
done