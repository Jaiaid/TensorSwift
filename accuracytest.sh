correct_cnt=0

for i in 100 1000
do
    row=$((1 + $RANDOM % $i))
    col=$((1 + $RANDOM % $i))

    for j in $(seq 1 8)
    do
        printf "\nworking on size ${i} and op ${j} with dims ${row}x${col}:\n"
        test_data=sample-${row}x${col}-${j}.ts
        correct_file=sample-${row}x${col}-${j}.np
        time taskset -c 1-8 python accuracytest.py ${correct_file} ${test_data} ${j} ${row} ${col}
        diff -qEwB ${correct_file} ${test_data} > /dev/null
        res=$?
        if [ "${res}" -eq "0" ]; then
            correct_cnt=$((correct_cnt+1))
        else 
            echo "incorrect on case of size ${i} and op ${j} with dims ${row}x${col}"
        fi
    done
done
printf "correct: ${correct_cnt}\n"
wait
