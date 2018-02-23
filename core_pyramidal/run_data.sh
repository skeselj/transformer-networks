#!/bin/bash

#for i in `seq 0 249`
#do
#    python gen_data.py -name=$i -count=20 &
#done

while read F; do
    python gen_data.py -name=$F -count=20 &
done < todo.txt
