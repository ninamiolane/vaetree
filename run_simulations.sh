#!/bin/bash

MAIN_DIR="/scratch/users/nmiolane/"
for LOGVARX_TRUE in -10 -5 -3.22 -2 -1.02 #-0.45 0
do
    for N in 10000 100000 #50 100  1000 10000 100000
    do
        for MANIFOLD in "s2" #"r2" "s2" "h2"
        do
            toyoutput_dir="${MAIN_DIR}toyoutput_manifold_gvae/logvarx_${LOGVARX_TRUE}_n_${N}_${MANIFOLD}"
            echo $toyoutput_dir
            if [ ! -d $toyoutput_dir ]; then
                mkdir $toyoutput_dir
                chmod -R 777 $toyoutput_dir
            fi
            python3 ~/code/vaetree/toypipeline.py $toyoutput_dir ${LOGVARX_TRUE} ${N} $MANIFOLD
done
done
done
