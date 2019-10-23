#!/bin/bash

MAIN_DIR="/scratch/users/nmiolane/"
for LATENT_DIM in 4 10 40 60 100
    do
        imoutput_dir="${MAIN_DIR}imoutput_connectomes_log_eucl/latent_dim_${LATENT_DIM}"
        echo $imoutput_dir
        if [ ! -d $imoutput_dir ]; then
            mkdir $imoutput_dir
            chmod -R 777 $imoutput_dir
        fi
        python3 ~/code/vaetree/impipeline.py $imoutput_dir ${LATENT_DIM}
done
