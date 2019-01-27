#!/bin/bash

SIMGS_DIR=./simgs
if [ ! -d $SIMGS_DIR ]; then
    mkdir $SIMGS_DIR
fi

# Build base image
img_type=base
cd devops
singularity_file=Singularity
img_name="${img_type}.simg"
if [ ! -f ../$SIMGS_DIR/$img_name ]; then
    echo "Building image $img_name from singularity file $singularity_file"
    sudo -H singularity build ../$SIMGS_DIR/$img_name $singularity_file
fi


# Run pipeline
cd ..
singularity run --nv $SIMGS_DIR/$img_name
