#!/bin/bash -l




#BSUB -P cryoem
#BSUB -J vaegan-pipeline
#BSUB -q slacgpu
#BSUB -n 10
#BSUB -R "span[hosts=1]"
#BSUB -W 72:00
#BSUB -e run.err
#BSUB -o run.out
#BSUB -gpu "num=1:mode=exclusive_process:j_exclusive=no:mps=no"
#BSUB -B

# set up env
source /etc/profile.d/modules.sh
export MODULEPATH=/usr/share/Modules/modulefiles:/opt/modulefiles:/afs/slac/package/singularity/modulefiles
module purge
module load PrgEnv-gcc/4.8.5

# change working directory
cd ~/gpfs_home/code/vaetree/

# run the command
singularity run --bind /gpfs,/scratch \
                --bind /gpfs/slac/cryo/fs1/u/nmiolane/data:/data \
                --bind /gpfs/slac/cryo/fs1/u/nmiolane:/home \
                --bind /gpfs/slac/cryo/fs1/u/nmiolane/results:/results \
                --nv ../simgs/toypipeline.simg
