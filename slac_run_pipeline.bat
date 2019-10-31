#!/bin/bash -l




#BSUB -P cryoem
#BSUB -J vaegan-pipeline
#BSUB -q slacgpu
#BSUB -n 2
#BSUB -R "span[hosts=1]"
#BSUB -W 72:00
#BSUB -e run.err
#BSUB -o run.out
#BSUB -B

# set up env
source /etc/profile.d/modules.sh
export MODULEPATH=/usr/share/Modules/modulefiles:/opt/modulefiles:/afs/slac/package/singularity/modulefiles
module purge
module load PrgEnv-gcc/4.8.5

# change working directory
cd ~/code/vaetree/

# run the command
singularity run -B /gpfs,/scratch ../simgs/pipeline.simg
