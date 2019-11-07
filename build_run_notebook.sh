#!/bin/bash

nvidia-docker build -f Dockerfile.jupyter . -t vaetree/notebook
nvidia-docker run -p 80:8888 -it \
                             -v/neuro/:/data/neuro \
                             -v/cryo/:/data/cryo \
                             -v/scratch/users/nmiolane/:/results \
                             -v/home/nina/ray_results/:/ray_results \
                             -v/home/nina/code/:/code \
                             -v/tmp/:/tmp \
                             vaetree/notebook
