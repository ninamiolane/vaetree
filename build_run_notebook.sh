#!/bin/bash

nvidia-docker build -f Dockerfile.jupyter . -t vaetree/notebook
nvidia-docker run -p8888:8888 -it -v/scratch/:/scratch -v/tmp/:/tmp -v/neuro/:/neuro -v/home/nina/:/home/nina vaetree/notebook
