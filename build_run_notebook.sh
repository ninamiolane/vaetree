#!/bin/bash

nvidia-docker build -f Dockerfile.jupyter . -t vaetree/notebook
nvidia-docker run -p8886:8886 -it -v/scratch/:/scratch -v/home/nina/:/home/nina vaetree/notebook
