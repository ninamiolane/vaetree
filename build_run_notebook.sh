#!/bin/bash

docker build -f Dockerfile.jupyter . -t vaetree/notebook
docker run -p8888:8888 -it -v /scratch/:/scratch -v /home/nina/:/home/nina vaetree/notebook
