#!/bin/bash

docker build -f Dockerfile.jupyter . -t vaetree/notebook
docker run -p8889:8888 -it -v/scratch/:/scratch -v/home/johmathe:/home/johmathe vaetree/notebook
