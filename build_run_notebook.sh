!/bin/bash

nvidia-docker build -f Dockerfile.jupyter . -t vaetree/notebook
nvidia-docker run -p 80:8888 -it -v/scratch/:/scratch -v/tmp/:/tmp -v/neuro/:/neuro -v/cryo/:/cryo -v/home/nina/:/home/nina vaetree/notebook
