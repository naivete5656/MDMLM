#!/bin/bash

docker run --runtime=nvidia --rm -it --shm-size 50G -p 8888:8888 --name  pytorch -v $(pwd):/workdir -e PASSWORD=humanif -w /workdir pytorch_jupyter
