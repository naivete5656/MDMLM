#!/bin/bash

docker run --runtime=nvidia \
     --rm -it \
     -p 8097:8097 -p 8888:8888 \
     --name root \
     -v $(pwd):/workdir \
     -e PASSWORD=password \
     -w /workdir naivete5656/mdmlm bash