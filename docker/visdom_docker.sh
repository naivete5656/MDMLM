#! /bin/sh
docker run -d --runtime=nvidia --rm -it -p 8888:8098 -p 8080:8097 --name root -v $(pwd):/workdir --shm-size 8G -e PASSWORD=humanif -e ENV_PATH=/workdir -w /workdir pytorch

