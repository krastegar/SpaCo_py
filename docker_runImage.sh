#!/bin/bash

#### Bash command that runs the docker container. #######
sudo docker run -d -p 8787:8787 -v $(pwd):/home/rstudio/ -e PASSWORD=seurat seurat_rstudio-server
