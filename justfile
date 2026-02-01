set shell := ["sh", "-c"]
set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]
#set allow-duplicate-recipe
#set positional-arguments
set dotenv-filename := ".env"
set export

import? "local.justfile"

RANDOM := env("RANDOM", "42")

init:
  #!/bin/bash
  conda create -n pytorch3d python=3.9
  conda activate pytorch3d
  conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
  conda install -c iopath iopath
  conda install pytorch3d -c pytorch3d
  conda install pytorch3d -c pytorch3d-nightly
  conda install -y jupyterlab notebook ipykernel

