#!/bin/bash

# unset SLURM_* vars
for var in $(compgen -v | grep '^SLURM_'); do unset "$var"; done;

# create links
make link_dirs

