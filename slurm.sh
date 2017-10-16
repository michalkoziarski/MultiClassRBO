#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00

module add plgrid/libs/libpng

python ${1} ${@:2}
